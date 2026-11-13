"""DPO on question twins (runs on the RunPod GPU): same paragraph, a real question vs its edited, unanswerable twin.

Twins come from twins.py (data_dir/twins.jsonl). For a twin (q answerable with gold answer a, q' its unanswerable
edit) the model scores four sequences, all with the exact eval prompt:

  (q, a)   (q, I don't know)   (q', a)   (q', I don't know)

With D(y|x) = log pi(y|x) - log pi_SFT(y|x), a variant's `side` decides how they are paired (2 pairs per twin, loss
-log sigmoid(beta * margin), averaged):

  input   (variant dpo-twin) one response, two questions: the gold answer is preferred after q over after q', and
          "I don't know" after q' over after q.
            margins  D(a|q) - D(a|q')      D(IDK|q') - D(IDK|q)
          Raising or lowering a response by the same amount after both questions leaves the loss unchanged, so the
          only way to lower it is to treat q and q' differently: the change in the question is the whole signal.
          A shift of every question's P(IDK) the same way (a threshold shift: what DPO v1/v2 and GRPO did) cannot
          lower it.
  output  (variant dpo-twin-out, control) standard DPO on the same twins: one question, two responses.
            margins  D(a|q) - D(IDK|q)     D(IDK|q') - D(a|q')
          Raising one response after every question the same way does lower this loss (a threshold shift can).

Both sides use the same four sequences, so the comparison isolates the pairing. "I don't know" is chosen exactly as
often as it is rejected in both.

Training continues the SFT LoRA adapter. The reference is that adapter before training: its log-probs of the four
sequences are computed once (no dropout) and written to results/<name>-train/ref_logps.jsonl. One run per learning
rate; the adapter is saved after each epoch to checkpoints/<name>/<lr tag>/epoch-<n>/.

Probes (step 0, every probe_every optimizer steps, every epoch end): P("I don't know"<|im_end|> | prompt) on every
dev question (means by class, AUC unanswerable vs answerable) and on the held-out twins (AUC, and how often the
unanswerable twin gets the higher P(IDK) than its answerable twin). Per-question values: p_idk.jsonl.

Selection: best (lr, epoch) on dev balanced score; test runs only if it passes the GRPO dev gate against SFT
(dpo_twin.gate: paired p < max_p, or >= min_gain with more questions gained than lost).

Usage (on the pod):
  python dpo_twin.py --variant dpo-twin --dry-run    # encode, checks, reference log-probs, one probe; no training
  python dpo_twin.py --variant dpo-twin              # train every learning rate
  python dpo_twin.py --variant dpo-twin --select     # after the dev evals; exit 0 if it passes the gate, 10 if not
"""

import argparse
import json
import shutil
import time
from collections import defaultdict

from config import add_config_arg, load_config
from dpo import IdkProbe, lr_tag
from grader import ABSTAIN

END = "<|im_end|>"
SIDES = ("input", "output")
# Order of a twin's four sequences.
Q_ANS, Q_IDK, U_ANS, U_IDK = range(4)
SEQ_NAMES = ("q_answer", "q_idk", "u_answer", "u_idk")
FAILS_GATE = 10


def variant(cfg, name):
    if name not in cfg["dpo_twin"]["variants"]:
        raise SystemExit(f"unknown variant {name!r}; config has {list(cfg['dpo_twin']['variants'])}")
    v = {**cfg["dpo_twin"]["variants"][name], "name": name}
    assert v["side"] in SIDES, v
    return v


def candidates(cfg, name):
    v = variant(cfg, name)
    return [(lr_tag(lr), e) for lr in v["learning_rates"] for e in range(1, cfg["dpo_twin"]["epochs"] + 1)]


def dev_run_name(name, tag, epoch):
    return f"{name}-dev-{tag}-e{epoch}"


def margins(delta, side):
    """delta[..., 4]: log pi - log pi_ref of (q, a), (q, IDK), (q', a), (q', IDK). The two DPO margins per twin
    (chosen minus rejected), named by the response for side=input and by the question for side=output."""
    if side == "input":
        return {"answer": delta[..., Q_ANS] - delta[..., U_ANS], "idk": delta[..., U_IDK] - delta[..., Q_IDK]}
    if side == "output":
        return {"answerable": delta[..., Q_ANS] - delta[..., Q_IDK], "unanswerable": delta[..., U_IDK] - delta[..., U_ANS]}
    raise ValueError(side)


def pair_records(twin, side):
    """The two preference pairs of a twin, readable (pairs.jsonl)."""
    q, u, a = twin["answerable"]["question"], twin["unanswerable"]["question"], twin["answerable"]["answer"]
    if side == "input":
        return [{"twin_id": twin["twin_id"], "response": a, "chosen_question": q, "rejected_question": u},
                {"twin_id": twin["twin_id"], "response": ABSTAIN, "chosen_question": u, "rejected_question": q}]
    return [{"twin_id": twin["twin_id"], "question": q, "chosen": a, "rejected": ABSTAIN},
            {"twin_id": twin["twin_id"], "question": u, "chosen": ABSTAIN, "rejected": a}]


def encode_twin(cfg, tok, twin):
    from prompts import render
    ctx = twin["context"]
    prompts = [tok(render(cfg, tok, {"context": ctx, "question": twin[k]["question"]}),
                   add_special_tokens=False)["input_ids"] for k in ("answerable", "unanswerable")]
    comps = [tok(x + END, add_special_tokens=False)["input_ids"] for x in (twin["answerable"]["answer"], ABSTAIN)]
    for x, c in zip((twin["answerable"]["answer"], ABSTAIN), comps):
        assert tok.decode(c) == x + END and c[-1] == tok.eos_token_id, (x, tok.decode(c))
    seqs = [(prompts[0], comps[0]), (prompts[0], comps[1]), (prompts[1], comps[0]), (prompts[1], comps[1])]
    return {"twin_id": twin["twin_id"], "ids": [p + c for p, c in seqs], "clen": [len(c) for _, c in seqs]}


def collate(items, pad_id):
    """B twins -> 4B left-padded sequences; completion_mask covers each completion inside the last K positions."""
    import torch
    seqs = [s for it in items for s in it["ids"]]
    clens = [c for it in items for c in it["clen"]]
    L, K = max(map(len, seqs)), max(clens)
    batch = {"input_ids": torch.tensor([[pad_id] * (L - len(s)) + s for s in seqs]),
             "attention_mask": torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs]),
             "completion_mask": torch.tensor([[0] * (K - c) + [1] * c for c in clens])}
    if "ref" in items[0]:
        batch["ref"] = torch.tensor([it["ref"] for it in items], dtype=torch.float32)
    return batch


def seq_logps(model, batch):
    """Summed log-prob of each sequence's completion tokens, shape [4B]."""
    import torch
    ids, mask, cmask = batch["input_ids"], batch["attention_mask"], batch["completion_mask"]
    K = cmask.shape[1]
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=K + 1)
    lp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
    tok_lp = lp.gather(-1, ids[:, -K:].unsqueeze(-1)).squeeze(-1)
    return (tok_lp * cmask).sum(-1)


def all_logps(model, items, pad_id, batch_size=4):
    """[n twins][4] completion log-probs, eval mode, no grad, bf16 autocast (as in training)."""
    import torch
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(items), batch_size):
            b = {k: v.to(dev) for k, v in collate(items[i:i + batch_size], pad_id).items()}
            out += seq_logps(model, b).view(-1, 4).tolist()
    if was_training:
        model.train()
    return out


def twin_choices(lps):
    """How often the model's own log-probs prefer the right response after each question of a twin."""
    n = len(lps)
    on_q = sum(x[Q_ANS] > x[Q_IDK] for x in lps)
    on_u = sum(x[U_IDK] > x[U_ANS] for x in lps)
    both = sum(x[Q_ANS] > x[Q_IDK] and x[U_IDK] > x[U_ANS] for x in lps)
    return {"n": n, "prefers_answer_on_real": round(on_q / n, 4), "prefers_idk_on_edited": round(on_u / n, 4),
            "both": round(both / n, 4)}


def heldout_twin_stats(p, n):
    """p: P(IDK) on twins-heldout-rows (n answerable twin questions, then their n unanswerable twins)."""
    from metrics import auc
    pa, pu = p[:n], p[n:]
    return {"auc": round(auc(pa, pu), 4), "unanswerable_higher": round(sum(u > a for a, u in zip(pa, pu)) / n, 4),
            "both_right_at_0.5": round(sum(u > 0.5 and a <= 0.5 for a, u in zip(pa, pu)) / n, 4),
            "mean_p_answerable": round(sum(pa) / n, 4), "mean_p_unanswerable": round(sum(pu) / n, 4)}


def make_trainer(side, beta, **kw):
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    class TwinDPOTrainer(Trainer):
        def __init__(self, **kw):
            super().__init__(**kw)
            # The loss is a mean per micro-batch: let Trainer divide it by the accumulation steps.
            self.model_accepts_loss_kwargs = False
            self.loss_is_scaled_for_ga = False
            self._stats = defaultdict(list)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            lp = seq_logps(model, inputs).view(-1, 4)
            delta = lp - inputs["ref"]
            ms = margins(delta, side)
            losses = {k: -F.logsigmoid(beta * m) for k, m in ms.items()}
            loss = torch.stack(list(losses.values()), -1).mean()
            with torch.no_grad():
                for k, m in ms.items():
                    self._stats[f"margin/{k}"].append((beta * m).mean().item())
                    self._stats[f"accuracy/{k}"].append((m > 0).float().mean().item())
                    self._stats[f"loss/{k}"].append(losses[k].mean().item())
                for i, s in enumerate(SEQ_NAMES):
                    self._stats[f"logp/{s}"].append(lp[:, i].mean().item())
                    self._stats[f"delta/{s}"].append(delta[:, i].mean().item())
                self._stats["policy_right/real"].append((lp[:, Q_ANS] > lp[:, Q_IDK]).float().mean().item())
                self._stats["policy_right/edited"].append((lp[:, U_IDK] > lp[:, U_ANS]).float().mean().item())
            return loss

        def log(self, logs, start_time=None):
            if self._stats:
                logs = {**logs, **{k: round(sum(v) / len(v), 5) for k, v in self._stats.items()}}
                self._stats.clear()
            super().log(logs, start_time)

    return TwinDPOTrainer(**kw)


def train(cfg, name, dry_run=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments

    from metrics import auc

    d, v = cfg["dpo_twin"], variant(cfg, name)
    model_id = cfg.model_id("main")
    sft_dir = cfg.checkpoints_dir / "sft" / "best"
    out = cfg.checkpoints_dir / name
    res = cfg.results_dir / f"{name}-train"
    res.mkdir(parents=True, exist_ok=True)

    twins = [json.loads(l) for l in open(cfg.data_dir / "twins.jsonl")]
    heldout = [json.loads(l) for l in open(cfg.data_dir / "twins-heldout-rows.jsonl")]
    n_held = len(heldout) // 2
    dev_rows = [json.loads(l) for l in open(cfg.data_dir / "dev.jsonl")]
    with open(res / "pairs.jsonl", "w") as f:
        for t in twins:
            for p in pair_records(t, v["side"]):
                f.write(json.dumps(p) + "\n")

    tok = AutoTokenizer.from_pretrained(model_id)
    assert tok.eos_token == END, tok.eos_token
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    items = [encode_twin(cfg, tok, t) for t in twins]
    lens = [len(s) for it in items for s in it["ids"]]
    print(f"{len(twins)} twins -> {2 * len(twins)} pairs ({v['side']} side); tokens/sequence max {max(lens)}, "
          f"mean {sum(lens) / len(lens):.0f} (max_length {d['max_length']})")
    assert max(lens) < d["max_length"]
    for it, t in list(zip(items, twins))[:2]:
        print(f"twin {t['twin_id']}:")
        for s, c, nm in zip(it["ids"], it["clen"], SEQ_NAMES):
            print(f"  {nm:<9} ...{tok.decode(s[-c - 14:-c])!r} | {tok.decode(s[-c:])!r}")

    dev_probe = IdkProbe(cfg, tok, dev_rows)
    twin_probe = IdkProbe(cfg, tok, heldout)
    dev_ans = [r["answerable"] for r in dev_rows]

    summaries, ref = {}, None
    for lr in v["learning_rates"]:
        tag = lr_tag(lr)
        torch.manual_seed(d["seed"])
        base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, sft_dir, is_trainable=True).to("cuda")
        if ref is None:  # the SFT adapter before any update: identical for every learning rate
            t0 = time.time()
            ref = all_logps(model, items, pad)
            with open(res / "ref_logps.jsonl", "w") as f:
                for t, x in zip(twins, ref):
                    f.write(json.dumps({"twin_id": t["twin_id"], **dict(zip(SEQ_NAMES, x))}) + "\n")
            print(f"reference log-probs ({time.time() - t0:.0f}s); SFT on the training twins: "
                  f"{json.dumps(twin_choices(ref))}")
            again = all_logps(model, items[:8], pad)
            drift = max(abs(a - b) for x, y in zip(again, ref[:8]) for a, b in zip(x, y))
            assert drift < 1e-3, f"reference log-probs are not reproducible (max diff {drift})"
        data = [{**it, "ref": r} for it, r in zip(items, ref)]
        log, probes, p_idk = [], [], []

        def probe(step, epoch, model, epoch_end=None):
            t0 = time.time()
            dv = dev_probe(model)
            dp = dev_probe.last_p
            dv["auc"] = round(auc([p for p, a in zip(dp, dev_ans) if a], [p for p, a in zip(dp, dev_ans) if not a]), 4)
            twin_probe(model)
            ht = heldout_twin_stats(twin_probe.last_p, n_held)
            rec = {"lr": tag, "step": step, "epoch": round(epoch or 0, 3), "epoch_end": epoch_end,
                   "dev": dv, "heldout_twins": ht}
            probes.append(rec)
            p_idk.append({"lr": tag, "step": step, "epoch_end": epoch_end,
                          "dev": {r["id"]: p for r, p in zip(dev_rows, dp)},
                          "heldout_twins": {r["id"]: p for r, p in zip(heldout, twin_probe.last_p)}})
            print(f"[probe {tag} step {step}] dev P(IDK) answerable {dv['answerable']['mean_p']:.3f} unanswerable "
                  f"{dv['unanswerable']['mean_p']:.3f} AUC {dv['auc']:.4f} | held-out twins AUC {ht['auc']:.4f} "
                  f"unanswerable higher {ht['unanswerable_higher']:.3f} both right {ht['both_right_at_0.5']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        class Callbacks(TrainerCallback):
            last = -1

            def on_train_begin(self, args, state, control, model=None, **kw):
                probe(0, 0, model)
                self.last = 0

            def on_step_end(self, args, state, control, model=None, **kw):
                if state.global_step % d["probe_every"] == 0 and state.global_step != self.last:
                    probe(state.global_step, state.epoch, model)
                    self.last = state.global_step

            def on_epoch_end(self, args, state, control, model=None, **kw):
                e = round(state.epoch)
                probe(state.global_step, state.epoch, model, epoch_end=e)
                self.last = state.global_step
                dst = out / tag / f"epoch-{e}"
                model.save_pretrained(dst)
                tok.save_pretrained(dst)
                print(f"saved {dst}", flush=True)

            def on_log(self, args, state, control, logs=None, **kw):
                log.append({"step": state.global_step, "epoch": state.epoch, **(logs or {})})

        args = TrainingArguments(
            output_dir=str(out / "trainer"),
            num_train_epochs=d["epochs"],
            learning_rate=lr,
            per_device_train_batch_size=d["per_device_batch_size"],
            gradient_accumulation_steps=d["grad_accum_steps"],
            lr_scheduler_type=d["lr_scheduler"],
            warmup_steps=d["warmup_ratio"],  # transformers 5: a float < 1 is a ratio of total steps
            bf16=True,
            gradient_checkpointing=True,  # as TRL's DPOConfig default (the DPO v1/v2 runs); OOMs on 24 GB without it
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            seed=d["seed"],
        )
        trainer = make_trainer(v["side"], d["beta"], model=model, args=args, train_dataset=data,
                               data_collator=lambda b: collate(b, pad), callbacks=[Callbacks()])
        trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in n for n in trainable), trainable[:5]
        trainer.model.print_trainable_parameters()
        if dry_run:
            batch = {k: x.to("cuda") for k, x in collate(data[:2], pad).items()}
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = trainer.compute_loss(model, batch)
            print(f"dry run: loss on 2 twins {loss.item():.4f} (log 2 = 0.6931 at the reference, up to dropout); "
                  f"{json.dumps({k: x[-1] for k, x in trainer._stats.items()})}")
            probe(0, 0, model)
            print("dry run: stopping before training")
            return

        t0 = time.time()
        trainer.train()
        secs = time.time() - t0
        shutil.rmtree(out / "trainer", ignore_errors=True)
        after = all_logps(model, items, pad)
        summaries[tag] = {"learning_rate": lr, "train_seconds": round(secs), "steps": trainer.state.global_step,
                          "training_twins_at_end": twin_choices(after)}
        for fname, rows in ((f"train_log_{tag}.jsonl", log), (f"probes_{tag}.jsonl", probes),
                            (f"p_idk_{tag}.jsonl", p_idk)):
            with open(res / fname, "w") as f:
                for x in rows:
                    f.write(json.dumps(x) + "\n")
        del trainer, model, base
        torch.cuda.empty_cache()

    summary = {"variant": name, "side": v["side"], "model": model_id, "start_adapter": str(sft_dir),
               "trainer": "dpo_twin.TwinDPOTrainer (transformers.Trainer)", "lora": cfg["lora"],
               "dpo_twin": {k: x for k, x in d.items() if k != "variants"}, "variant_config": v,
               "twins": len(twins), "pairs": 2 * len(twins), "heldout_twins": n_held,
               "training_twins_at_sft": twin_choices(ref), "runs": summaries,
               "eval_fingerprint": cfg.eval_fingerprint(), "gpu": torch.cuda.get_device_name(0)}
    json.dump(summary, open(res / "train_summary.json", "w"), indent=2)
    print(f"wrote {res}/train_summary.json, train_log_*, probes_*, p_idk_*, pairs.jsonl, ref_logps.jsonl")


def select(cfg, name) -> bool:
    from grpo import gate_passes
    from metrics import paired_auc_compare, paired_compare

    d = cfg["dpo_twin"]
    res = cfg.results_dir / f"{name}-train"
    scores = {}
    for tag, e in candidates(cfg, name):
        m = json.load(open(cfg.results_dir / dev_run_name(name, tag, e) / "metrics.json"))
        scores[(tag, e)] = m["balanced_score"]["value"]
        print(f"{tag} epoch {e}: dev balanced {m['balanced_score']['value']:.3f}  halluc "
              f"{m['hallucination_rate']['value']:.3f}  over-refusal {m['over_refusal_rate']['value']:.3f}  "
              f"accuracy {m['answer_accuracy']['value']:.3f}")
    order = list(scores)  # ties go to the earlier entry: lower lr, then earlier epoch
    tag, e = max(order, key=lambda k: (scores[k], -order.index(k)))
    out = cfg.checkpoints_dir / name
    shutil.rmtree(out / "best", ignore_errors=True)
    shutil.copytree(out / tag / f"epoch-{e}", out / "best")

    gate = d["gate"]
    load = lambda run: [json.loads(l) for l in open(cfg.results_dir / run / "outputs.jsonl")]
    c = paired_compare(load(gate["reference_dev_run"]), load(dev_run_name(name, tag, e)),
                       n_boot=cfg["eval"]["bootstrap_samples"], seed=cfg["eval"]["seed"])["balanced_score"]
    ok = gate_passes(c["diff"], c["mcnemar_p"], c["only_b"], c["only_a"], gate["max_p"], gate["min_gain"])

    recs = [json.loads(l) for l in open(res / f"p_idk_{tag}.jsonl")]
    sft = next(r for r in recs if r["step"] == 0)
    best = next(r for r in recs if r["epoch_end"] == e)
    dev_ans = {x["id"]: x["answerable"] for x in map(json.loads, open(cfg.data_dir / "dev.jsonl"))}
    split = lambda p, a: [p[i] for i in sorted(p) if dev_ans[i] == a]
    ac = paired_auc_compare(split(sft["dev"], True), split(sft["dev"], False), split(best["dev"], True),
                            split(best["dev"], False), n_boot=cfg["eval"]["bootstrap_samples"], seed=cfg["eval"]["seed"])
    held = [json.loads(l)["id"] for l in open(cfg.data_dir / "twins-heldout-rows.jsonl")]
    n = len(held) // 2
    ht = {k: heldout_twin_stats([r["heldout_twins"][i] for i in held], n) for k, r in (("sft", sft), ("best", best))}
    print(f"P(IDK) AUC on dev: SFT {ac['a']:.4f}  {tag}/epoch-{e} {ac['b']:.4f}  diff {ac['diff']:+.4f}  paired CI "
          f"[{ac['ci95'][0]:+.4f}, {ac['ci95'][1]:+.4f}]")
    print(f"held-out twins: SFT {json.dumps(ht['sft'])}\n                best {json.dumps(ht['best'])}")

    (cfg.results_dir / "idk-probe").mkdir(exist_ok=True)
    json.dump({"label": name, "model": cfg.model_id("main"), "adapter": f"checkpoints/{name}/best", "split": "dev",
               "source": f"{name}-train/p_idk_{tag}.jsonl epoch_end {e}", "p_idk": best["dev"]},
              open((cfg.results_dir / "idk-probe").joinpath(f"{name}.json"), "w"), indent=1)

    summary = json.load(open(res / "train_summary.json"))
    summary.update({
        "dev_balanced": {f"{t}/epoch-{ep}": s for (t, ep), s in scores.items()}, "best": f"{tag}/epoch-{e}",
        "best_dev_run": dev_run_name(name, tag, e),
        "gate": {**gate, "diff": c["diff"], "ci95": c["ci95"], "mcnemar_p": c["mcnemar_p"], "gained": c["only_b"],
                 "lost": c["only_a"], "passes": ok},
        "p_idk_auc_dev": {"sft": ac["a"], "best": ac["b"], "diff": ac["diff"], "ci95": ac["ci95"]},
        "heldout_twins": ht})
    json.dump(summary, open(res / "train_summary.json", "w"), indent=2)
    print(f"best {tag}/epoch-{e} -> {out / 'best'}: dev balanced {scores[(tag, e)]:.3f} vs "
          f"{gate['reference_dev_run']} {c['a']:.3f}  diff {100 * c['diff']:+.1f} pts  paired CI "
          f"[{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}]  gained {c['only_b']} lost {c['only_a']}  "
          f"McNemar p {c['mcnemar_p']:.3g}  -> {'PASSES the gate' if ok else 'does NOT pass the gate'}")
    return ok


def main():
    p = add_config_arg(argparse.ArgumentParser())
    p.add_argument("--variant", required=True, help="a key of dpo_twin.variants in config.json")
    p.add_argument("--select", action="store_true", help="pick the best (lr, epoch) from the dev evals, apply the gate")
    p.add_argument("--dry-run", action="store_true", help="encode, reference log-probs, one probe; don't train")
    p.add_argument("--candidates", action="store_true", help="print '<lr tag> <epoch>' per trained adapter")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.candidates:
        print("\n".join(f"{t} {e}" for t, e in candidates(cfg, args.variant)))
    elif args.select:
        raise SystemExit(0 if select(cfg, args.variant) else FAILS_GATE)
    else:
        train(cfg, args.variant, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
