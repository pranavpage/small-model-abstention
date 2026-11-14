"""Stage 3: DPO on top of the best SFT adapter (runs on the RunPod GPU).

Every DPO variant is named in config.json under dpo.variants (pair_mode + learning_rates); shared settings sit in
dpo.*. A variant's outputs never overwrite another's: checkpoints/<name>/, results/<name>-train/,
dev runs <name>-dev-<lr tag>-e<n>, test run <name>.

Candidate pairs come from results/<dpo.samples_run>/samples.jsonl (sample.py: the SFT model sampled on the
dpo split):
  unanswerable  chosen "I don't know"                      rejected  each distinct hallucinated sample
  answerable    chosen the model's most common sample that  rejected  each distinct sampled "I don't know", and
                is strict-correct (token F1 >= 0.5) and               each distinct wrong sample that is NOT a
                <= max_answer_words; else the shortest                partial span of a gold answer (those are
                gold span if <= max_answer_words; else the            often fine answers the grader marks down,
                question is skipped                                   e.g. "ADSL-broadband" for "ADSL")
pair_mode decides which of them are kept:
  class_balanced  (variant "dpo", the first one tried) answerable pairs subsampled to the number of unanswerable
                  pairs. "I don't know" ended up chosen 279 times and rejected 172 times, a net push towards
                  abstaining; the trained model refused more without discriminating better.
  idk_balanced    (variant "dpo-v2") "I don't know" is chosen exactly as often as it is rejected: unanswerable
                  pairs and answerable pairs with a rejected "I don't know" are both subsampled to the smaller
                  count; every answerable correct-vs-wrong pair (no "I don't know" on either side) is kept.

Training continues the SFT LoRA adapter (loaded trainable). TRL copies it into a frozen "ref" adapter, so the
DPO reference is the SFT model, not the base model. One run per learning rate; the trained "default" adapter is
saved after each epoch to checkpoints/<name>/<lr tag>/epoch-<n>/.

During training, P("I don't know" + <|im_end|> | prompt) under the policy is logged on every dev question
(answerable and unanswerable separately) at step 0 and every dpo.idk_probe_every optimizer steps.

Usage (on the pod):
  python -m abstention.dpo --variant dpo-v2 --pairs-only     # build and print the pair set, no GPU
  python -m abstention.dpo --variant dpo-v2 --dry-run        # build the trainer, run the checks and one probe, don't train
  python -m abstention.dpo --variant dpo-v2                  # train every learning rate
  python -m abstention.dpo --variant dpo-v2 --select         # after the dev evals: best (lr, epoch) -> checkpoints/<name>/best
                                                  # exit 0 if it beats SFT on dev balanced score, 10 if not
"""

import argparse
import hashlib
import json
import random
import shutil
import statistics
import time
from collections import Counter

from abstention.config import add_config_arg, load_config
from abstention.grader import ABSTAIN, max_f1, normalize

END = "<|im_end|>"
SFT_DEV_RUN = "sft-dev-e2"  # the selected SFT epoch on dev; DPO must beat its balanced score to reach test
NOT_BETTER_THAN_SFT = 10


def lr_tag(lr: float) -> str:
    return f"lr{lr:g}"


def variant(cfg, name):
    if name not in cfg["dpo"]["variants"]:
        raise SystemExit(f"unknown DPO variant {name!r}; config has {list(cfg['dpo']['variants'])}")
    return {**cfg["dpo"]["variants"][name], "name": name}


def candidates(cfg, name):
    """Every (lr tag, epoch) trained for a variant, in sweep order."""
    v = variant(cfg, name)
    return [(lr_tag(lr), e) for lr in v["learning_rates"] for e in range(1, cfg["dpo"]["epochs"] + 1)]


def is_partial_span(output: str, golds: list[str]) -> bool:
    o = normalize(output)
    return bool(o) and any(o in normalize(g) or normalize(g) in o for g in golds if normalize(g))


def build_pairs(samples, max_answer_words, seed, mode="class_balanced"):
    """Preference pairs from sample.py rows. Returns (pairs, stats)."""
    ans, unans, stats = [], [], Counter()
    for r in samples:
        outs = r["samples"]
        if not r["answerable"]:
            rejected = list(dict.fromkeys(s["output"] for s in outs if s["cell"] == "hallucination"))
            unans += [{"id": r["id"], "answerable": False, "chosen": ABSTAIN, "rejected": x,
                       "chosen_source": "abstain", "rejected_cell": "hallucination"} for x in rejected]
            continue
        good = [s["output"] for s in outs if s["cell"] == "correct" and max_f1(s["output"], r["gold"]) >= 0.5
                and len(s["output"].split()) <= max_answer_words]
        rejected = []
        for s in outs:
            if s["cell"] == "over_refusal":
                rejected.append((s["output"], "over_refusal"))
            elif s["cell"] == "wrong":
                if is_partial_span(s["output"], r["gold"]):
                    stats["dropped_partial_span_rejected"] += 1
                else:
                    rejected.append((s["output"], "wrong"))
        rejected = list(dict.fromkeys(rejected))
        if not rejected:
            continue
        gold = min(r["gold"], key=len)
        if good:
            chosen, source = Counter(good).most_common(1)[0][0], "sampled"
        elif len(gold.split()) <= max_answer_words:
            chosen, source = gold, "gold"
        else:
            stats["dropped_long_gold_questions"] += 1
            continue
        ans += [{"id": r["id"], "answerable": True, "chosen": chosen, "rejected": x,
                 "chosen_source": source, "rejected_cell": cell} for x, cell in rejected]

    stats["answerable_pairs_before_balancing"] = len(ans)
    stats["unanswerable_pairs_before_balancing"] = len(unans)
    rng = random.Random(seed)
    if mode == "class_balanced":  # subsample only the larger side (keeps the first variant's pairs byte-identical)
        if len(ans) > len(unans):
            ans = rng.sample(ans, len(unans))
        elif len(unans) > len(ans):
            unans = rng.sample(unans, len(ans))
    elif mode == "idk_balanced":
        refused = [p for p in ans if p["rejected"] == ABSTAIN]
        wrong = [p for p in ans if p["rejected"] != ABSTAIN]
        n = min(len(refused), len(unans))
        ans, unans = rng.sample(refused, n) + wrong, rng.sample(unans, n)
    else:
        raise ValueError(f"unknown pair_mode {mode!r}")
    pairs = ans + unans
    rng.shuffle(pairs)
    stats = dict(stats)
    stats.update({
        "pair_mode": mode, "pairs": len(pairs), "answerable_pairs": len(ans), "unanswerable_pairs": len(unans),
        "questions": len({p["id"] for p in pairs}),
        "idk_chosen": sum(p["chosen"] == ABSTAIN for p in pairs),
        "idk_rejected": sum(p["rejected"] == ABSTAIN for p in pairs),
        "answerable_rejected": dict(Counter(p["rejected_cell"] for p in ans)),
        "answerable_chosen_source": dict(Counter(p["chosen_source"] for p in ans)),
    })
    return pairs, stats


def load_pairs(cfg, name):
    d = cfg["dpo"]
    src = cfg.results_dir / d["samples_run"] / "samples.jsonl"
    raw = src.read_text()
    samples = [json.loads(l) for l in raw.splitlines()]
    pairs, stats = build_pairs(samples, d["max_answer_words"], d["seed"], variant(cfg, name)["pair_mode"])
    stats["samples_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    return pairs, stats


class IdkProbe:
    """P("I don't know<|im_end|>" | prompt) under the current policy for a fixed set of questions."""

    def __init__(self, cfg, tok, rows, batch_size=8):
        from abstention.prompts import render
        self.tok, self.bs = tok, batch_size
        self.answerable = [r["answerable"] for r in rows]
        self.prompts, completion = [], None
        for r in rows:
            p = render(cfg, tok, r)
            p_ids = tok(p, add_special_tokens=False)["input_ids"]
            full = tok(p + ABSTAIN + END, add_special_tokens=False)["input_ids"]
            assert full[:len(p_ids)] == p_ids, "prompt is not a token prefix of prompt + abstain"
            c = full[len(p_ids):]
            assert completion in (None, c), "abstain tokenizes differently after different prompts"
            completion = c
            self.prompts.append(p_ids)
        assert tok.decode(completion) == ABSTAIN + END
        self.completion = completion

    def __call__(self, model):
        import torch
        was_training = model.training
        model.eval()
        dev = next(model.parameters()).device
        c = torch.tensor(self.completion, device=dev)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        logps = []
        with torch.no_grad():
            for i in range(0, len(self.prompts), self.bs):
                seqs = [p + self.completion for p in self.prompts[i:i + self.bs]]
                L = max(map(len, seqs))
                ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=dev)  # left padding
                mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                            logits_to_keep=len(self.completion) + 1)
                lp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
                logps += lp.gather(-1, c.expand(len(seqs), -1).unsqueeze(-1)).squeeze(-1).sum(-1).tolist()
        if was_training:
            model.train()
        import math
        self.last_p = [math.exp(x) for x in logps]  # per question, in row order
        res = {}
        for name, flag in (("answerable", True), ("unanswerable", False)):
            ps = [math.exp(x) for x, a in zip(logps, self.answerable) if a == flag]
            res[name] = {"mean_p": round(sum(ps) / len(ps), 4), "median_p": round(statistics.median(ps), 4),
                         "frac_p_gt_0.5": round(sum(p > 0.5 for p in ps) / len(ps), 4), "n": len(ps)}
        return res


def train(cfg, name, dry_run=False):
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    from abstention.prompts import render

    d, v = cfg["dpo"], variant(cfg, name)
    model_id = cfg.model_id("main")
    sft_dir = cfg.checkpoints_dir / "sft" / "best"
    out = cfg.checkpoints_dir / name
    res = cfg.results_dir / f"{name}-train"
    res.mkdir(parents=True, exist_ok=True)

    pairs, stats = load_pairs(cfg, name)
    print(json.dumps(stats, indent=2))
    with open(res / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    tok = AutoTokenizer.from_pretrained(model_id)
    assert tok.eos_token == END, tok.eos_token
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(cfg.data_dir / "dpo.jsonl")}
    ds = Dataset.from_list([{"prompt": render(cfg, tok, rows[p["id"]]), "chosen": p["chosen"],
                             "rejected": p["rejected"]} for p in pairs])
    probe = IdkProbe(cfg, tok, [json.loads(l) for l in open(cfg.data_dir / "dev.jsonl")])

    summaries = {}
    for lr in v["learning_rates"]:
        tag = lr_tag(lr)
        torch.manual_seed(d["seed"])
        base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, sft_dir, is_trainable=True)
        log, probes = [], []

        class Callbacks(TrainerCallback):
            def _probe(self, state, model):
                p = {"step": state.global_step, "epoch": round(state.epoch or 0, 3), **probe(model)}
                probes.append(p)
                print(f"idk probe step {p['step']}: answerable mean P {p['answerable']['mean_p']:.3f} "
                      f"(>0.5: {p['answerable']['frac_p_gt_0.5']:.2f})  unanswerable mean P "
                      f"{p['unanswerable']['mean_p']:.3f} (>0.5: {p['unanswerable']['frac_p_gt_0.5']:.2f})")

            def on_train_begin(self, args, state, control, model=None, **kw):
                self._probe(state, model)

            def on_step_end(self, args, state, control, model=None, **kw):
                if state.global_step % d["idk_probe_every"] == 0 or state.global_step == state.max_steps:
                    self._probe(state, model)

            def on_epoch_end(self, args, state, control, model=None, **kw):
                dst = out / tag / f"epoch-{round(state.epoch)}"
                model.save_pretrained(dst, selected_adapters=["default"])
                tok.save_pretrained(dst)
                print(f"saved {dst}")

            def on_log(self, args, state, control, logs=None, **kw):
                log.append({"step": state.global_step, "epoch": state.epoch, **(logs or {})})

        args = DPOConfig(
            output_dir=str(out / "trainer"),
            beta=d["beta"],
            num_train_epochs=d["epochs"],
            learning_rate=lr,
            per_device_train_batch_size=d["per_device_batch_size"],
            gradient_accumulation_steps=d["grad_accum_steps"],
            lr_scheduler_type=d["lr_scheduler"],
            warmup_steps=d["warmup_ratio"],  # transformers 5: a float < 1 is a ratio of total steps
            max_length=d["max_length"],
            bf16=True,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            seed=d["seed"],
        )
        cb = Callbacks()
        trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, callbacks=[cb])
        pm = trainer.model
        assert "ref" in pm.peft_config, "TRL did not create the frozen SFT reference adapter"
        trainable = [n for n, p in pm.named_parameters() if p.requires_grad]
        assert trainable and not any(".ref." in n for n in trainable), "reference adapter is trainable"
        pm.print_trainable_parameters()

        # Check what TRL built: prompt = the eval prompt tokenized alone; completion = target + <|im_end|>.
        td = trainer.train_dataset
        lens = [len(x["prompt_ids"]) + max(len(x["chosen_ids"]), len(x["rejected_ids"])) for x in td]
        print(f"tokens/example: max {max(lens)}, mean {sum(lens) / len(lens):.0f} (max_length {d['max_length']})")
        assert max(lens) < d["max_length"], "an example hit max_length and was truncated"
        assert len(td) == len(ds), f"TRL dropped {len(ds) - len(td)} examples"
        for i in range(3):
            ex, t = ds[i], td[i]
            assert t["prompt_ids"] == tok(ex["prompt"], add_special_tokens=False)["input_ids"], f"prompt mismatch at {i}"
            c, r = tok.decode(t["chosen_ids"]), tok.decode(t["rejected_ids"])
            assert c == ex["chosen"] + END and r == ex["rejected"] + END, (c, r)
            print(f"example {i}: ...{tok.decode(t['prompt_ids'][-8:])!r} | chosen {c!r} | rejected {r!r}")
        if dry_run:
            pm.to(trainer.accelerator.device)
            t0 = time.time()
            print("probe at SFT (no training):", json.dumps(probe(pm)), f"({time.time() - t0:.0f}s)")
            print("dry run: stopping before training")
            return

        ref_before = {n: p.detach().clone() for n, p in pm.named_parameters() if ".ref." in n}
        t0 = time.time()
        trainer.train()
        secs = time.time() - t0
        assert all(torch.equal(p, ref_before[n]) for n, p in pm.named_parameters() if ".ref." in n), "ref adapter moved"
        shutil.rmtree(out / "trainer", ignore_errors=True)
        summaries[tag] = {"learning_rate": lr, "train_seconds": round(secs)}
        for fname, items in ((f"train_log_{tag}.jsonl", log), (f"idk_probe_{tag}.jsonl", probes)):
            with open(res / fname, "w") as f:
                for x in items:
                    f.write(json.dumps(x) + "\n")
        del trainer, model, base, pm
        torch.cuda.empty_cache()

    summary = {"variant": name, "model": model_id, "start_adapter": str(sft_dir), "trainer": "trl.DPOTrainer",
               "lora": cfg["lora"], "dpo": {k: x for k, x in d.items() if k != "variants"}, "variant_config": v,
               "pairs": stats, "runs": summaries, "eval_fingerprint": cfg.eval_fingerprint(),
               "gpu": torch.cuda.get_device_name(0)}
    json.dump(summary, open(res / "train_summary.json", "w"), indent=2)
    print(f"wrote {res}/train_summary.json, train_log_*.jsonl, idk_probe_*.jsonl, pairs.jsonl")


def select(cfg, name) -> bool:
    """Best (lr, epoch) on dev balanced score (runs <name>-dev-<tag>-e<n>) -> checkpoints/<name>/best.
    Returns whether it beats SFT's dev balanced score (strictly)."""
    out = cfg.checkpoints_dir / name
    scores = {}
    for tag, e in candidates(cfg, name):
        m = json.load(open(cfg.results_dir / f"{name}-dev-{tag}-e{e}" / "metrics.json"))
        scores[f"{tag}/epoch-{e}"] = m["balanced_score"]["value"]
        print(f"{tag} epoch {e}: dev balanced {m['balanced_score']['value']:.3f}  "
              f"halluc {m['hallucination_rate']['value']:.3f}  over-refusal {m['over_refusal_rate']['value']:.3f}")
    order = list(scores)  # ties go to the earlier entry: lower lr, then earlier epoch
    best = max(order, key=lambda k: (scores[k], -order.index(k)))
    shutil.rmtree(out / "best", ignore_errors=True)
    shutil.copytree(out / best, out / "best")
    sft = json.load(open(cfg.results_dir / SFT_DEV_RUN / "metrics.json"))["balanced_score"]["value"]
    beats = scores[best] > sft
    res = cfg.results_dir / f"{name}-train"
    summary = json.load(open(res / "train_summary.json"))
    summary.update({"dev_balanced": scores, "best": best, "sft_dev_balanced": sft, "best_beats_sft_on_dev": beats})
    json.dump(summary, open(res / "train_summary.json", "w"), indent=2)
    print(f"best {best} -> {out / 'best'}: dev balanced {scores[best]:.3f} vs SFT {sft:.3f} -> "
          f"{'beats SFT' if beats else 'does NOT beat SFT'}")
    return beats


def main():
    p = add_config_arg(argparse.ArgumentParser())
    p.add_argument("--variant", required=True, help="a key of dpo.variants in config.json, e.g. dpo-v2")
    p.add_argument("--select", action="store_true", help="pick the best (lr, epoch) from the dev evals")
    p.add_argument("--dry-run", action="store_true", help="build the dataset and trainer, print checks, don't train")
    p.add_argument("--pairs-only", action="store_true", help="build and summarise the pair set only (CPU)")
    p.add_argument("--candidates", action="store_true", help="print '<lr tag> <epoch>' per trained adapter")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.candidates:
        print("\n".join(f"{t} {e}" for t, e in candidates(cfg, args.variant)))
    elif args.pairs_only:
        print(json.dumps(load_pairs(cfg, args.variant)[1], indent=2))
    elif args.select:
        raise SystemExit(0 if select(cfg, args.variant) else NOT_BETTER_THAN_SFT)
    else:
        train(cfg, args.variant, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
