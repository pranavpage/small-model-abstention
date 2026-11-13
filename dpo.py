"""Stage 3: DPO on top of the best SFT adapter (runs on the RunPod GPU).

Pairs come from results/<dpo-samples>/samples.jsonl (sample_dpo.py: the SFT model sampled on the dpo split):
  unanswerable  chosen "I don't know"                      rejected  each distinct hallucinated sample
  answerable    chosen the model's most common sample that  rejected  each distinct sampled "I don't know", and
                is strict-correct (token F1 >= 0.5) and               each distinct wrong sample that is NOT a
                <= max_answer_words; else the shortest                partial span of a gold answer (those are
                gold span if <= max_answer_words; else the            often fine answers the grader marks down,
                question is skipped                                   e.g. "ADSL-broadband" for "ADSL")
Answerable pairs are randomly subsampled to the number of unanswerable pairs so the set is 50/50.

Training continues the SFT LoRA adapter (loaded trainable). TRL copies it into a frozen "ref" adapter, so the
DPO reference is the SFT model, not the base model. One run per learning rate in dpo.learning_rates; an adapter
(the trained "default" adapter only) is saved after each epoch to checkpoints/dpo/<lr tag>/epoch-<n>/.

Usage (on the pod):
  python dpo.py --pairs-only          # build and print the pair set, no GPU
  python dpo.py --dry-run             # build the trainer, print tokenization / adapter checks, don't train
  python dpo.py                       # train every learning rate
  python dpo.py --select              # after the dev evals: copy the best (lr, epoch) to checkpoints/dpo/best
"""

import argparse
import hashlib
import json
import random
import shutil
import time
from collections import Counter

from config import add_config_arg, load_config
from grader import ABSTAIN, max_f1, normalize

END = "<|im_end|>"


def lr_tag(lr: float) -> str:
    return f"lr{lr:g}"


def candidates(cfg):
    """Every (lr tag, epoch) trained, in sweep order."""
    d = cfg["dpo"]
    return [(lr_tag(lr), e) for lr in d["learning_rates"] for e in range(1, d["epochs"] + 1)]


def is_partial_span(output: str, golds: list[str]) -> bool:
    o = normalize(output)
    return bool(o) and any(o in normalize(g) or normalize(g) in o for g in golds if normalize(g))


def build_pairs(samples, max_answer_words, seed):
    """Preference pairs from sample_dpo.py rows. Returns (pairs, stats)."""
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
    rng = random.Random(seed)
    if len(ans) > len(unans):
        ans = rng.sample(ans, len(unans))
    elif len(unans) > len(ans):
        unans = rng.sample(unans, len(ans))
    pairs = ans + unans
    rng.shuffle(pairs)
    stats = dict(stats)
    stats.update({
        "pairs": len(pairs), "answerable_pairs": len(ans), "unanswerable_pairs": len(unans),
        "questions": len({p["id"] for p in pairs}),
        "answerable_rejected": dict(Counter(p["rejected_cell"] for p in ans)),
        "answerable_chosen_source": dict(Counter(p["chosen_source"] for p in ans)),
    })
    return pairs, stats


def load_pairs(cfg):
    d = cfg["dpo"]
    src = cfg.results_dir / d["samples_run"] / "samples.jsonl"
    raw = src.read_text()
    samples = [json.loads(l) for l in raw.splitlines()]
    pairs, stats = build_pairs(samples, d["max_answer_words"], d["seed"])
    stats["samples_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    return pairs, stats


def train(cfg, dry_run=False):
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    from prompts import render

    d = cfg["dpo"]
    model_id = cfg.model_id("main")
    sft_dir = cfg.checkpoints_dir / "sft" / "best"
    out = cfg.checkpoints_dir / "dpo"
    res = cfg.results_dir / "dpo-train"
    res.mkdir(parents=True, exist_ok=True)

    pairs, stats = load_pairs(cfg)
    print(json.dumps(stats, indent=2))
    with open(res / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    tok = AutoTokenizer.from_pretrained(model_id)
    assert tok.eos_token == END, tok.eos_token
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(cfg.data_dir / "dpo.jsonl")}
    ds = Dataset.from_list([{"prompt": render(cfg, tok, rows[p["id"]]), "chosen": p["chosen"],
                             "rejected": p["rejected"]} for p in pairs])

    summaries = {}
    for lr in d["learning_rates"]:
        tag = lr_tag(lr)
        torch.manual_seed(d["seed"])
        base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, sft_dir, is_trainable=True)
        log = []

        class EpochSaver(TrainerCallback):
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
        trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, callbacks=[EpochSaver()])
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
            print("dry run: stopping before training")
            return

        ref_before = {n: p.detach().clone() for n, p in pm.named_parameters() if ".ref." in n}
        t0 = time.time()
        trainer.train()
        secs = time.time() - t0
        assert all(torch.equal(p, ref_before[n]) for n, p in pm.named_parameters() if ".ref." in n), "ref adapter moved"
        shutil.rmtree(out / "trainer", ignore_errors=True)
        summaries[tag] = {"learning_rate": lr, "train_seconds": round(secs)}
        with open(res / f"train_log_{tag}.jsonl", "w") as f:
            for x in log:
                f.write(json.dumps(x) + "\n")
        del trainer, model, base, pm
        torch.cuda.empty_cache()

    summary = {"model": model_id, "start_adapter": str(sft_dir), "trainer": "trl.DPOTrainer", "lora": cfg["lora"],
               "dpo": d, "pairs": stats, "runs": summaries, "eval_fingerprint": cfg.eval_fingerprint(),
               "gpu": torch.cuda.get_device_name(0)}
    json.dump(summary, open(res / "train_summary.json", "w"), indent=2)
    print(f"wrote {res}/train_summary.json, train_log_*.jsonl, pairs.jsonl")


def select(cfg):
    """Pick the (lr, epoch) with the best dev balanced score (runs dpo-dev-<tag>-e<n>) -> checkpoints/dpo/best."""
    out = cfg.checkpoints_dir / "dpo"
    scores = {}
    for tag, e in candidates(cfg):
        m = json.load(open(cfg.results_dir / f"dpo-dev-{tag}-e{e}" / "metrics.json"))
        scores[f"{tag}/epoch-{e}"] = m["balanced_score"]["value"]
        print(f"{tag} epoch {e}: dev balanced {m['balanced_score']['value']:.3f}  "
              f"halluc {m['hallucination_rate']['value']:.3f}  over-refusal {m['over_refusal_rate']['value']:.3f}")
    order = list(scores)  # ties go to the earlier entry: lower lr, then earlier epoch
    best = max(order, key=lambda k: (scores[k], -order.index(k)))
    shutil.rmtree(out / "best", ignore_errors=True)
    shutil.copytree(out / best, out / "best")
    res = cfg.results_dir / "dpo-train"
    summary = json.load(open(res / "train_summary.json"))
    summary["dev_balanced"] = scores
    summary["best"] = best
    json.dump(summary, open(res / "train_summary.json", "w"), indent=2)
    print(f"best {best} -> {out / 'best'}")


def main():
    p = add_config_arg(argparse.ArgumentParser())
    p.add_argument("--select", action="store_true", help="pick the best (lr, epoch) from the dev evals")
    p.add_argument("--dry-run", action="store_true", help="build the dataset and trainer, print checks, don't train")
    p.add_argument("--pairs-only", action="store_true", help="build and summarise the pair set only (CPU)")
    p.add_argument("--candidates", action="store_true", help="print '<lr tag> <epoch>' per trained adapter")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.candidates:
        print("\n".join(f"{t} {e}" for t, e in candidates(cfg)))
    elif args.pairs_only:
        print(json.dumps(load_pairs(cfg)[1], indent=2))
    elif args.select:
        select(cfg)
    else:
        train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
