"""Stage 2: LoRA SFT of the main model on the frozen sft split (runs on the RunPod GPU).

Targets: the gold span for answerable questions, exactly "I don't know" for unanswerable ones.
Rows whose gold answer is longer than sft.max_answer_words are dropped (they contradict the prompt's
"10 words or fewer"), and the same number of random unanswerable rows is dropped so the set stays 50/50.

The prompt is rendered with prompts.render(), byte-identical to eval; the loss covers only the target
tokens plus <|im_end|> (TRL SFTTrainer, prompt/completion dataset). An adapter is saved after every epoch to checkpoints/sft/epoch-<n>/.

Usage (on the pod):
  python sft.py --dry-run             # print the tokenized examples and loss mask, no training
  python sft.py                       # train, write checkpoints/sft/epoch-*/ and results/sft-train/
  python sft.py --select              # after the dev evals: copy the best epoch to checkpoints/sft/best
"""

import argparse
import hashlib
import json
import random
import shutil
import time

from config import add_config_arg, load_config
from grader import ABSTAIN

END = "<|im_end|>"


def target_for(row) -> str:
    return min(row["answers"], key=len) if row["answerable"] else ABSTAIN


def build_examples(rows, max_answer_words, seed):
    """Drop answerable rows with over-long targets and an equal number of random unanswerable rows.
    Returns (kept rows in original order, dropped long ids, dropped unanswerable ids)."""
    long_ids = {r["id"] for r in rows if r["answerable"] and len(target_for(r).split()) > max_answer_words}
    unans = sorted(r["id"] for r in rows if not r["answerable"])
    balance_ids = set(random.Random(seed).sample(unans, len(long_ids)))
    kept = [r for r in rows if r["id"] not in long_ids and r["id"] not in balance_ids]
    return kept, sorted(long_ids), sorted(balance_ids)


def train(cfg, dry_run=False):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from prompts import render

    s, lo = cfg["sft"], cfg["lora"]
    model_id = cfg.model_id("main")
    out = cfg.checkpoints_dir / "sft"
    res = cfg.results_dir / "sft-train"

    raw = (cfg.data_dir / "sft.jsonl").read_text()
    rows = [json.loads(l) for l in raw.splitlines()]
    kept, long_ids, balance_ids = build_examples(rows, s["max_answer_words"], s["seed"])
    n_ans = sum(r["answerable"] for r in kept)
    print(f"sft rows: {len(rows)} -> {len(kept)} ({n_ans} answerable, {len(kept) - n_ans} unanswerable); "
          f"dropped {len(long_ids)} long answers + {len(balance_ids)} random unanswerable")
    assert n_ans * 2 == len(kept), "SFT set is not 50/50"

    # Plain-text prompt/completion: the prompt is the exact eval string from prompts.render(), so TRL does not
    # apply the chat template again; it appends eos (<|im_end|>) to the completion and masks the prompt.
    tok = AutoTokenizer.from_pretrained(model_id)
    assert tok.eos_token == END, tok.eos_token
    ds = Dataset.from_list([{"prompt": render(cfg, tok, r), "completion": target_for(r)} for r in kept])
    ds = ds.shuffle(seed=s["seed"])

    torch.manual_seed(s["seed"])
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    peft_cfg = LoraConfig(r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
                          target_modules=lo["target_modules"], task_type="CAUSAL_LM")
    log = []

    class EpochSaver(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kw):
            d = out / f"epoch-{round(state.epoch)}"
            model.save_pretrained(d)
            tok.save_pretrained(d)
            print(f"saved {d}")

        def on_log(self, args, state, control, logs=None, **kw):
            log.append({"step": state.global_step, "epoch": state.epoch, **(logs or {})})

    args = SFTConfig(
        output_dir=str(out / "trainer"),
        num_train_epochs=s["epochs"],
        learning_rate=s["learning_rate"],
        per_device_train_batch_size=s["per_device_batch_size"],
        gradient_accumulation_steps=s["grad_accum_steps"],
        lr_scheduler_type=s["lr_scheduler"],
        warmup_steps=s["warmup_ratio"],  # transformers 5: a float < 1 is a ratio of total steps
        max_length=s["max_length"],
        completion_only_loss=s["loss_on_answer_only"],
        packing=False,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=s["seed"],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok,
                         peft_config=peft_cfg, callbacks=[EpochSaver()])
    trainer.model.print_trainable_parameters()

    # Check what TRL built: prompt tokens identical to tokenizing the eval prompt alone, loss only on the answer.
    td = trainer.train_dataset
    lens = [len(x) for x in td["input_ids"]]
    print(f"tokens/example: max {max(lens)}, mean {sum(lens) / len(lens):.0f} (max_length {s['max_length']})")
    assert max(lens) < s["max_length"], "an example hit max_length and was truncated"
    assert len(td) == len(ds), f"TRL dropped {len(ds) - len(td)} examples"
    for i in range(3):
        ex, ids = ds[i], td[i]["input_ids"]
        if "labels" in td.column_names:  # TRL 1.17 writes labels (-100 = no loss) instead of completion_mask
            mask = [int(l != -100) for l in td[i]["labels"]]
            assert [l for l in td[i]["labels"] if l != -100] == [t for t, m in zip(ids, mask) if m]
        else:
            mask = td[i]["completion_mask"]
        p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        assert ids[:len(p)] == p and mask == [0] * len(p) + [1] * (len(ids) - len(p)), f"prompt/mask mismatch at {i}"
        trained = tok.decode([t for t, m in zip(ids, mask) if m])
        print(f"example {i}: ...{tok.decode(ids[len(p) - 12:len(p)])!r} | loss on: {trained!r}")
    if dry_run:
        print("dry run: stopping before training")
        return

    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    shutil.rmtree(out / "trainer", ignore_errors=True)

    summary = {
        "model": model_id, "trainer": "trl.SFTTrainer", "lora": lo, "sft": s,
        "data_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "n_rows": len(rows), "n_train": len(kept), "n_answerable": n_ans,
        "dropped_long_answer_ids": long_ids, "dropped_unanswerable_ids": balance_ids,
        "train_seconds": round(secs), "eval_fingerprint": cfg.eval_fingerprint(),
        "gpu": torch.cuda.get_device_name(0),
    }
    json.dump(summary, open(res / "train_summary.json", "w"), indent=2)
    with open(res / "train_log.jsonl", "w") as f:
        for x in log:
            f.write(json.dumps(x) + "\n")
    print(f"trained in {secs:.0f}s; wrote {res}/train_summary.json, train_log.jsonl")


def select(cfg):
    """Pick the epoch with the best dev balanced score (runs sft-dev-e<n>) and copy it to checkpoints/sft/best."""
    out = cfg.checkpoints_dir / "sft"
    scores = {}
    for e in range(1, cfg["sft"]["epochs"] + 1):
        m = json.load(open(cfg.results_dir / f"sft-dev-e{e}" / "metrics.json"))
        scores[e] = m["balanced_score"]["value"]
        print(f"epoch {e}: dev balanced {scores[e]:.3f}  halluc {m['hallucination_rate']['value']:.3f}  "
              f"over-refusal {m['over_refusal_rate']['value']:.3f}")
    best = max(scores, key=lambda e: (scores[e], -e))  # ties go to the earlier epoch
    shutil.rmtree(out / "best", ignore_errors=True)
    shutil.copytree(out / f"epoch-{best}", out / "best")
    res = cfg.results_dir / "sft-train"
    summary = json.load(open(res / "train_summary.json"))
    summary["dev_balanced_by_epoch"] = scores
    summary["best_epoch"] = best
    json.dump(summary, open(res / "train_summary.json", "w"), indent=2)
    print(f"best epoch {best} -> {out / 'best'}")


def main():
    p = add_config_arg(argparse.ArgumentParser())
    p.add_argument("--select", action="store_true", help="pick the best epoch from the dev evals")
    p.add_argument("--dry-run", action="store_true", help="build the dataset and trainer, print checks, don't train")
    args = p.parse_args()
    cfg = load_config(args.config)
    select(cfg) if args.select else train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
