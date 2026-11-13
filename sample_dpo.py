"""Sample the SFT model on a split, grade every output, and count the DPO preference pairs (runs on the RunPod GPU).

For every question: one greedy output (the eval decoding, so its mistakes are comparable to test) and
--n samples at --temperature (defaults: dpo.samples_per_question at dpo.sample_temperature). Every output is
graded with the frozen grader. GRPO uses the same script on the grpo split (8 samples at grpo.temperature) to
find the questions whose rollouts disagree; --recheck-n draws a second, independent set of samples per question
("recheck") that is never used for filtering, only to measure how many kept questions are still zero-variance on
a fresh draw.

A pair needs a rejected output the model actually produced:
  unanswerable  chosen "I don't know"                    rejected  a sampled answer (hallucination)
  answerable    chosen the gold answer (or a correct     rejected  a sampled "I don't know" (over-refusal)
                sample)                                            or a sampled wrong answer
Chosen can always be the canonical target (gold span / "I don't know"); `on_policy` marks questions where the
model also sampled a good output itself.

Writes <results_dir>/<run>/samples.jsonl (one row per question) and pair_counts.json.

Usage (on the pod):
  python sample_dpo.py --adapter checkpoints/sft/best --run dpo-samples
  python sample_dpo.py --run dpo-samples --recount     # recount from saved samples, no GPU
  python sample_dpo.py --split grpo --n 8 --recheck-n 8 --temperature 1.0 --run grpo-samples
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from config import add_config_arg, load_config
from grader import ABSTAIN, GRADER_VERSION, grade
from prompts import render


def generate(cfg, model_id, adapter, rows, n, temperature, recheck_n=0):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    v, e = cfg["vllm"], cfg["eval"]
    llm = LLM(model=model_id, dtype=v["dtype"], max_model_len=v["max_model_len"],
              gpu_memory_utilization=v["gpu_memory_utilization"], enable_lora=adapter is not None,
              max_lora_rank=v["max_lora_rank"], seed=e["seed"])
    prompts = [render(cfg, llm.get_tokenizer(), r) for r in rows]
    lora = LoRARequest("adapter", 1, str(adapter)) if adapter else None
    t0 = time.time()
    greedy = llm.generate(prompts, SamplingParams(temperature=e["temperature"], max_tokens=e["max_new_tokens"]),
                          lora_request=lora)
    sampled = llm.generate(prompts, SamplingParams(n=n + recheck_n, temperature=temperature,
                                                   max_tokens=e["max_new_tokens"], seed=e["seed"]),
                           lora_request=lora)
    print(f"generated {len(rows)} greedy + {len(rows) * (n + recheck_n)} samples in {time.time() - t0:.0f}s")
    texts = [[c.text.strip() for c in o.outputs] for o in sampled]
    return [o.outputs[0].text.strip() for o in greedy], [t[:n] for t in texts], [t[n:] for t in texts]


def graded(text, row):
    g = grade(text, row["answerable"], row["answers"])
    return {"output": text, "cell": g["cell"], "lenient_score": g["lenient_score"], "format_ok": g["format_ok"]}


def pairs_for(row) -> dict:
    """Chosen/rejected candidates for one question, from its graded samples."""
    s = row["samples"]
    if row["answerable"]:
        good = [x["output"] for x in s if x["cell"] == "correct"]
        bad = [x["output"] for x in s if x["cell"] in ("wrong", "over_refusal")]
        chosen = list(dict.fromkeys([row["gold"][0]] + good))
        kinds = Counter(x["cell"] for x in s if x["cell"] in ("wrong", "over_refusal"))
    else:
        good = [x["output"] for x in s if x["cell"] == "correct_abstain"]
        bad = [x["output"] for x in s if x["cell"] == "hallucination"]
        chosen = [ABSTAIN]
        kinds = Counter(x["cell"] for x in s if x["cell"] == "hallucination")
    rejected = list(dict.fromkeys(bad))
    return {"chosen": chosen, "rejected": rejected, "on_policy": bool(good and bad), "rejected_kinds": dict(kinds)}


def count(rows) -> dict:
    out = {}
    for name, ans in (("answerable", True), ("unanswerable", False)):
        rs = [r for r in rows if r["answerable"] == ans]
        p = [r["pairs"] for r in rs]
        kinds = Counter()
        for x in p:
            kinds.update({k: 1 for k in x["rejected_kinds"]})
        out[name] = {
            "questions": len(rs),
            "questions_with_pair": sum(bool(x["rejected"]) for x in p),
            "questions_with_on_policy_pair": sum(x["on_policy"] for x in p),
            "questions_by_rejected_kind": dict(kinds),
            "unique_pairs": sum(len(x["chosen"]) * len(x["rejected"]) for x in p),
            "bad_samples": sum(sum(x["rejected_kinds"].values()) for x in p),
            "greedy_cells": dict(Counter(r["greedy"]["cell"] for r in rs)),
            "sample_cells": dict(Counter(s["cell"] for r in rs for s in r["samples"])),
        }
    out["total"] = {k: out["answerable"][k] + out["unanswerable"][k]
                    for k in ("questions_with_pair", "questions_with_on_policy_pair", "unique_pairs")}
    return out


def main():
    p = add_config_arg(argparse.ArgumentParser())
    p.add_argument("--run", default="dpo-samples")
    p.add_argument("--model", default="main")
    p.add_argument("--adapter", default="checkpoints/sft/best")
    p.add_argument("--split", default="dpo")
    p.add_argument("--n", type=int, help="samples per question (default dpo.samples_per_question)")
    p.add_argument("--temperature", type=float, help="sampling temperature (default dpo.sample_temperature)")
    p.add_argument("--recheck-n", type=int, default=0, help="extra independent samples per question, stored apart")
    p.add_argument("--recount", action="store_true", help="recount pairs from saved samples.jsonl")
    args = p.parse_args()
    cfg = load_config(args.config)
    n = args.n or cfg["dpo"]["samples_per_question"]
    temperature = cfg["dpo"]["sample_temperature"] if args.temperature is None else args.temperature
    out_dir = cfg.results_dir / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.recount:
        rows = [json.loads(l) for l in open(out_dir / "samples.jsonl")]
        for r in rows:
            r["pairs"] = pairs_for(r)
    else:
        data = [json.loads(l) for l in open(cfg.data_dir / f"{args.split}.jsonl")]
        adapter = Path(args.adapter)
        adapter = adapter if adapter.is_absolute() else cfg.root / adapter
        model_id = cfg.model_id(args.model)
        print(f"model={model_id} adapter={adapter} split={args.split} n={len(data)} run={args.run}")
        greedy, samples, recheck = generate(cfg, model_id, adapter, data, n, temperature, args.recheck_n)
        rows = []
        for d, g, ss, rc in zip(data, greedy, samples, recheck):
            r = {"id": d["id"], "question": d["question"], "answerable": d["answerable"], "gold": d["answers"],
                 "greedy": graded(g, d), "samples": [graded(s, d) for s in ss]}
            if rc:
                r["recheck"] = [graded(s, d) for s in rc]
            r["pairs"] = pairs_for(r)
            rows.append(r)
        with open(out_dir / "samples.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    counts = count(rows)
    counts["config"] = {"model": cfg.model_id(args.model), "adapter": args.adapter, "split": args.split,
                        "samples_per_question": len(rows[0]["samples"]),
                        "recheck_samples_per_question": len(rows[0].get("recheck", [])),
                        "sample_temperature": temperature,
                        "max_new_tokens": cfg["eval"]["max_new_tokens"], "grader_version": GRADER_VERSION}
    json.dump(counts, open(out_dir / "pair_counts.json", "w"), indent=2)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
