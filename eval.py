"""Evaluate a model on a frozen split with vLLM (runs on the RunPod GPU).

Prompt, decoding and paths come from config.json. Writes to <results_dir>/<run>/:
  outputs.jsonl    one graded row per question
  metrics.json     2x2 rates with bootstrap intervals, strict vs lenient, reward, config fingerprint
  handcheck.jsonl  random graded outputs for the manual grader check

Usage (on the pod):
  python eval.py --model main --split dev --run baseline-dev
  python eval.py --model main --split test --run baseline
  python eval.py --model ceiling --split test --run ceiling
  python eval.py --model main --adapter checkpoints/sft/best --split test --run sft
  python eval.py --run baseline --regrade          # re-score saved outputs, no GPU needed
"""

import argparse
import json
import random
import time
from pathlib import Path

from config import add_config_arg, load_config
from grader import ABSTAIN, GRADER_VERSION, grade
from metrics import compute_metrics, format_table
from prompts import render


def generate_vllm(cfg, model_id, adapter, rows):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    v = cfg["vllm"]
    llm = LLM(
        model=model_id,
        dtype=v["dtype"],
        max_model_len=v["max_model_len"],
        gpu_memory_utilization=v["gpu_memory_utilization"],
        enable_lora=adapter is not None,
        max_lora_rank=v["max_lora_rank"],
        seed=cfg["eval"]["seed"],
    )
    prompts = [render(cfg, llm.get_tokenizer(), r) for r in rows]
    params = SamplingParams(temperature=cfg["eval"]["temperature"], max_tokens=cfg["eval"]["max_new_tokens"])
    lora = LoRARequest("adapter", 1, str(adapter)) if adapter else None
    t0 = time.time()
    outs = llm.generate(prompts, params, lora_request=lora)
    print(f"generated {len(outs)} outputs in {time.time() - t0:.0f}s")
    return [o.outputs[0].text.strip() for o in outs]


def write_results(cfg, out_dir, items, run_config):
    graded = []
    with open(out_dir / "outputs.jsonl", "w") as f:
        for r, text in items:
            g = {"id": r["id"], "question": r["question"], "answerable": r["answerable"],
                 "gold": r["answers"], "output": text, **grade(text, r["answerable"], r["answers"])}
            graded.append(g)
            f.write(json.dumps(g) + "\n")

    e = cfg["eval"]
    metrics = compute_metrics(graded, n_boot=e["bootstrap_samples"], seed=e["seed"])
    metrics["config"] = run_config
    metrics["grader_version"] = GRADER_VERSION
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)

    sample = random.Random(e["seed"]).sample(graded, min(e["handcheck_samples"], len(graded)))
    keys = ["id", "question", "answerable", "gold", "output", "cell", "strict_cell", "strict_score", "lenient_score"]
    with open(out_dir / "handcheck.jsonl", "w") as f:
        for g in sample:
            f.write(json.dumps({k: g[k] for k in keys}) + "\n")

    print(format_table(metrics))
    print(f"\nwrote {out_dir}/outputs.jsonl, metrics.json, handcheck.jsonl")


def main():
    p = add_config_arg(argparse.ArgumentParser())
    p.add_argument("--run", required=True, help="results subfolder name")
    p.add_argument("--model", help="'main' | 'debug' | 'ceiling' (from config.models) or an HF id / local path")
    p.add_argument("--adapter", help="LoRA adapter directory (relative paths resolve under paths.root)")
    p.add_argument("--split", default="dev")
    p.add_argument("--limit", type=int, help="debug only: first N rows; not allowed on test")
    p.add_argument("--force", action="store_true", help="allow re-running an existing test-set run")
    p.add_argument("--regrade", action="store_true", help="re-score saved outputs with the current grader")
    args = p.parse_args()
    cfg = load_config(args.config)
    out_dir = cfg.results_dir / args.run

    if args.regrade:
        prev = [json.loads(l) for l in open(out_dir / "outputs.jsonl")]
        run_config = json.load(open(out_dir / "metrics.json"))["config"]
        items = [({"id": r["id"], "question": r["question"], "answerable": r["answerable"],
                   "answers": r["gold"]}, r["output"]) for r in prev]
        print(f"regrading {len(items)} saved outputs in {out_dir} (generation config unchanged)")
        return write_results(cfg, out_dir, items, run_config)

    if not args.model:
        p.error("--model is required unless --regrade")
    if ABSTAIN not in cfg["prompt"]["system"]:
        raise SystemExit(f"config prompt.system must tell the model to reply exactly: {ABSTAIN}")
    if args.split == "test":
        if args.limit:
            raise SystemExit("--limit is for debugging and is not allowed on the test split")
        if (out_dir / "metrics.json").exists() and not args.force:
            raise SystemExit(f"{out_dir} already has test results; the test set runs once per stage (--force to override)")

    rows = [json.loads(l) for l in open(cfg.data_dir / f"{args.split}.jsonl")]
    if args.limit:
        rows = rows[:args.limit]
    model_id = cfg.model_id(args.model)
    adapter = None
    if args.adapter:
        adapter = Path(args.adapter)
        adapter = adapter if adapter.is_absolute() else cfg.root / adapter
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={model_id} adapter={adapter} split={args.split} n={len(rows)} run={args.run}")

    texts = generate_vllm(cfg, model_id, adapter, rows)
    run_config = {
        "model": model_id, "adapter": str(adapter) if adapter else None, "split": args.split, "n": len(rows),
        "backend": "vllm", "dtype": cfg["vllm"]["dtype"], "temperature": cfg["eval"]["temperature"],
        "max_new_tokens": cfg["eval"]["max_new_tokens"], "prompt": cfg["prompt"],
        "eval_fingerprint": cfg.eval_fingerprint(), "experiment": cfg["experiment"],
    }
    write_results(cfg, out_dir, list(zip(rows, texts)), run_config)


if __name__ == "__main__":
    main()
