"""Paired comparison of two eval runs on the same questions (CPU only, reads saved outputs).

Every model answers the same frozen questions, so differences are tested question by question:
a paired bootstrap interval on each difference, and an exact McNemar test on the questions that
flipped. Refuses runs with different splits, question sets, eval fingerprints or grader versions.

Writes <dir>/compare/<a>__vs__<b>.json.

Usage:
  python compare.py baseline sft                     # runs under paths.results_dir (on the pod)
  python compare.py sft-dev-e1 sft-dev-e2 --dir results   # the copies tracked in the repo
"""

import argparse
import json
from pathlib import Path

from config import add_config_arg, load_config
from metrics import format_compare, paired_compare


def load_run(d: Path):
    m = json.load(open(d / "metrics.json"))
    rows = [json.loads(l) for l in open(d / "outputs.jsonl")]
    return m, rows


def main():
    p = add_config_arg(argparse.ArgumentParser())
    p.add_argument("a", help="reference run (e.g. sft)")
    p.add_argument("b", help="run compared against it (e.g. dpo)")
    p.add_argument("--dir", help="folder holding the runs (default: paths.results_dir)")
    args = p.parse_args()
    cfg = load_config(args.config)
    root = Path(args.dir) if args.dir else cfg.results_dir

    (ma, ra), (mb, rb) = load_run(root / args.a), load_run(root / args.b)
    for key in ["split", "eval_fingerprint"]:
        if ma["config"][key] != mb["config"][key]:
            raise SystemExit(f"{key} differs: {args.a}={ma['config'][key]} {args.b}={mb['config'][key]}")
    if ma["grader_version"] != mb["grader_version"]:
        raise SystemExit(f"grader_version differs: {ma['grader_version']} vs {mb['grader_version']}")

    e = cfg["eval"]
    c = paired_compare(ra, rb, n_boot=e["bootstrap_samples"], seed=e["seed"])
    c["runs"] = {"a": args.a, "b": args.b, "split": ma["config"]["split"],
                 "eval_fingerprint": ma["config"]["eval_fingerprint"], "grader_version": ma["grader_version"]}
    print(format_compare(c, args.a, args.b))
    out = root / "compare" / f"{args.a}__vs__{args.b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(c, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
