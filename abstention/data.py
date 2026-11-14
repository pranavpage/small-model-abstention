"""Build the frozen SQuAD 2.0 splits described in the brief and save them under paths.data_dir.

Train-derived splits (sft, dpo, grpo, dev) are sampled without overlap from SQuAD train.
The test split is sampled from SQuAD's official dev split (different Wikipedia articles).
Every split is 50/50 answerable / unanswerable. Sampling is done once with data.seed.
A manifest with a sha256 per split is written so later runs can prove they used the same files.

Usage (on the pod): python -m abstention.data
"""

import argparse
import hashlib
import json
import random

from abstention.config import add_config_arg, load_config


def to_record(ex):
    answers = list(dict.fromkeys(a.strip() for a in ex["answers"]["text"]))
    return {
        "id": ex["id"],
        "title": ex["title"],
        "context": ex["context"],
        "question": ex["question"],
        "answers": answers,
        "answerable": len(answers) > 0,
    }


def split_pools(ds):
    records = [to_record(ex) for ex in ds]
    return [r for r in records if r["answerable"]], [r for r in records if not r["answerable"]]


def take_balanced(ans, unans, n, rng):
    half = n // 2
    if len(ans) < half or len(unans) < half:
        raise SystemExit(f"not enough examples left for a split of {n}")
    picked = ans[:half] + unans[:half]
    del ans[:half]
    del unans[:half]
    rng.shuffle(picked)
    return picked


def main():
    p = add_config_arg(argparse.ArgumentParser())
    args = p.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    out = cfg.data_dir
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    squad = load_dataset(d["dataset"])
    rng = random.Random(d["seed"])

    splits = {}
    ans, unans = split_pools(squad["train"])
    rng.shuffle(ans)
    rng.shuffle(unans)
    for name, n in d["train_splits"].items():
        splits[name] = take_balanced(ans, unans, n, rng)

    ans, unans = split_pools(squad["validation"])
    rng.shuffle(ans)
    rng.shuffle(unans)
    splits["test"] = take_balanced(ans, unans, d["test_size"], rng)

    # Sanity: no question id in two splits; test articles unseen in train-derived splits.
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not {r["id"] for r in splits[a]} & {r["id"] for r in splits[b]}, f"overlap {a}/{b}"
    train_titles = {r["title"] for n in d["train_splits"] for r in splits[n]}
    assert not {r["title"] for r in splits["test"]} & train_titles, "test articles overlap train"

    manifest = {"dataset": d["dataset"], "seed": d["seed"], "splits": {}}
    for name, rows in splits.items():
        text = "".join(json.dumps(r) + "\n" for r in rows)
        (out / f"{name}.jsonl").write_text(text)
        manifest["splits"][name] = {"n": len(rows), "answerable": sum(r["answerable"] for r in rows),
                                    "sha256": hashlib.sha256(text.encode()).hexdigest()}
        print(f"{name}: {len(rows)} ({manifest['splits'][name]['answerable']} answerable)")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"no overlap; test articles disjoint from train. wrote {out}/manifest.json")


if __name__ == "__main__":
    main()
