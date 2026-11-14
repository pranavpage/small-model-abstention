"""Question twins for input-side DPO: an unanswerable SQuAD 2.0 question and the answerable question it was edited from.

SQuAD 2.0's unanswerable questions were written by crowdworkers looking at a paragraph's answerable questions, and
many are one small edit away from one of them ("What did X use ..." -> "What did X not use ...", "North America" ->
"South America"). The dataset does not link them, so twins are matched by question wording, within one paragraph:

  similarity  difflib ratio between the two questions' normalised word sequences (grader.normalize)
  match       each unanswerable question's most similar answerable question in the same paragraph, kept if
              similarity >= min_similarity and it beats the runner-up by >= min_margin; then one-to-one, greedily
              by similarity (an answerable question is used by at most one twin)
  answer      the shortest gold answer of the answerable question, <= max_answer_words

Training twins: at least one question in `anchor_split` (the DPO split, so the stage draws on the same paragraphs
as the other DPO variants) and no question in `exclude_splits`; no paragraph that holds a dev question.
Held-out twins: `heldout` twins sampled from paragraphs with no question in any split; used only to probe whether
the model tells unseen twins apart.

Writes (paths.data_dir):  twins.jsonl          training twins (with paragraph)
                          twins-heldout.jsonl  held-out twins
                          twins-heldout-rows.jsonl  the held-out questions as eval rows (answerable first, then
                                                    unanswerable, in twin order)
and results/twins/ (tracked): the same twins without paragraphs, plus stats.json.

Usage (on the pod, CPU only): python -m abstention.twins
"""

import argparse
import difflib
import hashlib
import json
import random
from collections import Counter, defaultdict

from abstention.config import add_config_arg, load_config
from abstention.grader import normalize

SPLITS = ("sft", "dpo", "grpo", "dev", "test")


def similarity(q1: str, q2: str) -> float:
    return difflib.SequenceMatcher(None, normalize(q1).split(), normalize(q2).split()).ratio()


def shortest_gold(answers):
    return min(dict.fromkeys(a.strip() for a in answers), key=len)


def match_twins(paragraphs, min_similarity, min_margin, max_answer_words):
    """paragraphs: {context: [record, ...]} with data.to_record fields. Returns one-to-one twins, best match first."""
    cands = []
    for context, qs in paragraphs.items():
        ans = [q for q in qs if q["answerable"]]
        for u in (q for q in qs if not q["answerable"]):
            scored = sorted(((similarity(u["question"], a["question"]), i, a) for i, a in enumerate(ans)
                             if normalize(a["question"]) != normalize(u["question"])), key=lambda x: (-x[0], x[1]))
            if not scored:
                continue
            best, _, a = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            if best < min_similarity or best - second < min_margin:
                continue
            gold = shortest_gold(a["answers"])
            if len(gold.split()) > max_answer_words:
                continue
            cands.append({"similarity": round(best, 4), "runner_up": round(second, 4), "title": u["title"],
                          "context": context, "unanswerable": {"id": u["id"], "question": u["question"]},
                          "answerable": {"id": a["id"], "question": a["question"], "answer": gold,
                                         "answers": a["answers"]}})
    cands.sort(key=lambda t: (-t["similarity"], t["unanswerable"]["id"]))
    used, twins = set(), []
    for t in cands:
        if t["answerable"]["id"] not in used:
            used.add(t["answerable"]["id"])
            twins.append(t)
    return twins


def select(twins, split_of, paragraphs, anchor_split, exclude_splits, n_heldout, seed):
    """(training twins, held-out twins)."""
    blocked = {c for c, qs in paragraphs.items() if any(split_of.get(q["id"]) == "dev" for q in qs)}
    touched = {c for c, qs in paragraphs.items() if any(q["id"] in split_of for q in qs)}
    train, free = [], []
    for t in twins:
        members = {split_of.get(t["unanswerable"]["id"]), split_of.get(t["answerable"]["id"])}
        t["splits"] = {"unanswerable": split_of.get(t["unanswerable"]["id"]),
                       "answerable": split_of.get(t["answerable"]["id"])}
        if t["context"] in blocked:
            continue
        if anchor_split in members and not members & set(exclude_splits):
            train.append(t)
        elif t["context"] not in touched:
            free.append(t)
    rng = random.Random(seed)
    heldout = rng.sample(free, n_heldout)
    for i, t in enumerate(train):
        t["twin_id"] = f"t{i:04d}"
    for i, t in enumerate(heldout):
        t["twin_id"] = f"h{i:04d}"
    return train, heldout


def heldout_rows(heldout):
    """Eval-format rows: every answerable twin question, then every unanswerable one, in twin order."""
    rows = [{"id": t["answerable"]["id"], "title": t["title"], "context": t["context"],
             "question": t["answerable"]["question"], "answers": t["answerable"]["answers"], "answerable": True}
            for t in heldout]
    rows += [{"id": t["unanswerable"]["id"], "title": t["title"], "context": t["context"],
              "question": t["unanswerable"]["question"], "answers": [], "answerable": False} for t in heldout]
    return rows


def main():
    p = add_config_arg(argparse.ArgumentParser())
    args = p.parse_args()
    cfg = load_config(args.config)
    tc = cfg["dpo_twin"]["twins"]

    from datasets import load_dataset

    from abstention.data import to_record
    split_of = {}
    for s in SPLITS:
        for line in open(cfg.data_dir / f"{s}.jsonl"):
            split_of[json.loads(line)["id"]] = s
    paragraphs = defaultdict(list)
    for ex in load_dataset(cfg["data"]["dataset"])["train"]:
        r = to_record(ex)
        paragraphs[r["context"]].append(r)

    twins = match_twins(paragraphs, tc["min_similarity"], tc["min_margin"], cfg["dpo_twin"]["max_answer_words"])
    train, heldout = select(twins, split_of, paragraphs, tc["anchor_split"], tc["exclude_splits"], tc["heldout"],
                            tc["seed"])

    out = cfg.data_dir
    tracked = cfg.results_dir / "twins"
    tracked.mkdir(parents=True, exist_ok=True)
    stats = {"config": tc, "max_answer_words": cfg["dpo_twin"]["max_answer_words"],
             "matched_one_to_one": len(twins), "train": len(train), "heldout": len(heldout),
             "train_split_membership": dict(Counter(f"{t['splits']['unanswerable']}/{t['splits']['answerable']}"
                                                    for t in train)),
             "train_similarity": dict(sorted(Counter(f"{int(t['similarity'] * 20) / 20:.2f}" for t in train).items())),
             "sha256": {}}
    for name, items in (("twins", train), ("twins-heldout", heldout), ("twins-heldout-rows", heldout_rows(heldout))):
        text = "".join(json.dumps(x) + "\n" for x in items)
        (out / f"{name}.jsonl").write_text(text)
        stats["sha256"][name] = hashlib.sha256(text.encode()).hexdigest()
    for name, items in (("twins", train), ("twins-heldout", heldout)):
        with open(tracked / f"{name}.jsonl", "w") as f:
            for t in items:
                f.write(json.dumps({k: v for k, v in t.items() if k != "context"}) + "\n")
    json.dump(stats, open(tracked / "stats.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "sha256"}, indent=2))
    for t in train[:: max(1, len(train) // 8)][:8]:
        print(f"{t['similarity']:.2f}  A: {t['answerable']['question']}  -> {t['answerable']['answer']}\n"
              f"      U: {t['unanswerable']['question']}")
    print(f"wrote {out}/twins*.jsonl and {tracked}/")


if __name__ == "__main__":
    main()
