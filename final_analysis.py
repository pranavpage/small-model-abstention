"""Final CPU-only analysis over the tracked results/ (no GPU, no training). Writes results/final/.

  final_table.csv / .md   every model: dev and test 2x2 rates, dev P(IDK) AUC with a paired 95% CI vs SFT
  padding.md              answer length, lenient-strict gap, answers longer than SFT's, biggest increases
  reward_groups.md        TRL advantage scaling check on saved groups; share of 1 / 2 / 3+ distinct rewards per run
                          at the start (SFT recheck draw) and end (best checkpoint's draw) of training

Usage: uv run python final_analysis.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from grader import reward
from metrics import paired_auc_compare

HERE = Path(__file__).parent
RES = HERE / "results"
OUT = RES / "final"


def best_step(run):
    s = RES / f"grpo-{run}-train" / "train_summary.json"
    return json.load(open(s))["best_step"]


def models():
    """(label, dev run, test run or None, probe label)."""
    ge = best_step("G-equal")
    ge_test = "grpo-G-equal" if (RES / "grpo-G-equal" / "metrics.json").exists() else None
    return [
        ("Baseline 1.7B", "baseline-dev", "baseline", "Baseline"),
        ("SFT", "sft-dev-e2", "sft", "SFT"),
        ("DPO v1", "dpo-dev-lr5e-06-e1", "dpo", "DPO-v1"),
        ("DPO v2", "dpo-v2-dev-lr2e-05-e1", "dpo-v2", "DPO-v2"),
        (f"G-default best (step {best_step('G-default')})", f"grpo-G-default-dev-s{best_step('G-default')}", None,
         "G-default"),
        (f"G-equal best (step {ge})", f"grpo-G-equal-dev-s{ge}", ge_test, "G-equal"),
        ("8B ceiling", "ceiling-dev", "ceiling", "Ceiling"),
    ]


def rows(run):
    return [json.loads(l) for l in open(RES / run / "outputs.jsonl")]


def rates(run):
    m = json.load(open(RES / run / "metrics.json"))
    return {k: m[k]["value"] for k in ("balanced_score", "hallucination_rate", "over_refusal_rate", "answer_accuracy")}


def probe(label):
    return json.load(open(RES / "idk-probe" / f"{label}.json"))["p_idk"]


def final_table():
    dev_ans = {r["id"]: r["answerable"] for r in rows("sft-dev-e2")}
    split = lambda p, a: [p[i] for i in sorted(p) if dev_ans[i] == a]
    sft_p = probe("SFT")
    out = []
    for label, dev, test, pl in models():
        d = rates(dev)
        t = rates(test) if test else None
        p = probe(pl)
        a = paired_auc_compare(split(sft_p, True), split(sft_p, False), split(p, True), split(p, False))
        out.append({"model": label, "dev_run": dev, "test_run": test, "dev": d, "test": t,
                    "auc": a["b"], "auc_diff_vs_sft": a["diff"], "auc_ci95": a["ci95"]})
    pct = lambda x: f"{100 * x:.1f}"
    lines = ["| Model | Dev balanced | Dev halluc. | Dev over-refusal | Dev accuracy | Dev P(IDK) AUC | AUC vs SFT [95% CI] "
             "| Test balanced | Test halluc. | Test over-refusal | Test accuracy |", "|" + "---|" * 11]
    csv = ["model,dev_run,test_run,dev_balanced,dev_hallucination,dev_over_refusal,dev_accuracy,dev_auc,"
           "auc_diff_vs_sft,auc_ci95_low,auc_ci95_high,test_balanced,test_hallucination,test_over_refusal,test_accuracy"]
    for r in out:
        d, t = r["dev"], r["test"]
        tc = [pct(t[k]) if t else "not run" for k in ("balanced_score", "hallucination_rate", "over_refusal_rate",
                                                      "answer_accuracy")]
        cmp = "—" if r["model"] == "SFT" else (f"{r['auc_diff_vs_sft']:+.3f} [{r['auc_ci95'][0]:+.3f}, "
                                               f"{r['auc_ci95'][1]:+.3f}]")
        lines.append(f"| {r['model']} | {pct(d['balanced_score'])} | {pct(d['hallucination_rate'])} | "
                     f"{pct(d['over_refusal_rate'])} | {pct(d['answer_accuracy'])} | {r['auc']:.3f} | {cmp} | "
                     + " | ".join(tc) + " |")
        tv = [f"{t[k]:.4f}" if t else "" for k in ("balanced_score", "hallucination_rate", "over_refusal_rate",
                                                   "answer_accuracy")]
        csv.append(",".join([r["model"], r["dev_run"], r["test_run"] or "", *(f"{d[k]:.4f}" for k in (
            "balanced_score", "hallucination_rate", "over_refusal_rate", "answer_accuracy")), f"{r['auc']:.4f}",
            f"{r['auc_diff_vs_sft']:.4f}", f"{r['auc_ci95'][0]:.4f}", f"{r['auc_ci95'][1]:.4f}", *tv]))
    (OUT / "final_table.md").write_text("\n".join(lines) + "\n")
    (OUT / "final_table.csv").write_text("\n".join(csv) + "\n")
    print("\n".join(lines))
    return out


def padding():
    ms = models()
    pairs = {"dev": [("SFT", "sft-dev-e2"), (ms[4][0], ms[4][1]), (ms[5][0], ms[5][1])],
             "test": [("SFT", "sft")] + ([(ms[5][0], ms[5][2])] if ms[5][2] else [])}
    lines = ["| Split | Model | Mean answer words | Lenient − strict gap | Answers longer than SFT's (both answered) |",
             "|---|---|---|---|---|"]
    examples = []
    for split, items in pairs.items():
        sft = {r["id"]: r for r in rows(items[0][1])}
        for label, run in items:
            rs = rows(run)
            answered = [r for r in rs if not r["abstain"]]
            scored = [r for r in rs if r["answerable"] and r["lenient_score"] is not None]
            words = lambda r: len(r["output"].split())
            gap = sum(r["lenient_score"] - r["strict_score"] for r in scored) / len(scored)
            both = [(r, sft[r["id"]]) for r in rs if not r["abstain"] and not sft[r["id"]]["abstain"]]
            longer = [(words(r) - words(s), r, s) for r, s in both if words(r) > words(s)]
            n_longer = f"{len(longer)} of {len(both)}" if label != "SFT" else "—"
            lines.append(f"| {split} | {label} | {sum(map(words, answered)) / len(answered):.2f} | {gap:.3f} | {n_longer} |")
            if label != "SFT":
                examples += [(d, split, label, r, s) for d, r, s in longer]
    examples.sort(key=lambda x: -x[0])
    lines += ["", "Biggest length increases over SFT's answer to the same question:", "",
              "| Model | Split | Question | Gold | SFT | Model | +words | Cell (SFT → model) |", "|---|---|---|---|---|---|---|---|"]
    seen = set()
    for d, split, label, r, s in examples:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        lines.append(f"| {label} | {split} | {r['question']} | {'; '.join(r['gold']) or '(unanswerable)'} | {s['output']} | "
                     f"{r['output']} | +{d} | {s['cell']} → {r['cell']} |")
        if len(seen) == 5:
            break
    (OUT / "padding.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def adv(rw):
    """TRL 1.17.1, scale_rewards="group": (r - group mean) / (group std + 1e-4), std with n - 1."""
    rw = np.asarray(rw, float)
    return (rw - rw.mean()) / (rw.std(ddof=1) + 1e-4)


def reward_groups():
    pool = [json.loads(l) for l in open(RES / "grpo-pool" / "pool.jsonl")]
    sft = {x["id"]: x for x in map(json.loads, open(RES / "grpo-samples" / "samples.jsonl"))}
    runs = {"G-default": (1.0, 0.5), "G-equal": (1.0, 1.0)}
    lines = ["## Five saved two-value groups: normalised advantages under different penalties", ""]
    picks, want = [], [False, False, False, True, True]
    for p in pool:
        if p["split"] != "grpo" or not want:
            continue
        x = sft[p["id"]]
        o = [s["output"] for s in x["samples"]]
        if x["answerable"] != want[0] or any(p2["id"] == p["id"] for p2 in picks):
            continue
        base = [reward(t, x["answerable"], x["gold"])["reward"] for t in o]
        if len(set(base)) != 2:
            continue
        if x["answerable"] and not any(s["cell"] == "over_refusal" for s in x["samples"]):
            continue  # want "I don't know" in the answerable examples, so r is in play
        picks.append(p)
        want.pop(0)
        settings = [(1.0, 0.5), (3.0, 0.5)] if not x["answerable"] else [(1.0, 0.5), (1.0, 1.0)]
        lines.append(f"**{'Unanswerable' if not x['answerable'] else 'Answerable'}:** {x['question']} "
                     f"(gold: {'; '.join(x['gold']) or 'none'})")
        lines.append("")
        lines.append("| Output | " + " | ".join(f"reward h={h},r={r}" for h, r in settings) + " | "
                     + " | ".join(f"advantage h={h},r={r}" for h, r in settings) + " |")
        lines.append("|" + "---|" * (1 + 2 * len(settings)))
        rws = [[reward(t, x["answerable"], x["gold"], h, r)["reward"] for t in o] for h, r in settings]
        advs = [adv(rw) for rw in rws]
        for i, t in enumerate(o):
            lines.append(f"| {t} | " + " | ".join(f"{rw[i]:+.2f}" for rw in rws) + " | "
                         + " | ".join(f"{a[i]:+.4f}" for a in advs) + " |")
        lines.append(f"\nLargest difference between the advantage columns: {np.abs(advs[0] - advs[1]).max():.1e}\n")

    lines += ["## Share of reward groups with 1 / 2 / 3+ distinct values (GRPO-split pool rows, 8 fresh samples each)", "",
              "Training rollouts were not saved, so each run is measured at its start (SFT, independent recheck draw) "
              "and at its end (a fresh draw from the run's best checkpoint), scored with that run's own reward.", "",
              "| Run | When | Class | 1 value (no signal) | 2 values (h, r cancel out) | 3+ values (h, r can matter) |",
              "|---|---|---|---|---|---|"]
    for run, (h, r) in runs.items():
        end = {x["id"]: x for x in map(json.loads, open(RES / f"grpo-{run}-best-samples" / "samples.jsonl"))}
        for when, src, key in (("start (SFT)", sft, "recheck"), (f"end (best, step {best_step(run)})", end, "samples")):
            for cls in ("answerable", "unanswerable", "all"):
                c = Counter()
                for p in pool:
                    if p["split"] != "grpo" or (cls != "all" and p["answerable"] != (cls == "answerable")):
                        continue
                    x = src[p["id"]]
                    k = len({reward(s["output"], x["answerable"], x["gold"], h, r)["reward"] for s in x[key]})
                    c[min(k, 3)] += 1
                n = sum(c.values())
                lines.append(f"| {run} | {when} | {cls} | {100 * c[1] / n:.0f}% | {100 * c[2] / n:.0f}% | "
                             f"{100 * c[3] / n:.0f}% |")
    (OUT / "reward_groups.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    final_table()
    print()
    padding()
    print()
    reward_groups()


if __name__ == "__main__":
    main()
