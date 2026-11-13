"""Known grader limitation: hyphens and dashes (CPU only, reads saved outputs; the grader itself is NOT changed).

grader.normalize() deletes "-" without inserting a space ("Anglo-Saxon" -> "anglosaxon", so it can't match
"Anglo Saxon"), and en/em dashes and minus signs are not in its punctuation set, so they survive as characters
("1914–18"). This script regrades every saved eval output (results/*/outputs.jsonl: every model, dev and test)
with one change, dashes -> spaces before normalising, and lists every answerable output whose 2x2 cell changes.
Unanswerable rows are unaffected (the abstain phrase has no dash).

A flip is not automatically a misgrade under v1.0: the alternative normalisation is also wrong sometimes
("1024-bit" for gold "512-bit" becomes 0.5 = correct). Each distinct flipped (gold, output) is judged in
VERDICTS below; "v1.0 wrong" means the frozen grader's cell is the wrong one.

Usage: uv run python audit_hyphens.py   -> results/grader-audit/hyphens.json and hyphens.md
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import grader as G

HERE = Path(__file__).parent
DASH = re.compile(r"[\-‐‑‒–—―−]")

# (normalised first gold, output) -> (is the v1.0 cell wrong?, reason). Filled in by reading every flipped case.
VERDICTS = {
    ("512-bit", "1024-bit"): (False, "different number; dashes-as-spaces wrongly credits the shared word 'bit'"),
    ("mid-2000s", "2000s"): (False, "drops 'mid' (the decade, not its middle); v1.0 'wrong' is defensible"),
    ("four", "four-valve"): (True, "gold number inside a hyphenated compound"),
    ("ten", "10 horsepower"): (True, "right answer (10 = ten); v1.0 misses it because 'ten-horsepower' becomes one "
                                     "token, and dashes-as-spaces credits it only through the word 'horsepower'"),
    ("ADSL", "ADSL-broadband"): (True, "gold span plus a hyphenated qualifier"),
    ("ADSL", "ADSL-broadband service is provided on the island."): (True, "right content in a sentence (the format "
                                                                          "rule, not the 2x2, covers sentences)"),
    ("Hargeisa", "Hargeisa-based"): (True, "gold span plus a hyphenated suffix"),
    ("Hargeisa", "Hargeisa-based telecommunications firm"): (True, "gold span plus a hyphenated suffix"),
}


def regrade(output, golds):
    orig = G.normalize
    G.normalize = lambda s: orig(DASH.sub(" ", s))
    try:
        return G.grade(output, True, golds)
    finally:
        G.normalize = orig


def main():
    runs, flips = {}, defaultdict(list)
    for f in sorted((HERE / "results").glob("*/outputs.jsonl")):
        run = f.parent.name
        meta = json.load(open(f.parent / "metrics.json"))["config"]
        n = changed = 0
        run_flips = []
        for r in map(json.loads, open(f)):
            if not r["answerable"]:
                continue
            n += 1
            a, b = G.grade(r["output"], True, r["gold"]), regrade(r["output"], r["gold"])
            changed += a["lenient_score"] != b["lenient_score"]
            if a["cell"] != b["cell"]:
                key = (r["gold"][0], r["output"])
                run_flips.append({"id": r["id"], "gold": r["gold"], "output": r["output"], "v1_cell": a["cell"],
                                  "dash_as_space_cell": b["cell"], "v1_lenient": a["lenient_score"],
                                  "dash_as_space_lenient": b["lenient_score"]})
                flips[key].append(run)
        runs[run] = {"split": meta["split"], "answerable_outputs": n, "score_changed": changed,
                     "cell_flips": len(run_flips), "flips": run_flips}

    unjudged = [k for k in flips if k not in VERDICTS]
    if unjudged:
        raise SystemExit(f"judge these flipped cases in VERDICTS first: {unjudged}")
    for r in runs.values():
        r["misgraded_by_v1"] = sum(VERDICTS[(x["gold"][0], x["output"])][0] for x in r["flips"])

    total = {s: {k: sum(r[k] for r in runs.values() if r["split"] == s)
                 for k in ("answerable_outputs", "score_changed", "cell_flips", "misgraded_by_v1")}
             for s in ("dev", "test")}
    out = HERE / "results" / "grader-audit"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"grader_version": G.GRADER_VERSION, "totals": total, "runs": runs,
               "cases": [{"gold": g, "output": o, "runs": rs, "v1_wrong": VERDICTS[(g, o)][0],
                          "reason": VERDICTS[(g, o)][1]} for (g, o), rs in flips.items()]},
              open(out / "hyphens.json", "w"), indent=2)

    lines = [f"# Known grader limitation: hyphens and dashes (grader v{G.GRADER_VERSION}, unchanged)", "",
             "The grader deletes `-` without a space and keeps en/em dashes, so hyphenated words can fail to match. "
             "Every saved eval output was regraded with dashes treated as spaces; only answerable outputs can change.",
             "", "| Split | Answerable outputs | Score changed | Cell flipped | Misgraded by v1.0 |", "|---|---|---|---|---|"]
    for s, t in total.items():
        lines.append(f"| {s} | {t['answerable_outputs']} | {t['score_changed']} | {t['cell_flips']} | "
                     f"{t['misgraded_by_v1']} |")
    lines += ["", "| Gold | Output | v1.0 | Dashes as spaces | v1.0 wrong? | Runs |", "|---|---|---|---|---|---|"]
    for (g, o), rs in flips.items():
        x = next(f for f in runs[rs[0]]["flips"] if f["output"] == o)
        lines.append(f"| {g} | {o} | {x['v1_cell']} ({x['v1_lenient']:.2f}) | {x['dash_as_space_cell']} "
                     f"({x['dash_as_space_lenient']:.2f}) | {'yes' if VERDICTS[(g, o)][0] else 'no'}: "
                     f"{VERDICTS[(g, o)][1]} | {', '.join(rs)} |")
    # Ranking check on the final model set: balanced score with v1.0 cells vs with every v1.0 misgrade corrected.
    from final_analysis import models
    lines += ["", "## Does correcting the misgrades change any ranking?", "",
              "| Model | Split | Misgraded | Balanced (v1.0) | Balanced (misgrades corrected) |", "|---|---|---|---|---|"]
    ranks = {}
    for split, col in (("dev", 1), ("test", 2)):
        scores = {}
        for m in models():
            run = m[col]
            if not run:
                continue
            fixed = {x["id"] for x in runs[run]["flips"] if VERDICTS[(x["gold"][0], x["output"])][0]}
            rs = [json.loads(l) for l in open(HERE / "results" / run / "outputs.jsonl")]
            right = lambda r, fix: (r["cell"] in ("correct", "correct_abstain")) != (fix and r["id"] in fixed)
            a = [r for r in rs if r["answerable"]]
            u = [r for r in rs if not r["answerable"]]
            bal = lambda fix: (sum(right(r, fix) for r in a) / len(a) + sum(right(r, fix) for r in u) / len(u)) / 2
            scores[m[0]] = (bal(False), bal(True))
            lines.append(f"| {m[0]} | {split} | {len(fixed)} | {100 * bal(False):.2f} | {100 * bal(True):.2f} |")
        names = list(scores)
        flips = [(x, y) for x in names for y in names
                 if scores[x][0] > scores[y][0] + 1e-9 and scores[y][1] > scores[x][1] + 1e-9]
        ties_broken = [(x, y) for x in names for y in names if abs(scores[x][0] - scores[y][0]) < 1e-9
                       and scores[x][1] > scores[y][1] + 1e-9]
        ranks[split] = (flips, ties_broken)
    for split, (flips, ties) in ranks.items():
        lines.append(f"\n{split}: {'no strict reversal' if not flips else 'REVERSED: ' + '; '.join(f'{x} vs {y}' for x, y in flips)}"
                     + ("" if not ties else "; ties broken: " + "; ".join(f"{x} now above {y}" for x, y in ties)))
    lines += ["", "Runs with a flip: " + ", ".join(f"{k} ({v['cell_flips']})" for k, v in runs.items() if v["cell_flips"]),
              "", "Same-question flips cancel in paired comparisons between models that both made them."]
    (out / "hyphens.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
