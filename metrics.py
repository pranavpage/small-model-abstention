"""2x2 metrics with bootstrap intervals, computed from graded rows. Shared by eval.py and report.py.

paired_compare() compares two models on the same questions: a paired bootstrap of each difference
(both models resampled with the same question indices) and an exact McNemar test on the questions
that flipped. Use it, not overlapping intervals, to decide whether two models differ.
"""

from math import comb

import numpy as np

from grader import DEFAULT_H, DEFAULT_R, drift_stats, reward

RATE_KEYS = ["hallucination_rate", "over_refusal_rate", "answer_accuracy", "answer_accuracy_strict",
             "precision_when_answering", "balanced_score", "balanced_score_strict"]


def _point(ac, acs, ar, uh):
    answers_given = (len(ac) - ar.sum()) + uh.sum()
    abst = 1 - uh.mean()
    return {
        "hallucination_rate": uh.mean(),
        "over_refusal_rate": ar.mean(),
        "answer_accuracy": ac.mean(),
        "answer_accuracy_strict": acs.mean(),
        "precision_when_answering": ac.sum() / answers_given if answers_given else float("nan"),
        "correct_abstain_rate": abst,
        "balanced_score": (ac.mean() + abst) / 2,
        "balanced_score_strict": (acs.mean() + abst) / 2,
    }


def compute_metrics(rows, n_boot=10_000, seed=1234):
    a = [r for r in rows if r["answerable"]]
    u = [r for r in rows if not r["answerable"]]
    a_correct = np.array([r["cell"] == "correct" for r in a])
    a_correct_strict = np.array([r["strict_cell"] == "correct" for r in a])
    a_refuse = np.array([r["cell"] == "over_refusal" for r in a])
    u_halluc = np.array([r["cell"] == "hallucination" for r in u])

    point = _point(a_correct, a_correct_strict, a_refuse, u_halluc)
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        ia = rng.integers(0, len(a), len(a))
        iu = rng.integers(0, len(u), len(u))
        for k, v in _point(a_correct[ia], a_correct_strict[ia], a_refuse[ia], u_halluc[iu]).items():
            boots[k].append(v)
    out = {}
    for k, v in point.items():
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        out[k] = {"value": float(v), "ci95": [float(lo), float(hi)]}

    rewards = [reward(r["output"], r["answerable"], r["gold"]) for r in rows]
    out["answer_scores"] = drift_stats(rewards)
    out["reward_default"] = {
        "h": DEFAULT_H, "r": DEFAULT_R,
        "mean": float(np.mean([x["reward"] for x in rewards])),
        "mean_base": float(np.mean([x["base"] for x in rewards])),
        "mean_format_penalty": float(np.mean([x["format_penalty"] for x in rewards])),
    }
    out["counts"] = {
        "answerable": len(a),
        "unanswerable": len(u),
        "answerable_correct": int(a_correct.sum()),
        "answerable_correct_strict": int(a_correct_strict.sum()),
        "answerable_wrong": int(len(a) - a_correct.sum() - a_refuse.sum()),
        "over_refusal": int(a_refuse.sum()),
        "hallucination": int(u_halluc.sum()),
        "correct_abstain": int((~u_halluc).sum()),
        "soft_refusals_counted_as_answers": sum(r["soft_refusal"] for r in rows),
        "outputs_over_30_words": sum(r["over_30_words"] for r in rows),
        "format_violations": sum(not r["format_ok"] for r in rows),
    }
    return out


def format_table(m) -> str:
    c = m["counts"]
    lines = [
        "2x2                     answers            says I don't know",
        f"answerable   ({c['answerable']})    correct {c['answerable_correct']:>3} / wrong {c['answerable_wrong']:>3}"
        f"   over-refusal {c['over_refusal']:>3}",
        f"unanswerable ({c['unanswerable']})    hallucination {c['hallucination']:>3}"
        f"        correct abstain {c['correct_abstain']:>3}",
        "",
    ]
    for k in RATE_KEYS:
        v, (lo, hi) = m[k]["value"], m[k]["ci95"]
        lines.append(f"{k:<26} {100*v:5.1f}%  [{100*lo:5.1f}, {100*hi:5.1f}]")
    s, rw = m["answer_scores"], m["reward_default"]
    lines.append("")
    if s["n_scored"]:
        lines.append(f"answer score (answered, answerable n={s['n_scored']}): strict {s['strict_mean']:.3f}  "
                     f"lenient {s['lenient_mean']:.3f}  gap {s['lenient_minus_strict']:+.3f}")
    lines.append(f"mean reward (h={rw['h']}, r={rw['r']}): {rw['mean']:+.3f}  "
                 f"(base {rw['mean_base']:+.3f}, format {rw['mean_format_penalty']:+.3f})")
    lines.append(f"soft refusals counted as answers: {c['soft_refusals_counted_as_answers']}  |  "
                 f"format violations: {c['format_violations']} (over 30 words: {c['outputs_over_30_words']})")
    return "\n".join(lines)


# Per-question binary outcomes that McNemar can test: (metric, stratum, indicator on a graded row).
# "right" = correct answer on answerable or abstain on unanswerable; its mean over both strata equals
# balanced_score when the strata are the same size (they are: every split is 50/50).
PAIRED_BINARY = [
    ("hallucination_rate", "unanswerable", lambda r: r["cell"] == "hallucination"),
    ("over_refusal_rate", "answerable", lambda r: r["cell"] == "over_refusal"),
    ("answer_accuracy", "answerable", lambda r: r["cell"] == "correct"),
    ("answer_accuracy_strict", "answerable", lambda r: r["strict_cell"] == "correct"),
    ("balanced_score", "all", lambda r: r["cell"] in ("correct", "correct_abstain")),
    ("balanced_score_strict", "all", lambda r: (r["strict_cell"] == "correct") if r["answerable"]
                                               else r["cell"] == "correct_abstain"),
]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: b and c are the discordant counts (only A had it / only B had it)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _arrays(rows_by_id, a_ids, u_ids):
    a = [rows_by_id[i] for i in a_ids]
    u = [rows_by_id[i] for i in u_ids]
    return (np.array([r["cell"] == "correct" for r in a]),
            np.array([r["strict_cell"] == "correct" for r in a]),
            np.array([r["cell"] == "over_refusal" for r in a]),
            np.array([r["cell"] == "hallucination" for r in u]))


def paired_compare(rows_a, rows_b, n_boot=10_000, seed=1234):
    """Difference B - A for every rate, with a paired, stratified bootstrap interval and, for per-question
    binary outcomes, the flip counts and exact McNemar p-value. Both runs must cover the same questions."""
    A = {r["id"]: r for r in rows_a}
    B = {r["id"]: r for r in rows_b}
    if A.keys() != B.keys():
        raise ValueError(f"runs cover different questions ({len(A.keys() ^ B.keys())} ids differ)")
    if any(A[i]["answerable"] != B[i]["answerable"] for i in A):
        raise ValueError("answerable flag differs between runs for the same id")
    a_ids = sorted(i for i in A if A[i]["answerable"])
    u_ids = sorted(i for i in A if not A[i]["answerable"])
    xa, xb = _arrays(A, a_ids, u_ids), _arrays(B, a_ids, u_ids)

    pa, pb = _point(*xa), _point(*xb)
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in RATE_KEYS}
    for _ in range(n_boot):
        ia = rng.integers(0, len(a_ids), len(a_ids))
        iu = rng.integers(0, len(u_ids), len(u_ids))
        da = _point(xa[0][ia], xa[1][ia], xa[2][ia], xa[3][iu])
        db = _point(xb[0][ia], xb[1][ia], xb[2][ia], xb[3][iu])
        for k in RATE_KEYS:
            boots[k].append(db[k] - da[k])

    out = {}
    for k in RATE_KEYS:
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        out[k] = {"a": float(pa[k]), "b": float(pb[k]), "diff": float(pb[k] - pa[k]),
                  "ci95": [float(lo), float(hi)]}
    strata = {"answerable": a_ids, "unanswerable": u_ids, "all": a_ids + u_ids}
    for k, stratum, ind in PAIRED_BINARY:
        ids = strata[stratum]
        only_a = sum(ind(A[i]) and not ind(B[i]) for i in ids)
        only_b = sum(ind(B[i]) and not ind(A[i]) for i in ids)
        out[k].update({"n": len(ids), "only_a": only_a, "only_b": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b)})
    out["n_answerable"], out["n_unanswerable"] = len(a_ids), len(u_ids)
    return out


def format_compare(c, name_a="A", name_b="B") -> str:
    lines = [f"A = {name_a}   B = {name_b}   paired on {c['n_answerable']} answerable + "
             f"{c['n_unanswerable']} unanswerable questions; diff = B - A",
             f"{'metric':<26}{'A':>8}{'B':>8}{'diff':>8}  {'paired 95% CI':<16}{'only A':>8}{'only B':>8}{'McNemar p':>11}"]
    for k in RATE_KEYS:
        x = c[k]
        row = (f"{k:<26}{100*x['a']:7.1f}%{100*x['b']:7.1f}%{100*x['diff']:+8.1f}  "
               f"[{100*x['ci95'][0]:+5.1f}, {100*x['ci95'][1]:+5.1f}]  ")
        if "mcnemar_p" in x:
            row += f"{x['only_a']:>8}{x['only_b']:>8}{x['mcnemar_p']:>11.2g}"
        lines.append(row)
    lines.append("only A / only B: questions where the outcome holds for that model and not the other "
                 "(balanced: questions only that model got right). Rates in percent, diff in points.")
    return "\n".join(lines)
