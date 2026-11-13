"""2x2 metrics with bootstrap intervals, computed from graded rows. Shared by eval.py and report.py."""

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
