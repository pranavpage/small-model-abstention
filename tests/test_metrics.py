"""metrics.py: 2x2 rates reproduce the rehearsal; paired comparison, McNemar and AUC."""

import json

import pytest

from abstention.config import REPO_ROOT
from abstention.grader import GRADER_VERSION, grade
from abstention.metrics import compute_metrics

REHEARSAL = REPO_ROOT / "rehearsal" / "results"


@pytest.mark.parametrize("run", ["baseline-qwen3-0.6b", "dev-prompt-v1", "dev-prompt-v2"])
def test_metrics_reproduce_rehearsal(run):
    saved = json.load(open(REHEARSAL / run / "metrics.json"))
    assert saved["grader_version"] == GRADER_VERSION
    rows = []
    for line in open(REHEARSAL / run / "outputs.jsonl"):
        r = json.loads(line)
        rows.append({**r, **grade(r["output"], r["answerable"], r["gold"])})
    m = compute_metrics(rows, n_boot=10_000, seed=1234)
    for k in ["hallucination_rate", "over_refusal_rate", "answer_accuracy", "answer_accuracy_strict", "balanced_score"]:
        assert m[k]["value"] == pytest.approx(saved[k]["value"]), k
        assert m[k]["ci95"] == pytest.approx(saved[k]["ci95"]), k
    assert m["counts"] == saved["counts"]


def _rows(prefix, answerable, cells):
    return [{"id": f"{prefix}{i}", "answerable": answerable, "cell": c,
             "strict_cell": c} for i, c in enumerate(cells)]


def test_mcnemar_exact_known_values():
    from abstention.metrics import mcnemar_exact
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 2 ** 10)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)


def test_paired_compare_identical_runs_is_zero():
    from abstention.metrics import paired_compare
    rows = _rows("a", True, ["correct"] * 30 + ["wrong"] * 10 + ["over_refusal"] * 10) \
        + _rows("u", False, ["hallucination"] * 20 + ["correct_abstain"] * 30)
    c = paired_compare(rows, rows, n_boot=200)
    for k in ["hallucination_rate", "over_refusal_rate", "answer_accuracy", "balanced_score"]:
        assert c[k]["diff"] == 0 and c[k]["ci95"] == [0, 0] and c[k]["mcnemar_p"] == 1.0


def test_paired_compare_detects_consistent_flips_that_unpaired_intervals_miss():
    from abstention.metrics import compute_metrics, paired_compare
    # 250 unanswerable: A hallucinates on 100, B on 85 of the same 100 (15 fixed, none broken).
    a = _rows("u", False, ["hallucination"] * 100 + ["correct_abstain"] * 150) \
        + _rows("a", True, ["correct"] * 250)
    b = _rows("u", False, ["correct_abstain"] * 15 + ["hallucination"] * 85 + ["correct_abstain"] * 150) \
        + _rows("a", True, ["correct"] * 250)
    for r in a + b:
        r.update(output="x", gold=["x"], soft_refusal=False, over_30_words=False, format_ok=True)
    c = paired_compare(a, b, n_boot=2000)["hallucination_rate"]
    assert c["only_a"] == 15 and c["only_b"] == 0
    assert c["mcnemar_p"] < 1e-3 and c["ci95"][1] < 0
    ma, mb = (compute_metrics(x, n_boot=2000)["hallucination_rate"]["ci95"] for x in (a, b))
    assert ma[0] < mb[1]  # the separate intervals overlap, so "overlap" alone would call this a tie


def test_paired_compare_rejects_different_questions():
    from abstention.metrics import paired_compare
    with pytest.raises(ValueError):
        paired_compare(_rows("a", True, ["correct"]), _rows("b", True, ["correct"]), n_boot=10)


def test_auc_matches_brute_force_and_ignores_uniform_shifts():
    import random
    from abstention.metrics import auc, paired_auc_compare
    rng = random.Random(0)
    neg = [rng.random() * 0.6 for _ in range(40)] + [0.5, 0.5]
    pos = [0.3 + rng.random() * 0.7 for _ in range(30)] + [0.5]
    brute = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    assert auc(neg, pos) == pytest.approx(brute)
    assert auc([0.1, 0.2], [0.8, 0.9]) == 1.0 and auc([0.8, 0.9], [0.1, 0.2]) == 0.0
    shifted = paired_auc_compare(neg, pos, [x * 0.5 for x in neg], [x * 0.5 for x in pos], n_boot=200)
    assert shifted["diff"] == pytest.approx(0) and shifted["ci95"] == pytest.approx([0, 0])  # threshold shift only
