"""dpo.py: preference-pair construction and the committed pair set."""

import json

from abstention.config import TRACKED_RESULTS
from abstention.grader import ABSTAIN, grade


def _sample_row(qid, answerable, gold, outputs):
    return {"id": qid, "answerable": answerable, "gold": gold,
            "samples": [{"output": o, **grade(o, answerable, gold)} for o in outputs]}


def test_dpo_pairs_filter_partial_spans_and_balance():
    from abstention.dpo import build_pairs
    rows = [
        # partial span "ADSL-broadband" is dropped; "cable" (wrong) and "I don't know" are kept; chosen = sampled
        _sample_row("a1", True, ["ADSL"], ["ADSL", "ADSL-broadband", "cable", ABSTAIN]),
        # nothing correct sampled -> chosen falls back to the shortest gold span
        _sample_row("a2", True, ["John", "King John"], [ABSTAIN, "Geoffrey", ABSTAIN, "Henry II"]),
        # gold longer than 10 words and never sampled -> question skipped
        _sample_row("a3", True, ["one two three four five six seven eight nine ten eleven"], [ABSTAIN] * 4),
        _sample_row("u1", False, [], ["Paris", ABSTAIN, "Paris", "Lyon"]),
        _sample_row("u2", False, [], ["1925", ABSTAIN, ABSTAIN, ABSTAIN]),
        _sample_row("u3", False, [], [ABSTAIN] * 4),
    ]
    pairs, stats = build_pairs(rows, max_answer_words=10, seed=0)
    assert stats["dropped_partial_span_rejected"] == 1 and stats["dropped_long_gold_questions"] == 1
    assert stats["answerable_pairs_before_balancing"] == 5  # a1: 2, a2: 3
    assert stats["unanswerable_pairs"] == 3 and stats["answerable_pairs"] == 3  # balanced down to 3 + 3
    assert not any(p["rejected"] == "ADSL-broadband" for p in pairs)
    for p in pairs:
        if p["answerable"]:
            assert p["chosen"] == {"a1": "ADSL", "a2": "John"}[p["id"]] and p["rejected"] != p["chosen"]
        else:
            assert p["chosen"] == ABSTAIN and p["rejected"] != ABSTAIN


def test_dpo_idk_balanced_pairs_have_equal_idk_chosen_and_rejected():
    from abstention.dpo import build_pairs
    rows = [
        _sample_row("a1", True, ["ADSL"], ["ADSL", ABSTAIN, "cable", "cable"]),  # 1 IDK-rejected + 1 wrong pair
        _sample_row("a2", True, ["John"], [ABSTAIN, "Geoffrey", ABSTAIN, "Henry II"]),  # 1 IDK + 2 wrong
        _sample_row("a3", True, ["Rome"], [ABSTAIN, "Rome", "Rome", ABSTAIN]),  # 1 IDK
        _sample_row("u1", False, [], ["Paris", ABSTAIN, "Lyon", "Nice"]),  # 3 hallucination pairs
        _sample_row("u2", False, [], ["1925", ABSTAIN, ABSTAIN, ABSTAIN]),  # 1
    ]
    pairs, stats = build_pairs(rows, max_answer_words=10, seed=0, mode="idk_balanced")
    assert stats["idk_chosen"] == stats["idk_rejected"] == 3  # min(3 IDK-rejected, 4 unanswerable)
    assert stats["answerable_rejected"] == {"over_refusal": 3, "wrong": 3}  # all wrong pairs kept
    assert stats["pairs"] == 9


def test_dpo_variants_and_candidates(cfg):
    from abstention.dpo import candidates
    d = cfg["dpo"]
    for name, v in d["variants"].items():
        assert v["pair_mode"] in ("class_balanced", "idk_balanced")
        c = candidates(cfg, name)
        assert len(c) == len(v["learning_rates"]) * d["epochs"] and len({t for t, _ in c}) == len(v["learning_rates"])
    assert d["variants"]["dpo"]["learning_rates"] == [5e-6, 5e-5]  # the first variant, already run: keep it reproducible


def test_dpo_first_variant_pairs_reproduce():
    """The committed pairs of the first DPO variant must be rebuilt byte-identically from the committed samples."""
    from abstention.dpo import build_pairs
    samples = [json.loads(l) for l in open(TRACKED_RESULTS / "dpo-samples" / "samples.jsonl")]
    committed = [json.loads(l) for l in open(TRACKED_RESULTS / "dpo-train" / "pairs.jsonl")]
    assert build_pairs(samples, 10, 1234, "class_balanced")[0] == committed
