"""sft.py: training-set construction."""

from abstention.grader import ABSTAIN


def test_sft_drops_long_answers_and_keeps_50_50():
    from abstention.sft import build_examples, target_for
    rows = [{"id": f"a{i}", "answerable": True, "answers": ["one two three"]} for i in range(6)]
    rows += [{"id": "long", "answerable": True, "answers": [" ".join(["w"] * 11), "short one"]},
             {"id": "long2", "answerable": True, "answers": [" ".join(["w"] * 12)]},
             {"id": "ok10", "answerable": True, "answers": [" ".join(["w"] * 10)]}]
    rows += [{"id": f"u{i}", "answerable": False, "answers": []} for i in range(9)]
    kept, long_ids, balance_ids = build_examples(rows, max_answer_words=10, seed=1234)
    assert long_ids == ["long2"]  # shortest gold is the target, so "long" keeps "short one"
    assert len(balance_ids) == 1 and balance_ids[0].startswith("u")
    assert sum(r["answerable"] for r in kept) * 2 == len(kept) == 16
    assert target_for(rows[6]) == "short one" and target_for(rows[-1]) == ABSTAIN
    assert build_examples(rows, 10, 1234) == (kept, long_ids, balance_ids)  # deterministic
