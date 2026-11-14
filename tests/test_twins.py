"""twins.py and dpo_twin.py: twin matching, held-out selection and the twin DPO pairing."""

from abstention.grader import ABSTAIN


def _rec(qid, question, answers, context="P1", title="T"):
    return {"id": qid, "title": title, "context": context, "question": question, "answers": answers,
            "answerable": bool(answers)}


def test_twins_match_best_edit_one_to_one_with_threshold_and_margin():
    from abstention.twins import match_twins
    paras = {
        "P1": [_rec("a1", "What did Saarinen use for inspiration?", ["San Gimignano"]),
               _rec("a2", "When was the rink built?", ["1958"]),
               _rec("u1", "What did Saarinen not use for inspiration?", []),
               _rec("u2", "What did Saarinen never use for inspiration?", []),   # same best match: a1 taken by u1
               _rec("u3", "Who paid for the stadium?", [])],                      # nothing similar
        "P2": [_rec("b1", "What does AC stand for?", ["alternating current"], "P2"),
               _rec("b2", "What does DC stand for?", ["direct current"], "P2"),
               _rec("v1", "What does EC stand for?", [], "P2")],                  # tie: no margin
        "P3": [_rec("c1", "Who wrote the long list of famous names?", ["one two three four five six seven eight nine "
                                                                       "ten eleven"], "P3"),
               _rec("w1", "Who wrote the short list of famous names?", [], "P3")],  # answer too long
    }
    twins = match_twins(paras, 0.8, 0.1, 10)
    assert [(t["unanswerable"]["id"], t["answerable"]["id"]) for t in twins] == [("u1", "a1")]
    assert twins[0]["answerable"]["answer"] == "San Gimignano" and twins[0]["context"] == "P1"


def test_twins_selection_anchors_on_dpo_and_holds_out_untouched_paragraphs():
    from abstention.twins import heldout_rows, select
    paras = {c: [_rec(f"{c}a", "q", ["x"], c), _rec(f"{c}u", "q not", [], c)] for c in ("P1", "P2", "P3", "P4", "P5")}
    paras["P4"].append(_rec("P4dev", "other", ["y"], "P4"))
    twins = [{"context": c, "title": "T", "unanswerable": {"id": f"{c}u", "question": "q not"},
              "answerable": {"id": f"{c}a", "question": "q", "answer": "x", "answers": ["x"]}} for c in paras]
    split_of = {"P1u": "dpo", "P2a": "dpo", "P3u": "sft", "P4a": "dpo", "P4dev": "dev"}
    train, held = select(twins, split_of, paras, "dpo", ["sft", "grpo", "dev", "test"], 1, 0)
    assert [t["context"] for t in train] == ["P1", "P2"]   # P3 touches sft, P4 is a dev paragraph
    assert [t["context"] for t in held] == ["P5"]          # the only paragraph with no split question
    rows = heldout_rows(held)
    assert [(r["id"], r["answerable"]) for r in rows] == [("P5a", True), ("P5u", False)]


def test_twin_losses_ignore_the_shift_their_pairing_cannot_see():
    """input side: blind to a response moving the same way after both questions (a threshold shift);
    output side: blind to a question moving both responses the same way. Both see a real twin distinction."""
    import numpy as np
    from abstention.dpo_twin import margins
    rng = np.random.default_rng(0)
    delta = rng.normal(size=(5, 4))  # (q,a) (q,IDK) (q',a) (q',IDK)
    threshold_shift = delta + np.array([0.0, 2.0, 0.0, 2.0])   # IDK up after every question
    question_shift = delta + np.array([1.5, 1.5, -0.7, -0.7])  # each question shifts both of its responses
    m = lambda d, s: np.stack(list(margins(d, s).values()))
    assert np.allclose(m(threshold_shift, "input"), m(delta, "input"))
    assert not np.allclose(m(threshold_shift, "output"), m(delta, "output"))
    assert np.allclose(m(question_shift, "output"), m(delta, "output"))
    assert not np.allclose(m(question_shift, "input"), m(delta, "input"))
    learned = delta + np.array([1.0, -1.0, -1.0, 1.0])  # answer on q, abstain on q': both margins rise by 2
    for side in ("input", "output"):
        assert np.allclose(m(learned, side), m(delta, side) + 2)


def test_twin_pairs_balance_idk(cfg):
    from abstention.dpo_twin import candidates, pair_records
    twin = {"twin_id": "t0", "answerable": {"question": "q", "answer": "x"}, "unanswerable": {"question": "q not"}}
    inp, out = pair_records(twin, "input"), pair_records(twin, "output")
    assert [p["response"] for p in inp] == ["x", ABSTAIN] and inp[0]["chosen_question"] == inp[1]["rejected_question"]
    assert sum(p["chosen"] == ABSTAIN for p in out) == sum(p["rejected"] == ABSTAIN for p in out) == 1
    d = cfg["dpo_twin"]
    assert {v["side"] for v in d["variants"].values()} == {"input", "output"}
    assert len({tuple(v["learning_rates"]) for v in d["variants"].values()}) == 1  # control differs only in pairing
    for name, v in d["variants"].items():
        assert len(candidates(cfg, name)) == len(v["learning_rates"]) * d["epochs"]
    assert d["gate"] == cfg["grpo"]["gate"]
