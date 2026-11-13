"""Grader tests built from real outputs seen in the baseline runs. Run: uv run python -m pytest -q"""

import json
from pathlib import Path

import pytest

from grader import GRADER_VERSION, drift_stats, grade, is_abstain, reward

FIXTURE = Path(__file__).parent / "fixtures" / "grader_golden.jsonl"


def cell(output, golds):
    return grade(output, answerable=bool(golds), golds=golds)["cell"]


# ---------- Freeze: grader must reproduce the golden fixture exactly ----------

def _golden():
    lines = FIXTURE.read_text().splitlines()
    return json.loads(lines[0]), [json.loads(l) for l in lines[1:]]


def test_fixture_matches_grader_version():
    header, _ = _golden()
    assert header["grader_version"] == GRADER_VERSION, (
        "GRADER_VERSION changed: regenerate the fixture with freeze_grader.py, deliberately")


def test_grader_reproduces_golden_fixture():
    _, cases = _golden()
    assert len(cases) == 600
    mismatches = []
    for c in cases:
        got = grade(c["output"], c["answerable"], c["gold"])
        if got != c["grade"]:
            mismatches.append((c["id"], c["output"], c["grade"], got))
        for setting, want in c["rewards"].items():
            h, r = (float(x.split("=")[1]) for x in setting.split(","))
            if reward(c["output"], c["answerable"], c["gold"], h, r)["reward"] != want:
                mismatches.append((c["id"], setting))
    assert not mismatches, f"grader is frozen at v{GRADER_VERSION}; {len(mismatches)} mismatches, e.g. {mismatches[:3]}"


# ---------- Abstain detection: exact phrase only ----------

def test_abstain_exact_phrase_variants():
    assert is_abstain("I don't know")
    assert is_abstain("I don't know.")
    assert is_abstain("i dont know")
    assert is_abstain("I don’t know.")


def test_abstain_own_words_counts_as_answer():
    assert cell("The size of Melbourne is not mentioned in the paragraph.", []) == "hallucination"
    assert cell("The form of Euler numbers is not mentioned in the paragraph. I don't know.", []) == "hallucination"


# ---------- Lenient vs strict answer scoring ----------

def test_exact_and_short_answers_correct_under_both():
    for out, gold in [("preventable financial problems", ["preventable financial problems"]),
                      ("The electric eels.", ["electric eels"])]:
        g = grade(out, True, gold)
        assert g["cell"] == g["strict_cell"] == "correct"


def test_sentence_containing_gold_lenient_correct_strict_wrong():
    for out, gold in [("The Rhine is the longest river in Germany.", ["Germany"]),
                      ("The atomic number of the element oxygen is 8.", ["8"]),
                      ("Soybeans were coming from Manchuria.", ["Manchuria"]),
                      ("Eisenhower began smoking cigarettes at West Point.", ["West Point"]),
                      ("Correlated color temperature (CCT)", ["CCT"])]:
        g = grade(out, True, gold)
        assert g["cell"] == "correct" and g["strict_cell"] == "wrong", out
        assert g["lenient_score"] == 1.0 and g["strict_score"] < 0.5


def test_containment_is_word_bounded():
    assert cell("In 1988 it was founded.", ["198"]) == "wrong"
    assert cell("The Catholic Church.", ["Catholics"]) == "wrong"


def test_long_paragraph_dump_not_credited():
    out = ("The IPCC has since acknowledged that the date is incorrect, while reaffirming that the "
           "conclusion in the final summary was robust. The date of 2035 has been correctly quoted "
           "by the IPCC from the WWF report, which")
    g = grade(out, True, ["the IPCC from the WWF report"])
    assert g["cell"] == "wrong" and g["lenient_score"] == g["strict_score"]


def test_wrong_answers_stay_wrong():
    assert cell("1760.", ["1840"]) == "wrong"
    assert cell("The answer is: Germany.", ["Soviet Union"]) == "wrong"
    assert cell("neuroticism.", ["chronic pain"]) == "wrong"


def test_refusal_on_answerable_is_over_refusal():
    assert cell("I don't know", ["Paris"]) == "over_refusal"


# ---------- Reward ----------

def test_reward_table_defaults():
    assert reward("Germany", True, ["Germany"])["reward"] == 1.0
    assert reward("I don't know", True, ["Germany"])["reward"] == -0.5
    assert reward("I don't know", False, [])["reward"] == 1.0
    assert reward("Germany", False, [])["reward"] == -1.0


def test_reward_answerable_uses_lenient_score():
    assert reward("The Rhine is the longest river in Germany.", True, ["Germany"])["reward"] == 1.0
    assert reward("Mersenne primes.", True, ["primorial primes"])["reward"] == pytest.approx(0.5)


def test_reward_sweep_settings():
    assert reward("Germany", False, [], h=3.0, r=0.0)["reward"] == -3.0
    assert reward("I don't know", True, ["Germany"], h=3.0, r=0.0)["reward"] == 0.0
    assert reward("Germany", False, [], h=0.3)["reward"] == pytest.approx(-0.3)


def test_length_penalty_makes_copying_a_sentence_cost_more_than_it_earns():
    gold = ["Germany"]
    copied = ("The Rhine, which rises in the Swiss Alps and flows north through several countries before "
              "reaching the North Sea, is the longest river entirely within the borders of Germany even today.")
    assert len(copied.split()) > 30
    r = reward(copied, True, gold)
    assert r["format_penalty"] == -0.5
    assert r["reward"] < 0 < reward("Germany", True, gold)["reward"]
    # Even when the copied sentence would otherwise earn something, the penalty outweighs it.
    assert r["base"] < 0.5


def test_format_penalty_cases():
    assert reward("", True, ["x"])["format_penalty"] == -0.5
    assert reward("Not mentioned. I don't know.", False, [])["format_penalty"] == -0.5
    assert reward("Not mentioned. I don't know.", False, [])["reward"] == -1.5
    assert reward("I don't know.", False, [])["format_penalty"] == 0.0
    assert reward(" ".join(["word"] * 30), True, ["x"])["format_penalty"] == 0.0
    assert reward(" ".join(["word"] * 31), True, ["x"])["format_penalty"] == -0.5


def test_drift_stats_flags_padding():
    batch = [reward("The Rhine is the longest river in Germany.", True, ["Germany"]),
             reward("Hagen", True, ["Hagen"]),
             reward("I don't know", False, [])]
    d = drift_stats(batch)
    assert d["n_scored"] == 2
    assert d["lenient_mean"] == 1.0 and d["strict_mean"] < 0.7 and d["lenient_minus_strict"] > 0.3
