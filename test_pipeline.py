"""Checks that run without a GPU: config sanity and metrics regression on the rehearsal outputs."""

import json
from pathlib import Path

import pytest

from config import load_config
from freeze_grader import REWARD_SETTINGS
from grader import ABSTAIN, GRADER_VERSION, grade
from metrics import compute_metrics

HERE = Path(__file__).parent
REHEARSAL = HERE / "rehearsal" / "results"


@pytest.fixture(scope="module")
def cfg():
    return load_config(HERE / "config.json")


def test_config_has_required_sections(cfg):
    for key in ["paths", "models", "prompt", "data", "eval", "vllm", "lora", "sft", "dpo", "grpo"]:
        assert key in cfg, key
    for m in ["main", "debug", "ceiling"]:
        assert cfg["models"][m]


def test_prompt_asks_for_exact_abstain_phrase(cfg):
    assert ABSTAIN in cfg["prompt"]["system"]
    assert "{context}" in cfg["prompt"]["user_template"] and "{question}" in cfg["prompt"]["user_template"]


def test_eval_settings_match_brief(cfg):
    assert cfg["eval"]["temperature"] == 0
    assert cfg["eval"]["max_new_tokens"] == 64


def test_split_sizes_are_even_for_50_50(cfg):
    sizes = list(cfg["data"]["train_splits"].values()) + [cfg["data"]["test_size"]]
    assert all(n % 2 == 0 for n in sizes)


def test_grpo_runs_are_covered_by_frozen_fixture(cfg):
    frozen = set(REWARD_SETTINGS)
    for name, run in cfg["grpo"]["runs"].items():
        assert (run["h"], run["r"]) in frozen, f"{name} uses a reward setting not covered by the grader fixture"


def test_paths_resolve_under_root(cfg):
    assert cfg.data_dir == cfg.root / "data"
    assert cfg.model_id("main") == cfg["models"]["main"]
    assert cfg.model_id("org/other-model") == "org/other-model"


def test_eval_fingerprint_changes_with_prompt(cfg):
    fp = cfg.eval_fingerprint()
    other = load_config(HERE / "config.json")
    other["prompt"] = {**other["prompt"], "system": other["prompt"]["system"] + " "}
    assert other.eval_fingerprint() != fp


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


def test_sft_drops_long_answers_and_keeps_50_50():
    from sft import build_examples, target_for
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


def _rows(prefix, answerable, cells):
    return [{"id": f"{prefix}{i}", "answerable": answerable, "cell": c,
             "strict_cell": c} for i, c in enumerate(cells)]


def test_mcnemar_exact_known_values():
    from metrics import mcnemar_exact
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 2 ** 10)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)


def test_paired_compare_identical_runs_is_zero():
    from metrics import paired_compare
    rows = _rows("a", True, ["correct"] * 30 + ["wrong"] * 10 + ["over_refusal"] * 10) \
        + _rows("u", False, ["hallucination"] * 20 + ["correct_abstain"] * 30)
    c = paired_compare(rows, rows, n_boot=200)
    for k in ["hallucination_rate", "over_refusal_rate", "answer_accuracy", "balanced_score"]:
        assert c[k]["diff"] == 0 and c[k]["ci95"] == [0, 0] and c[k]["mcnemar_p"] == 1.0


def test_paired_compare_detects_consistent_flips_that_unpaired_intervals_miss():
    from metrics import compute_metrics, paired_compare
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
    from metrics import paired_compare
    with pytest.raises(ValueError):
        paired_compare(_rows("a", True, ["correct"]), _rows("b", True, ["correct"]), n_boot=10)
