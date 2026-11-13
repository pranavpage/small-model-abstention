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
