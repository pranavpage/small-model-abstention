"""config.json sanity: the settings every stage relies on."""

from abstention.config import DEFAULT_CONFIG, load_config
from abstention.freeze_grader import REWARD_SETTINGS
from abstention.grader import ABSTAIN


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
    other = load_config(DEFAULT_CONFIG)
    other["prompt"] = {**other["prompt"], "system": other["prompt"]["system"] + " "}
    assert other.eval_fingerprint() != fp
