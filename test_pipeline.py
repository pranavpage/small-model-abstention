"""Checks that run without a GPU: config sanity and metrics regression on the rehearsal outputs."""

import json
from collections import Counter
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


def _sample_row(qid, answerable, gold, outputs):
    return {"id": qid, "answerable": answerable, "gold": gold,
            "samples": [{"output": o, **grade(o, answerable, gold)} for o in outputs]}


def test_dpo_pairs_filter_partial_spans_and_balance():
    from dpo import build_pairs
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
    from dpo import build_pairs
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
    from dpo import candidates
    d = cfg["dpo"]
    for name, v in d["variants"].items():
        assert v["pair_mode"] in ("class_balanced", "idk_balanced")
        c = candidates(cfg, name)
        assert len(c) == len(v["learning_rates"]) * d["epochs"] and len({t for t, _ in c}) == len(v["learning_rates"])
    assert d["variants"]["dpo"]["learning_rates"] == [5e-6, 5e-5]  # the first variant, already run: keep it reproducible


def test_dpo_first_variant_pairs_reproduce():
    """The committed pairs of the first DPO variant must be rebuilt byte-identically from the committed samples."""
    from dpo import build_pairs
    samples = [json.loads(l) for l in open(HERE / "results" / "dpo-samples" / "samples.jsonl")]
    committed = [json.loads(l) for l in open(HERE / "results" / "dpo-train" / "pairs.jsonl")]
    assert build_pairs(samples, 10, 1234, "class_balanced")[0] == committed


def _sampled(qid, answerable, gold, outputs):
    return {"id": qid, "answerable": answerable, "gold": gold, "samples": [{"output": o} for o in outputs]}


def test_grpo_pool_keeps_varying_groups_tops_up_and_balances():
    from grpo import build_pool
    grpo_rows = [
        _sampled("a_mixed", True, ["Rome"], ["Rome"] * 6 + [ABSTAIN] * 2),         # varies: kept
        _sampled("a_partial", True, ["King John"], ["King John"] * 7 + ["John"]),  # 1.0 vs 0.67: kept
        _sampled("a_guess", True, ["Rome"], [ABSTAIN] * 6 + ["Paris"] * 2),        # IDK vs wrong only: dropped
        _sampled("a_same", True, ["Rome"], ["Rome"] * 8),                          # all 1.0: dropped
        _sampled("a_all_idk", True, ["Rome"], [ABSTAIN] * 8),                      # same mistake 8x: dropped
        _sampled("u_mixed", False, [], [ABSTAIN] * 7 + ["Paris"]),                 # varies: kept
        _sampled("u_same", False, [], [ABSTAIN] * 8),                              # dropped
    ]
    topup = [_sampled("d_ans", True, ["Nile"], ["Nile", "Nile", "Amazon", "Nile"]),  # answerable: not topped up
             _sampled("d_unans", False, [], ["Paris", ABSTAIN, ABSTAIN, ABSTAIN]),    # added
             _sampled("d_same", False, [], ["Paris"] * 4)]                            # zero variance: not added
    for x in grpo_rows + topup:
        for smp in x["samples"]:
            smp["cell"] = grade(smp["output"], x["answerable"], x["gold"])["cell"]
    pool, stats = build_pool(grpo_rows, topup, h=1.0, r=0.5, seed=0)
    assert {p["id"] for p in pool} == {"a_mixed", "a_partial", "u_mixed", "d_unans"}
    assert stats["grpo_split"]["answerable"]["varying_without_correct_dropped"] == 1
    assert stats["grpo_split"]["answerable"]["zero_variance"] == 2
    assert stats["added_from_topup"] == {"answerable": 0, "unanswerable": 1}
    assert stats["pool_rows_by_class"] == {"answerable": 2, "unanswerable": 2} and stats["repeat_factor"] == 1.0
    assert len({p["uid"] for p in pool}) == len(pool)
    assert {p["split"] for p in pool if p["id"] == "d_unans"} == {"dpo"}
    assert build_pool(grpo_rows, topup, 1.0, 0.5, 0) == (pool, stats)  # deterministic

    pool, stats = build_pool(grpo_rows, topup, 1.0, 0.5, 0, need_correct=False)
    assert "a_guess" in {p["id"] for p in pool} and stats["pool_rows_by_class"]["unanswerable"] == 3


def test_grpo_pool_balancing_remainder_is_a_subset():
    from grpo import build_pool
    rows = [_sampled(f"a{i}", True, ["x"], ["x"] * 4 + [ABSTAIN] * 4) for i in range(7)]
    rows += [_sampled(f"u{i}", False, [], [ABSTAIN] * 4 + ["y"] * 4) for i in range(3)]
    for x in rows:
        for smp in x["samples"]:
            smp["cell"] = grade(smp["output"], x["answerable"], x["gold"])["cell"]
    pool, stats = build_pool(rows, [], 1.0, 0.5, seed=1)
    u = Counter(p["id"] for p in pool if not p["answerable"])
    assert sum(u.values()) == 7 and sorted(u.values()) == [2, 2, 3]
    assert stats["repeated_class"] == "unanswerable" and stats["repeat_factor"] == round(7 / 3, 3)


def test_auc_matches_brute_force_and_ignores_uniform_shifts():
    import random
    from metrics import auc, paired_auc_compare
    rng = random.Random(0)
    neg = [rng.random() * 0.6 for _ in range(40)] + [0.5, 0.5]
    pos = [0.3 + rng.random() * 0.7 for _ in range(30)] + [0.5]
    brute = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    assert auc(neg, pos) == pytest.approx(brute)
    assert auc([0.1, 0.2], [0.8, 0.9]) == 1.0 and auc([0.8, 0.9], [0.1, 0.2]) == 0.0
    shifted = paired_auc_compare(neg, pos, [x * 0.5 for x in neg], [x * 0.5 for x in pos], n_boot=200)
    assert shifted["diff"] == pytest.approx(0) and shifted["ci95"] == pytest.approx([0, 0])  # threshold shift only


def test_grpo_gate():
    from grpo import gate_passes
    assert gate_passes(0.01, 0.05, 8, 5, max_p=0.1, min_gain=0.02)        # significant
    assert not gate_passes(0.01, 0.3, 8, 5, max_p=0.1, min_gain=0.02)     # small and noisy
    assert gate_passes(6 / 300, 0.15, 20, 14, max_p=0.1, min_gain=0.02)   # +2 points, gained > lost
    assert not gate_passes(0.02, 0.15, 10, 10, max_p=0.1, min_gain=0.02)  # +2 points but not more gained
    assert not gate_passes(-0.01, 0.01, 2, 5, max_p=0.1, min_gain=0.02)   # significantly worse


def test_grpo_reward_counts_zero_variance_groups_per_class():
    from grpo import Reward
    rf = Reward(h=1.0, r=0.5, num_generations=4)
    logged = {}
    out = rf(prompts=None, completions=["Rome"] * 4 + [ABSTAIN, "Paris", ABSTAIN, ABSTAIN] + [ABSTAIN] * 4,
             uid=["a#0"] * 4 + ["u#0"] * 4 + ["u2#0"] * 4, answerable=[True] * 4 + [False] * 8,
             gold=[["Rome"]] * 4 + [[]] * 8, log_metric=lambda k, v: logged.__setitem__(k, v))
    assert out[:5] == [1.0] * 4 + [1.0] and out[5] == -1.0
    assert logged["zero_var_share/answerable"] == 1.0 and logged["zero_var_share/unanswerable"] == 0.5
    w = rf.window_stats()
    assert w["answerable"]["groups"] == 1 and w["unanswerable"]["zero_var_share"] == 0.5
    with pytest.raises(AssertionError):
        rf(prompts=None, completions=["x"] * 3, uid=["a"] * 3, answerable=[True] * 3, gold=[["x"]] * 3)


def test_grpo_config(cfg):
    g = cfg["grpo"]
    assert (g["questions_per_step"] * g["num_generations"]) % g["per_device_batch_size"] == 0
    assert g["pool"]["filter_run"] in g["runs"]
    assert g["max_completion_length"] == cfg["eval"]["max_new_tokens"]


def test_grpo_committed_pool_reproduces(cfg):
    """results/grpo-pool/pool.jsonl must be rebuilt byte-identically from the committed samples and config."""
    from grpo import make_pool
    _, stats, text, _ = make_pool(cfg, HERE / "results")
    assert text == (HERE / "results" / "grpo-pool" / "pool.jsonl").read_text()
    assert stats["pool_rows_by_class"]["answerable"] == stats["pool_rows_by_class"]["unanswerable"]


def test_grpo_run_learning_rate_override(cfg):
    from grpo import run_learning_rate
    assert run_learning_rate(cfg, "G-default") == cfg["grpo"]["learning_rate"] == 1e-6  # the run already done
    assert run_learning_rate(cfg, "G-equal") == 1e-5
    assert cfg["grpo"]["runs"]["G-equal"]["h"] == 1.0 and cfg["grpo"]["runs"]["G-equal"]["r"] == 1.0
