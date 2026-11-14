"""grpo.py: question pool, dev gate, reward logging and per-run settings."""

from collections import Counter

import pytest

from abstention.config import TRACKED_RESULTS
from abstention.grader import ABSTAIN, grade


def _sampled(qid, answerable, gold, outputs):
    return {"id": qid, "answerable": answerable, "gold": gold, "samples": [{"output": o} for o in outputs]}


def test_grpo_pool_keeps_varying_groups_tops_up_and_balances():
    from abstention.grpo import build_pool
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
    from abstention.grpo import build_pool
    rows = [_sampled(f"a{i}", True, ["x"], ["x"] * 4 + [ABSTAIN] * 4) for i in range(7)]
    rows += [_sampled(f"u{i}", False, [], [ABSTAIN] * 4 + ["y"] * 4) for i in range(3)]
    for x in rows:
        for smp in x["samples"]:
            smp["cell"] = grade(smp["output"], x["answerable"], x["gold"])["cell"]
    pool, stats = build_pool(rows, [], 1.0, 0.5, seed=1)
    u = Counter(p["id"] for p in pool if not p["answerable"])
    assert sum(u.values()) == 7 and sorted(u.values()) == [2, 2, 3]
    assert stats["repeated_class"] == "unanswerable" and stats["repeat_factor"] == round(7 / 3, 3)


def test_grpo_gate():
    from abstention.grpo import gate_passes
    assert gate_passes(0.01, 0.05, 8, 5, max_p=0.1, min_gain=0.02)        # significant
    assert not gate_passes(0.01, 0.3, 8, 5, max_p=0.1, min_gain=0.02)     # small and noisy
    assert gate_passes(6 / 300, 0.15, 20, 14, max_p=0.1, min_gain=0.02)   # +2 points, gained > lost
    assert not gate_passes(0.02, 0.15, 10, 10, max_p=0.1, min_gain=0.02)  # +2 points but not more gained
    assert not gate_passes(-0.01, 0.01, 2, 5, max_p=0.1, min_gain=0.02)   # significantly worse


def test_grpo_reward_counts_zero_variance_groups_per_class():
    from abstention.grpo import Reward
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
    from abstention.grpo import make_pool
    _, stats, text, _ = make_pool(cfg, TRACKED_RESULTS)
    assert text == (TRACKED_RESULTS / "grpo-pool" / "pool.jsonl").read_text()
    assert stats["pool_rows_by_class"]["answerable"] == stats["pool_rows_by_class"]["unanswerable"]


def test_grpo_run_learning_rate_override(cfg):
    from abstention.grpo import run_learning_rate
    assert run_learning_rate(cfg, "G-default") == cfg["grpo"]["learning_rate"] == 1e-6  # the run already done
    assert run_learning_rate(cfg, "G-equal") == 1e-5
    assert cfg["grpo"]["runs"]["G-equal"]["h"] == 1.0 and cfg["grpo"]["runs"]["G-equal"]["r"] == 1.0
