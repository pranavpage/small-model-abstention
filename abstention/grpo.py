"""Stage 4: GRPO on top of the best SFT adapter (runs on the RunPod GPU).

Question pool (CPU, `--pool`): a GRPO group whose 8 rewards are all equal has zero advantage and teaches nothing,
and the SFT model gets most questions right every time. So the pool keeps only the questions where the SFT
model's own samples disagree:
  1. results/<grpo.samples_run>/samples.jsonl (sample.py on the grpo split, 8 samples at T=1.0) is scored
     with the reward of grpo.pool.filter_run; a question is kept if its 8 rewards are not all equal. This keeps
     partial answers too (lenient score 0.67 vs 1.0), not only right-vs-wrong.
  2. With grpo.pool.drop_answerable_without_correct, an answerable question also needs at least one correct
     sample. Without one, its only contrast is "I don't know" (-r) vs a wrong answer (~0), which teaches guessing.
  3. The varying questions of grpo.pool.topup_classes (unanswerable) from the DPO split
     (results/<grpo.pool.topup_samples_run>, 4 samples each; SFT-best never trained on them) are added. Their
     prompts come from data/dpo.jsonl.
  4. The smaller class is then repeated only as needed to reach 50/50: whole copies, then a seeded random subset
     for the remainder. Each pool row has a unique `uid` (<split>:<id>#<copy>).
The same pool is used for every run of the (h, r) sweep, so only the reward changes between runs.

Training: TRL GRPOTrainer continues the SFT LoRA adapter; with kl_beta > 0 TRL copies it into a frozen "ref"
adapter, so the KL reference is the SFT model. Each optimizer step = grpo.questions_per_step questions x
grpo.num_generations rollouts; max_steps = ceil(grpo.passes x pool size / questions_per_step).

Logged every rollout batch (train_log.jsonl): TRL's own metrics (reward, kl, frac_reward_zero_std, ...) plus the
share of zero-variance groups per class. Logged at step 0 and every grpo.eval_every_steps (evals.jsonl), with the
adapter saved to checkpoints/grpo/<run>/step-<n>:
  dev balanced score, hallucination, over-refusal, answer accuracy (HF greedy decoding: a live monitor; the
  official numbers come from eval.py on each saved checkpoint afterwards), P("I don't know") on answerable vs
  unanswerable dev questions, mean answer length (words, non-abstain outputs) on dev and in training rollouts,
  and the per-class share of zero-variance groups since the previous eval.

Selection (`--select`, after eval.py has scored every checkpoint on dev as grpo-<run>-dev-s<step>): the best dev
balanced score goes to checkpoints/grpo/<run>/best. It may reach test only if it passes the gate against SFT on
dev (grpo.gate): paired McNemar p < max_p with a positive difference, or a gain >= min_gain with more questions
gained than lost.

Usage (on the pod; --pool also on the laptop from the tracked results/):
  python -m abstention.grpo --pool [--results-dir results]   # build the pool, print stats and examples -> grpo-pool/
  python -m abstention.grpo --run G-default --dry-run        # build trainer, checks, step-0 dev monitor; no training
  python -m abstention.grpo --run G-default                  # train
  python -m abstention.grpo --run G-default --candidates     # print '<step>' per saved checkpoint
  python -m abstention.grpo --run G-default --select         # exit 0 if the best checkpoint passes the dev gate, 10 if not
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from abstention.config import add_config_arg, load_config
from abstention.grader import GRADER_VERSION, grade, reward
from abstention.metrics import auc

SPLIT_FILES = {"grpo": "grpo.jsonl", "dpo": "dpo.jsonl"}
NOT_BETTER_THAN_SFT = 10
CLASSES = ("answerable", "unanswerable")


def cls(answerable: bool) -> str:
    return "answerable" if answerable else "unanswerable"


def run_settings(cfg, name):
    if name not in cfg["grpo"]["runs"]:
        raise SystemExit(f"unknown GRPO run {name!r}; config has {list(cfg['grpo']['runs'])}")
    r = cfg["grpo"]["runs"][name]
    return r["h"], r["r"]


def run_learning_rate(cfg, name):
    """grpo.runs.<name>.learning_rate if set, else grpo.learning_rate."""
    run_settings(cfg, name)
    return cfg["grpo"]["runs"][name].get("learning_rate", cfg["grpo"]["learning_rate"])


def group_rewards(outputs, answerable, gold, h, r):
    return [reward(o, answerable, gold, h, r)["reward"] for o in outputs]


def zero_variance(rewards) -> bool:
    return max(rewards) - min(rewards) < 1e-9


# ---------------------------------------------------------------- pool (CPU)

def _scan(rows, split, h, r, need_correct=False, key="samples"):
    """Kept questions of one sampled split (rewards not all equal; answerable ones optionally also need a correct
    sample), and per-class counts."""
    kept, n, zero, no_correct = [], Counter(), Counter(), Counter()
    for x in rows:
        c = cls(x["answerable"])
        n[c] += 1
        if zero_variance(group_rewards([s["output"] for s in x[key]], x["answerable"], x["gold"], h, r)):
            zero[c] += 1
        elif need_correct and x["answerable"] and not any(s["cell"] == "correct" for s in x[key]):
            no_correct[c] += 1
        else:
            kept.append({"id": x["id"], "split": split, "answerable": x["answerable"]})
    stats = {c: {"questions": n[c], "zero_variance": zero[c], "varying_without_correct_dropped": no_correct[c],
                 "kept": n[c] - zero[c] - no_correct[c],
                 "zero_variance_share": round(zero[c] / n[c], 4) if n[c] else None} for c in CLASSES}
    stats["samples_per_question"] = len(rows[0][key]) if rows else 0
    return kept, stats


def build_pool(grpo_rows, topup_rows, h, r, seed, topup_classes=("unanswerable",), need_correct=True):
    """Returns (pool rows, stats). Deterministic for fixed inputs and seed."""
    kept, s_grpo = _scan(grpo_rows, "grpo", h, r, need_correct)
    extra, s_top = _scan(topup_rows, "dpo", h, r, need_correct)
    extra = [k for k in extra if cls(k["answerable"]) in topup_classes]
    kept += extra
    stats = {"filter": {"h": h, "r": r, "drop_answerable_without_correct": need_correct},
             "grpo_split": s_grpo, "topup_split": s_top, "topup_classes": list(topup_classes),
             "added_from_topup": {c: sum(cls(k["answerable"]) == c for k in extra) for c in CLASSES}}

    by = {c: sorted((k for k in kept if cls(k["answerable"]) == c), key=lambda k: (k["split"], k["id"]))
          for c in CLASSES}
    small, big = sorted(CLASSES, key=lambda c: (len(by[c]), c))
    if not by[small]:
        raise ValueError(f"no varying {small} questions; cannot balance")
    rng = random.Random(seed)
    reps, rem = divmod(len(by[big]), len(by[small]))
    copies = by[small] * reps + rng.sample(by[small], rem)
    seen = Counter()
    pool = []
    for k in by[big] + copies:
        key = (k["split"], k["id"])
        pool.append({**k, "copy": seen[key], "uid": f"{k['split']}:{k['id']}#{seen[key]}"})
        seen[key] += 1
    rng.shuffle(pool)
    stats.update({
        "unique_questions": {c: len(by[c]) for c in CLASSES},
        "repeated_class": small, "repeat_factor": round(len(by[big]) / len(by[small]), 3),
        "pool_rows": len(pool), "pool_rows_by_class": {c: sum(cls(p["answerable"]) == c for p in pool) for c in CLASSES},
        "pool_rows_by_split": dict(Counter(p["split"] for p in pool)),
    })
    return pool, stats


def recheck_stats(grpo_rows, pool, h, r):
    """Zero-variance share on the independent recheck draw (never used for filtering): the whole grpo split
    (= no filter) vs the grpo-split rows of the pool, weighted by pool copies (= what training starts from)."""
    if not grpo_rows or "recheck" not in grpo_rows[0]:
        return None
    zv = {x["id"]: zero_variance(group_rewards([s["output"] for s in x["recheck"]], x["answerable"], x["gold"], h, r))
          for x in grpo_rows}
    ans = {x["id"]: x["answerable"] for x in grpo_rows}
    out = {"samples_per_question": len(grpo_rows[0]["recheck"])}
    for c in CLASSES:
        all_ids = [i for i in zv if cls(ans[i]) == c]
        pool_ids = [p["id"] for p in pool if p["split"] == "grpo" and cls(p["answerable"]) == c]
        out[c] = {"whole_split": round(sum(zv[i] for i in all_ids) / len(all_ids), 4),
                  "pool": round(sum(zv[i] for i in pool_ids) / len(pool_ids), 4) if pool_ids else None}
    return out


def load_samples(results_dir: Path, run: str):
    return [json.loads(l) for l in open(results_dir / run / "samples.jsonl")]


def make_pool(cfg, results_dir: Path):
    g, pc = cfg["grpo"], cfg["grpo"]["pool"]
    h, r = run_settings(cfg, pc["filter_run"])
    grpo_rows = load_samples(results_dir, g["samples_run"])
    topup_rows = load_samples(results_dir, pc["topup_samples_run"])
    pool, stats = build_pool(grpo_rows, topup_rows, h, r, pc["seed"], tuple(pc["topup_classes"]),
                             pc["drop_answerable_without_correct"])
    stats["recheck_zero_variance_share"] = recheck_stats(grpo_rows, pool, h, r)
    stats["filter_run"] = pc["filter_run"]
    stats["grader_version"] = GRADER_VERSION
    text = "".join(json.dumps(p) + "\n" for p in pool)
    stats["pool_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return pool, stats, text, grpo_rows


def print_examples(grpo_rows, pool, h, r, seed):
    """One kept and one dropped question per class, with every sample and its reward."""
    in_pool = Counter(p["id"] for p in pool if p["split"] == "grpo")
    rng = random.Random(seed)
    for c in CLASSES:
        rows = [x for x in grpo_rows if cls(x["answerable"]) == c]
        for label, pick in (("KEPT", [x for x in rows if in_pool[x["id"]]]),
                            ("DROPPED", [x for x in rows if not in_pool[x["id"]]])):
            if not pick:
                continue
            x = rng.choice(pick)
            rw = group_rewards([s["output"] for s in x["samples"]], x["answerable"], x["gold"], h, r)
            print(f"\n[{label}] {c}  id={x['id']}  pool copies={in_pool[x['id']]}")
            print(f"  Q: {x['question']}\n  gold: {x['gold'] or '(unanswerable)'}")
            print(f"  greedy: {x['greedy']['output']!r} ({x['greedy']['cell']})")
            for s, w in zip(x["samples"], rw):
                print(f"    {w:+.2f}  {s['cell']:<16} {s['output']!r}")


# ---------------------------------------------------------------- dev gate (CPU)

def gate_passes(diff, p, gained, lost, max_p, min_gain) -> bool:
    """Beat SFT on dev: positive difference with paired p < max_p, or a gain >= min_gain with gained > lost."""
    return (diff > 0 and p < max_p) or (diff >= min_gain - 1e-12 and gained > lost)


def dev_run_name(name, step):
    return f"grpo-{name}-dev-s{step}"


def finalize_stopped(cfg, name):
    """train_summary.json for a run stopped before max_steps: its checkpoints are the step-<n> folders saved so far."""
    out = cfg.checkpoints_dir / "grpo" / name
    res = cfg.results_dir / f"grpo-{name}-train"
    steps = sorted(int(d.name.split("-")[1]) for d in out.glob("step-*"))
    if not steps:
        raise SystemExit(f"no checkpoints under {out}")
    shutil.rmtree(out / "trainer", ignore_errors=True)
    _, pool_stats, _, _ = make_pool(cfg, cfg.results_dir)
    h, r = run_settings(cfg, name)
    g = cfg["grpo"]
    summary = {"run": name, "learning_rate": run_learning_rate(cfg, name), "h": h, "r": r,
               "model": cfg.model_id("main"), "start_adapter": str(cfg.checkpoints_dir / "sft" / "best"),
               "trainer": "trl.GRPOTrainer", "grpo": {k: v for k, v in g.items() if k != "runs"},
               "stopped_early": True, "stopped_after_step": steps[-1], "checkpoints": steps, "pool": pool_stats,
               "eval_fingerprint": cfg.eval_fingerprint(), "grader_version": GRADER_VERSION}
    json.dump(summary, open(res / "train_summary.json", "w"), indent=2)
    print(f"{name}: stopped run finalized with checkpoints {steps}")


def select(cfg, name) -> bool:
    from abstention.metrics import paired_auc_compare, paired_compare
    g = cfg["grpo"]
    res = cfg.results_dir / f"grpo-{name}-train"
    summary = json.load(open(res / "train_summary.json"))
    scores = {}
    for step in summary["checkpoints"]:
        m = json.load(open(cfg.results_dir / dev_run_name(name, step) / "metrics.json"))
        scores[step] = m["balanced_score"]["value"]
        print(f"step {step:>4}: dev balanced {scores[step]:.3f}  halluc {m['hallucination_rate']['value']:.3f}  "
              f"over-refusal {m['over_refusal_rate']['value']:.3f}  accuracy {m['answer_accuracy']['value']:.3f}")
    best = max(scores, key=lambda s: (scores[s], -s))  # ties go to the earlier step
    out = cfg.checkpoints_dir / "grpo" / name
    shutil.rmtree(out / "best", ignore_errors=True)
    shutil.copytree(out / f"step-{best}", out / "best")

    gate = g["gate"]
    load = lambda run: [json.loads(l) for l in open(cfg.results_dir / run / "outputs.jsonl")]
    c = paired_compare(load(gate["reference_dev_run"]), load(dev_run_name(name, best)),
                       n_boot=cfg["eval"]["bootstrap_samples"], seed=cfg["eval"]["seed"])["balanced_score"]
    ok = gate_passes(c["diff"], c["mcnemar_p"], c["only_b"], c["only_a"], gate["max_p"], gate["min_gain"])
    probes = {x["step"]: x["p_idk"] for x in map(json.loads, open(res / "p_idk.jsonl"))}
    dev_ans = {x["id"]: x["answerable"] for x in map(json.loads, open(cfg.data_dir / "dev.jsonl"))}
    split = lambda p, a: [p[i] for i in sorted(p) if dev_ans[i] == a]
    ac = paired_auc_compare(split(probes[0], True), split(probes[0], False), split(probes[best], True),
                            split(probes[best], False), n_boot=cfg["eval"]["bootstrap_samples"], seed=cfg["eval"]["seed"])
    mean_p = lambda p, a: sum(split(p, a)) / len(split(p, a))
    summary["p_idk_auc_dev"] = {"sft_step0": ac["a"], f"step{best}": ac["b"], "diff": ac["diff"], "ci95": ac["ci95"],
                                "mean_p_idk": {"sft_step0": {"answerable": mean_p(probes[0], True),
                                                             "unanswerable": mean_p(probes[0], False)},
                                               f"step{best}": {"answerable": mean_p(probes[best], True),
                                                               "unanswerable": mean_p(probes[best], False)}}}
    print(f"P(IDK) AUC on dev (unanswerable vs answerable): SFT {ac['a']:.3f}  step {best} {ac['b']:.3f}  "
          f"diff {ac['diff']:+.3f}  paired CI [{ac['ci95'][0]:+.3f}, {ac['ci95'][1]:+.3f}]")
    summary.update({"dev_balanced": scores, "best_step": best,
                    "gate": {**gate, "diff": c["diff"], "ci95": c["ci95"], "mcnemar_p": c["mcnemar_p"],
                             "gained": c["only_b"], "lost": c["only_a"], "passes": ok}})
    json.dump(summary, open(res / "train_summary.json", "w"), indent=2)
    print(f"best step {best} -> {out / 'best'}: dev balanced {scores[best]:.3f} vs {gate['reference_dev_run']} "
          f"{c['a']:.3f}  diff {100 * c['diff']:+.1f} pts  paired CI [{100 * c['ci95'][0]:+.1f}, "
          f"{100 * c['ci95'][1]:+.1f}]  gained {c['only_b']} lost {c['only_a']}  McNemar p {c['mcnemar_p']:.3g}"
          f"  -> {'PASSES the gate' if ok else 'does NOT pass the gate'}")
    return ok


# ---------------------------------------------------------------- training (GPU)

def words(text):
    return len(text.split())


class Reward:
    """GRPO reward (grader.reward) that also counts zero-variance groups per class and rollout lengths."""

    def __init__(self, h, r, num_generations):
        self.h, self.r, self.k = h, r, num_generations
        self.__name__ = "abstention_reward"
        self.reset()

    def reset(self):
        self.window = {c: Counter() for c in CLASSES}

    def __call__(self, prompts, completions, uid, answerable, gold, log_metric=None, **kw):
        outs = [c.strip() for c in completions]
        rs = [reward(o, a, gl, self.h, self.r) for o, a, gl in zip(outs, answerable, gold)]
        groups = defaultdict(list)
        for i, u in enumerate(uid):
            groups[u].append(i)
        assert all(len(ix) == self.k for ix in groups.values()), {u: len(ix) for u, ix in groups.items()}
        batch = {c: Counter() for c in CLASSES}
        for ix in groups.values():
            c = cls(answerable[ix[0]])
            batch[c]["groups"] += 1
            batch[c]["zero_var"] += zero_variance([rs[i]["reward"] for i in ix])
        for i, (o, a) in enumerate(zip(outs, answerable)):
            c = batch[cls(a)]
            c["completions"] += 1
            c["abstain"] += rs[i]["cell"] in ("over_refusal", "correct_abstain")
            if rs[i]["cell"] not in ("over_refusal", "correct_abstain"):
                c["answers"] += 1
                c["answer_words"] += words(o)
        for c in CLASSES:
            self.window[c].update(batch[c])
            if log_metric and batch[c]["groups"]:
                log_metric(f"zero_var_share/{c}", batch[c]["zero_var"] / batch[c]["groups"])
                log_metric(f"abstain_share/{c}", batch[c]["abstain"] / batch[c]["completions"])
        return [x["reward"] for x in rs]

    def window_stats(self):
        out = {}
        for c in CLASSES:
            w = self.window[c]
            out[c] = {"groups": w["groups"],
                      "zero_var_share": round(w["zero_var"] / w["groups"], 4) if w["groups"] else None,
                      "abstain_share": round(w["abstain"] / w["completions"], 4) if w["completions"] else None,
                      "mean_answer_words": round(w["answer_words"] / w["answers"], 2) if w["answers"] else None}
        return out


class DevMonitor:
    """Greedy (HF generate) dev metrics + P("I don't know") probe under the current policy."""

    def __init__(self, cfg, tok, rows, batch_size=32):
        from abstention.dpo import IdkProbe
        from abstention.prompts import render
        self.tok, self.rows, self.bs = tok, rows, batch_size
        self.max_new = cfg["eval"]["max_new_tokens"]
        self.ids = [tok(render(cfg, tok, r), add_special_tokens=False)["input_ids"] for r in rows]
        self.probe = IdkProbe(cfg, tok, rows)

    def generate(self, model):
        import torch
        was_training = model.training
        model.eval()
        dev = next(model.parameters()).device
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        texts = []
        with torch.no_grad():
            for i in range(0, len(self.ids), self.bs):
                seqs = self.ids[i:i + self.bs]
                L = max(map(len, seqs))
                ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=dev)
                mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
                out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=self.max_new,
                                     do_sample=False, temperature=None, top_p=None, top_k=None, use_cache=True,
                                     pad_token_id=pad, eos_token_id=self.tok.eos_token_id)
                texts += [t.strip() for t in self.tok.batch_decode(out[:, L:], skip_special_tokens=True)]
        if was_training:
            model.train()
        return texts

    def __call__(self, model):
        texts = self.generate(model)
        graded = [{**grade(t, r["answerable"], r["answers"]), "id": r["id"], "output": t,
                   "answerable": r["answerable"]} for t, r in zip(texts, self.rows)]
        a = [x for x in graded if x["answerable"]]
        u = [x for x in graded if not x["answerable"]]
        acc = sum(x["cell"] == "correct" for x in a) / len(a)
        abst = sum(x["cell"] == "correct_abstain" for x in u) / len(u)
        answered = [x for x in graded if not x["abstain"]]
        m = {"balanced": round((acc + abst) / 2, 4), "answer_accuracy": round(acc, 4),
             "hallucination": round(1 - abst, 4),
             "over_refusal": round(sum(x["cell"] == "over_refusal" for x in a) / len(a), 4),
             "mean_answer_words": round(sum(words(x["output"]) for x in answered) / len(answered), 2)
             if answered else None,
             "p_idk": self.probe(model)}
        scored = [x for x in a if x["lenient_score"] is not None]  # answered, answerable
        m["lenient_minus_strict"] = (round(sum(x["lenient_score"] - x["strict_score"] for x in scored) / len(scored), 4)
                                     if scored else None)
        p = self.probe.last_p
        m["p_idk"]["auc"] = round(auc([q for q, r in zip(p, self.rows) if r["answerable"]],
                                      [q for q, r in zip(p, self.rows) if not r["answerable"]]), 4)
        self.last_p = {r["id"]: q for q, r in zip(p, self.rows)}
        return m, graded


def train(cfg, name, dry_run=False):
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from abstention.prompts import render

    g = cfg["grpo"]
    h, r = run_settings(cfg, name)
    lr = run_learning_rate(cfg, name)
    flags = g["flags"]
    model_id = cfg.model_id("main")
    sft_dir = cfg.checkpoints_dir / "sft" / "best"
    out = cfg.checkpoints_dir / "grpo" / name
    res = cfg.results_dir / f"grpo-{name}-train"
    res.mkdir(parents=True, exist_ok=True)

    pool, pool_stats, pool_text, _ = make_pool(cfg, cfg.results_dir)
    (res / "pool.jsonl").write_text(pool_text)
    print(json.dumps(pool_stats, indent=2))

    q_per_step, k = g["questions_per_step"], g["num_generations"]
    completions_per_step = q_per_step * k
    assert completions_per_step % g["per_device_batch_size"] == 0
    grad_accum = completions_per_step // g["per_device_batch_size"]
    max_steps = math.ceil(g["passes"] * len(pool) / q_per_step)
    every = g["eval_every_steps"]
    checkpoints = sorted(set(range(every, max_steps + 1, every)) | {max_steps})
    print(f"pool {len(pool)} rows -> {max_steps} steps of {q_per_step} questions x {k} rollouts "
          f"({g['passes']} passes); checkpoints at {checkpoints}")

    tok = AutoTokenizer.from_pretrained(model_id)
    assert tok.eos_token == "<|im_end|>", tok.eos_token
    src = {s: {json.loads(l)["id"]: json.loads(l) for l in open(cfg.data_dir / f)} for s, f in SPLIT_FILES.items()}
    ds = Dataset.from_list([{"prompt": render(cfg, tok, src[p["split"]][p["id"]]), "uid": p["uid"],
                             "answerable": p["answerable"], "gold": src[p["split"]][p["id"]]["answers"]}
                            for p in pool])
    dev_rows = [json.loads(l) for l in open(cfg.data_dir / "dev.jsonl")]
    monitor = DevMonitor(cfg, tok, dev_rows)
    reward_fn = Reward(h, r, k)

    torch.manual_seed(g["seed"])
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, sft_dir, is_trainable=True)
    evals_path, log_path = res / "evals.jsonl", res / "train_log.jsonl"
    if not dry_run:
        for path in (evals_path, log_path, res / "p_idk.jsonl"):
            path.write_text("")

    kl_window, history = [], []

    def evaluate(step, epoch, model):
        t0 = time.time()
        m, _ = monitor(model)
        kl = {"mean": round(sum(kl_window) / len(kl_window), 5), "max": round(max(kl_window), 5)} if kl_window else None
        rec = {"step": step, "epoch": round(epoch or 0, 3), "dev": m, "rollouts_since_last_eval": reward_fn.window_stats(),
               "kl_since_last_eval": kl, "seconds": round(time.time() - t0)}
        reward_fn.reset()
        kl_window.clear()
        history.append(rec)
        warn = []
        if kl and kl["mean"] > flags["kl_mean_above"]:
            warn.append(f"KL mean {kl['mean']:.3f} > {flags['kl_mean_above']}")
        n = flags["length_rising_evals"]
        lens = [x["dev"]["mean_answer_words"] for x in history[-(n + 1):]]
        if len(lens) == n + 1 and None not in lens and all(b > a for a, b in zip(lens, lens[1:])):
            warn.append(f"dev answer length rose {n} evals in a row: {lens}")
        rec["flags"] = warn
        with open(evals_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(res / "p_idk.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "p_idk": monitor.last_p}) + "\n")
        z = rec["rollouts_since_last_eval"]
        print(f"[eval step {step}] dev balanced {m['balanced']:.3f} (halluc {m['hallucination']:.3f}, over-refusal "
              f"{m['over_refusal']:.3f}, acc {m['answer_accuracy']:.3f}) | P(IDK) answerable "
              f"{m['p_idk']['answerable']['mean_p']:.3f} unanswerable {m['p_idk']['unanswerable']['mean_p']:.3f} "
              f"AUC {m['p_idk']['auc']:.3f} | "
              f"dev answer words {m['mean_answer_words']} | zero-var groups answerable "
              f"{z['answerable']['zero_var_share']} unanswerable {z['unanswerable']['zero_var_share']} "
              f"| rollout answer words {z['answerable']['mean_answer_words']} | lenient-strict "
              f"{m['lenient_minus_strict']} | KL {kl['mean'] if kl else None} ({rec['seconds']}s)", flush=True)
        for w in warn:
            print(f"[FLAG step {step}] {w}", flush=True)

    class Callbacks(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kw):
            evaluate(0, 0, model)

        def on_step_end(self, args, state, control, model=None, **kw):
            if state.global_step in checkpoints:
                dst = out / f"step-{state.global_step}"
                model.save_pretrained(dst, selected_adapters=["default"])
                tok.save_pretrained(dst)
                evaluate(state.global_step, state.epoch, model)

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "kl" in logs:
                kl_window.append(logs["kl"])
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **(logs or {})}) + "\n")

    args = GRPOConfig(
        output_dir=str(out / "trainer"),
        learning_rate=lr,
        lr_scheduler_type=g["lr_scheduler"],
        max_steps=max_steps,
        per_device_train_batch_size=g["per_device_batch_size"],
        gradient_accumulation_steps=grad_accum,
        num_generations=k,
        max_completion_length=g["max_completion_length"],
        temperature=g["temperature"], top_p=1.0, top_k=0,
        beta=g["kl_beta"],
        loss_type=g["loss_type"],
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=g["seed"],
    )
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], args=args, train_dataset=ds,
                          processing_class=tok, callbacks=[Callbacks()])
    pm = trainer.model
    assert "ref" in pm.peft_config, "TRL did not create the frozen SFT reference adapter"
    trainable = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert trainable and not any(".ref." in n for n in trainable), "reference adapter is trainable"
    pm.print_trainable_parameters()
    ta = trainer.args
    print(f"generation_batch_size {ta.generation_batch_size} = {ta.generation_batch_size // k} questions x {k}; "
          f"steps_per_generation {ta.steps_per_generation}; grad_accum {ta.gradient_accumulation_steps}; "
          f"gradient_checkpointing {ta.gradient_checkpointing}")
    assert ta.generation_batch_size == completions_per_step, "one generation batch must be one optimizer step"
    plen = [len(tok(x, add_special_tokens=False)["input_ids"]) for x in ds["prompt"]]
    print(f"prompt tokens: max {max(plen)}, mean {sum(plen) / len(plen):.0f}")

    if dry_run:
        pm.to(trainer.accelerator.device)
        t0 = time.time()
        m, graded = monitor(pm)
        print(f"step-0 dev monitor (SFT, HF greedy, {time.time() - t0:.0f}s): {json.dumps(m)}")
        ref = cfg.results_dir / g["gate"]["reference_dev_run"] / "outputs.jsonl"
        if ref.exists():
            vl = {x["id"]: x for x in map(json.loads, open(ref))}
            same = sum(x["output"] == vl[x["id"]]["output"] for x in graded)
            same_cell = sum(x["cell"] == vl[x["id"]]["cell"] for x in graded)
            print(f"HF greedy vs vLLM greedy ({g['gate']['reference_dev_run']}): identical output {same}/{len(graded)}, "
                  f"same cell {same_cell}/{len(graded)}")
        # Reward path on real rollouts: 8 samples for one answerable and one unanswerable pool question.
        dev = trainer.accelerator.device
        for want in (True, False):
            i = next(j for j in range(len(ds)) if ds[j]["answerable"] == want)
            ids = tok(ds[i]["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                o = pm.generate(input_ids=ids, do_sample=True, temperature=g["temperature"], top_p=1.0, top_k=0,
                                max_new_tokens=g["max_completion_length"], num_return_sequences=k,
                                pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
            comps = tok.batch_decode(o[:, ids.shape[1]:], skip_special_tokens=True)
            rw = reward_fn(prompts=[ds[i]["prompt"]] * k, completions=comps, uid=[ds[i]["uid"]] * k,
                           answerable=[want] * k, gold=[ds[i]["gold"]] * k)
            print(f"rollout check {ds[i]['uid']} (answerable={want}, gold={ds[i]['gold']}):")
            for c, w in zip(comps, rw):
                print(f"   {w:+.2f}  {c.strip()!r}")
        print("dry run: stopping before training")
        return

    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    shutil.rmtree(out / "trainer", ignore_errors=True)
    summary = {"run": name, "learning_rate": lr, "h": h, "r": r, "model": model_id, "start_adapter": str(sft_dir),
               "trainer": "trl.GRPOTrainer", "lora": cfg["lora"], "grpo": {k_: v for k_, v in g.items() if k_ != "runs"},
               "max_steps": max_steps, "checkpoints": checkpoints, "pool": pool_stats, "train_seconds": round(secs),
               "eval_fingerprint": cfg.eval_fingerprint(), "grader_version": GRADER_VERSION,
               "gpu": torch.cuda.get_device_name(0)}
    json.dump(summary, open(res / "train_summary.json", "w"), indent=2)
    print(f"wrote {res}/train_summary.json, evals.jsonl, train_log.jsonl, pool.jsonl")


def main():
    p = add_config_arg(argparse.ArgumentParser())
    p.add_argument("--run", help="a key of grpo.runs in config.json, e.g. G-default")
    p.add_argument("--pool", action="store_true", help="build the question pool, print stats + examples (CPU)")
    p.add_argument("--results-dir", help="where the samples runs live (default paths.results_dir)")
    p.add_argument("--dry-run", action="store_true", help="build trainer, run checks and the step-0 monitor")
    p.add_argument("--candidates", action="store_true", help="print the step of every saved checkpoint")
    p.add_argument("--select", action="store_true", help="pick the best checkpoint on dev and apply the gate")
    p.add_argument("--finalize-stopped", action="store_true", help="write train_summary.json for a stopped run")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.pool:
        rd = Path(args.results_dir) if args.results_dir else cfg.results_dir
        pool, stats, text, grpo_rows = make_pool(cfg, rd)
        (rd / "grpo-pool").mkdir(parents=True, exist_ok=True)
        (rd / "grpo-pool" / "pool.jsonl").write_text(text)
        json.dump(stats, open(rd / "grpo-pool" / "pool_stats.json", "w"), indent=2)
        print(json.dumps(stats, indent=2))
        h, r = run_settings(cfg, cfg["grpo"]["pool"]["filter_run"])
        print_examples(grpo_rows, pool, h, r, cfg["grpo"]["pool"]["seed"])
        print(f"\nwrote {rd / 'grpo-pool'}/pool.jsonl, pool_stats.json")
        return
    if not args.run:
        p.error("--run is required")
    if args.candidates:
        s = json.load(open(cfg.results_dir / f"grpo-{args.run}-train" / "train_summary.json"))
        print("\n".join(str(x) for x in s["checkpoints"]))
    elif args.finalize_stopped:
        finalize_stopped(cfg, args.run)
    elif args.select:
        raise SystemExit(0 if select(cfg, args.run) else NOT_BETTER_THAN_SFT)
    else:
        train(cfg, args.run, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
