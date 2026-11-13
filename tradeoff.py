"""SFT's hallucination / over-refusal tradeoff curve on dev, and where each trained model actually operates (CPU only).

Curve: SFT's P("I don't know<|im_end|>" | prompt) on every dev question (the step-0 probe of a GRPO run, i.e. the
SFT adapter; results/<probe_run>/p_idk.jsonl). For a cut-off t the model abstains when P > t:
  hallucination rate = unanswerable questions with P <= t      over-refusal rate = answerable questions with P > t
Only the abstain decision matters for these two rates, so the curve needs no extra generation.

Operating points: each model's greedy dev outputs (vLLM eval runs).

On the curve or below it: for a model at hallucination h and over-refusal o, gap = o - O_SFT(h), where O_SFT(h) is
the lowest over-refusal SFT reaches by thresholding at hallucination <= h (linear between frontier points, which a
randomised cut-off attains). gap < 0 means fewer refusals than SFT at the same hallucination rate, i.e. better
judgment, not just a different threshold. 95% interval: paired, stratified bootstrap (the same resampled questions
for the curve and the model). "Below" only if the whole interval is < 0.

Usage: uv run --with matplotlib python tradeoff.py   -> results/tradeoff/{tradeoff.png,tradeoff.json,tradeoff.md}
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CUTOFFS = np.round(np.arange(0.05, 0.951, 0.05), 2)
MODELS = [  # (label, dev run)
    ("Baseline", "baseline-dev"),
    ("SFT", "sft-dev-e2"),
    ("DPO v1", "dpo-dev-lr5e-06-e1"),
    ("DPO v2", "dpo-v2-dev-lr2e-05-e1"),
    ("G-default best", "grpo-G-default-dev-s200"),
    ("G-equal best", None),  # filled from results/grpo-G-equal-train/train_summary.json when it exists
]


def frontier(p_ans, p_un):
    """(hallucination, over-refusal) at every distinct cut-off, plus the lower-left frontier for interpolation."""
    ts = np.unique(np.concatenate([[-1.0], p_ans, p_un, [2.0]]))
    h = (p_un[None, :] <= ts[:, None]).mean(1)
    o = (p_ans[None, :] > ts[:, None]).mean(1)
    xs, ys = [], []
    for x in np.unique(h):  # lowest over-refusal at each hallucination level
        xs.append(x)
        ys.append(o[h == x].min())
    return np.array(xs), np.minimum.accumulate(np.array(ys))


def curve_at(p_ans, p_un, cutoffs):
    return [((p_un <= t).mean(), (p_ans > t).mean()) for t in cutoffs]


def gap(p_ans, p_un, halluc, over_ref):
    xs, ys = frontier(p_ans, p_un)
    return over_ref - np.interp(halluc, xs, ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--probe-run", default="grpo-G-default-train", help="run whose step-0 p_idk.jsonl is SFT's")
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    res = Path(args.results)

    p0 = next(x for x in map(json.loads, open(res / args.probe_run / "p_idk.jsonl")) if x["step"] == 0)["p_idk"]
    ref = {r["id"]: r for r in map(json.loads, open(res / "sft-dev-e2" / "outputs.jsonl"))}
    a_ids = sorted(i for i in p0 if ref[i]["answerable"])
    u_ids = sorted(i for i in p0 if not ref[i]["answerable"])
    p_ans, p_un = np.array([p0[i] for i in a_ids]), np.array([p0[i] for i in u_ids])

    models = []
    for label, run in MODELS:
        if run is None:
            s = res / "grpo-G-equal-train" / "train_summary.json"
            if not s.exists() or "best_step" not in json.load(open(s)):
                print(f"skipping {label}: not selected yet")
                continue
            run = f"grpo-G-equal-dev-s{json.load(open(s))['best_step']}"
        rows = {r["id"]: r for r in map(json.loads, open(res / run / "outputs.jsonl"))}
        assert rows.keys() == ref.keys(), f"{run} covers different questions"
        models.append({"label": label, "run": run,
                       "hall": np.array([rows[i]["cell"] == "hallucination" for i in u_ids], float),
                       "refuse": np.array([rows[i]["cell"] == "over_refusal" for i in a_ids], float)})

    rng = np.random.default_rng(args.seed)
    boots = [(rng.integers(0, len(a_ids), len(a_ids)), rng.integers(0, len(u_ids), len(u_ids))) for _ in range(args.boot)]
    out = {"probe_run": args.probe_run, "n_answerable": len(a_ids), "n_unanswerable": len(u_ids),
           "curve": [{"cutoff": float(t), "hallucination": float(h), "over_refusal": float(o)}
                     for t, (h, o) in zip(CUTOFFS, curve_at(p_ans, p_un, CUTOFFS))], "models": []}
    for m in models:
        h, o = m["hall"].mean(), m["refuse"].mean()
        g = gap(p_ans, p_un, h, o)
        gs = [gap(p_ans[ia], p_un[iu], m["hall"][iu].mean(), m["refuse"][ia].mean()) for ia, iu in boots]
        lo, hi = np.percentile(gs, [2.5, 97.5])
        verdict = "below SFT's curve" if hi < 0 else ("above SFT's curve" if lo > 0 else "on SFT's curve")
        out["models"].append({"label": m["label"], "run": m["run"], "hallucination": float(h), "over_refusal": float(o),
                              "gap": float(g), "gap_ci95": [float(lo), float(hi)], "verdict": verdict})
        print(f"{m['label']:<15} halluc {100*h:5.1f}%  over-refusal {100*o:5.1f}%  gap {100*g:+5.1f} pts "
              f"[{100*lo:+5.1f}, {100*hi:+5.1f}]  -> {verdict}")

    dst = res / "tradeoff"
    dst.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst / "tradeoff.json", "w"), indent=2)
    lines = ["# Hallucination vs over-refusal on dev: SFT's threshold curve and actual operating points", "",
             f"Curve: SFT abstains when P(\"I don't know\") > cut-off (0.05 to 0.95), from `{args.probe_run}` step 0. "
             f"{len(a_ids)} answerable + {len(u_ids)} unanswerable dev questions.", "",
             "Gap = model over-refusal minus the lowest over-refusal SFT reaches by thresholding at the same "
             "hallucination rate (negative = below the curve = better judgment). 95% CI: paired stratified bootstrap.", "",
             "| Model | Dev run | Hallucination | Over-refusal | Gap (pts) | 95% CI | Verdict |", "|---|---|---|---|---|---|---|"]
    for m in out["models"]:
        lines.append(f"| {m['label']} | `{m['run']}` | {100*m['hallucination']:.1f}% | {100*m['over_refusal']:.1f}% | "
                     f"{100*m['gap']:+.1f} | [{100*m['gap_ci95'][0]:+.1f}, {100*m['gap_ci95'][1]:+.1f}] | {m['verdict']} |")
    lines += ["", "| Cut-off | Hallucination | Over-refusal |", "|---|---|---|"]
    lines += [f"| {c['cutoff']:.2f} | {100*c['hallucination']:.1f}% | {100*c['over_refusal']:.1f}% |" for c in out["curve"]]
    (dst / "tradeoff.md").write_text("\n".join(lines) + "\n")
    plot(out, dst / "tradeoff.png")
    print(f"wrote {dst}/tradeoff.png, tradeoff.json, tradeoff.md")


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    xs = [100 * c["hallucination"] for c in out["curve"]]
    ys = [100 * c["over_refusal"] for c in out["curve"]]
    ax.plot(xs, ys, color=series, lw=2, zorder=2, solid_capstyle="round")
    ax.scatter(xs, ys, s=22, color=series, edgecolor=surface, linewidth=1.5, zorder=3)
    for c, x, y in zip(out["curve"], xs, ys):
        if c["cutoff"] in (0.05, 0.5, 0.95):
            ax.annotate(f"cut-off {c['cutoff']:.2f}", (x, y), xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color=ink2)
    k = list(CUTOFFS).index(0.8)
    ax.annotate("SFT, abstaining when its own\nP(\"I don't know\") > cut-off", (xs[k], ys[k]), xytext=(-20, 55),
                textcoords="offset points", fontsize=9, color=ink2,
                arrowprops={"arrowstyle": "-", "color": series, "lw": 1})
    xmax = 30
    offsets = {"SFT": (9, 6, "left"), "DPO v1": (9, 2, "left"), "DPO v2": (-9, -12, "right"),
               "G-default best": (-9, -4, "right"), "G-equal best": (-9, 8, "right")}
    for m in out["models"]:
        x, y = 100 * m["hallucination"], 100 * m["over_refusal"]
        sign = lambda v: f"{v:+.1f}".replace("-", "\u2212")
        if x > xmax:  # off-chart: pointer at the right edge
            ax.annotate(f"{m['label']} ({sign(100 * m['gap'])}) at {x:.1f}%  \u2192", (xmax, y), xytext=(-4, 22),
                        textcoords="offset points", fontsize=9, color=ink, ha="right", va="center")
            continue
        ax.scatter([x], [y], s=64, color=ink, edgecolor=surface, linewidth=2, zorder=4)
        dx, dy, ha = offsets.get(m["label"], (8, 4, "left"))
        ax.annotate(f"{m['label']} ({sign(100 * m['gap'])})", (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, color=ink, ha=ha, va="center")
    ax.set_xlabel("Hallucination rate on unanswerable dev questions (%)", color=ink2, fontsize=10)
    ax.set_ylabel("Over-refusal rate on answerable dev questions (%)", color=ink2, fontsize=10)
    ax.set_title("Dev: SFT's tradeoff curve vs each model's operating point\n"
                 "(lower-left is better; number = points above (+) or below (−) SFT's curve)",
                 color=ink, fontsize=11, loc="left")
    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=0)
    ax.grid(color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    main()
