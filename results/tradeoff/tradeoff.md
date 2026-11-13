# Hallucination vs over-refusal on dev: SFT's threshold curve and actual operating points

Curve: SFT abstains when P("I don't know") > cut-off (0.05 to 0.95), from `grpo-G-default-train` step 0. 150 answerable + 150 unanswerable dev questions.

Gap = model over-refusal minus the lowest over-refusal SFT reaches by thresholding at the same hallucination rate (negative = below the curve = better judgment). 95% CI: paired stratified bootstrap.

| Model | Dev run | Hallucination | Over-refusal | Gap (pts) | 95% CI | Verdict |
|---|---|---|---|---|---|---|
| Baseline | `baseline-dev` | 54.7% | 8.0% | +6.7 | [+2.7, +12.0] | above SFT's curve |
| SFT | `sft-dev-e2` | 12.7% | 12.0% | +1.3 | [-1.3, +4.7] | on SFT's curve |
| DPO v1 | `dpo-dev-lr5e-06-e1` | 8.7% | 19.3% | +0.7 | [-2.0, +8.0] | on SFT's curve |
| DPO v2 | `dpo-v2-dev-lr2e-05-e1` | 12.0% | 10.7% | -0.7 | [-4.0, +2.7] | on SFT's curve |
| G-default best | `grpo-G-default-dev-s200` | 8.0% | 17.3% | -4.0 | [-7.3, +4.0] | on SFT's curve |
| G-equal best | `grpo-G-equal-dev-s200` | 11.3% | 12.7% | +0.7 | [-6.7, +5.3] | on SFT's curve |
| Twin input | `dpo-twin-dev-lr5e-06-e2` | 11.3% | 12.0% | +0.0 | [-7.3, +4.0] | on SFT's curve |
| Twin output | `dpo-twin-out-dev-lr5e-06-e2` | 11.3% | 10.0% | -2.0 | [-10.0, +2.0] | on SFT's curve |
| 8B ceiling | `ceiling-dev` | 36.7% | 2.0% | -0.7 | [-4.7, +2.7] | on SFT's curve |

| Cut-off | Hallucination | Over-refusal |
|---|---|---|
| 0.05 | 6.0% | 35.3% |
| 0.10 | 8.0% | 26.7% |
| 0.15 | 8.0% | 22.7% |
| 0.20 | 8.7% | 20.7% |
| 0.25 | 9.3% | 16.0% |
| 0.30 | 10.0% | 14.7% |
| 0.35 | 11.3% | 12.0% |
| 0.40 | 12.0% | 11.3% |
| 0.45 | 14.0% | 10.7% |
| 0.50 | 14.7% | 9.3% |
| 0.55 | 14.7% | 8.7% |
| 0.60 | 14.7% | 8.7% |
| 0.65 | 14.7% | 8.7% |
| 0.70 | 16.7% | 8.0% |
| 0.75 | 18.0% | 8.0% |
| 0.80 | 20.0% | 7.3% |
| 0.85 | 21.3% | 7.3% |
| 0.90 | 22.7% | 7.3% |
| 0.95 | 27.3% | 5.3% |
