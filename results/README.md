# results/

Every run from the pod, copied here with `pod/pull_results.sh` and committed. The analysis scripts and several tests read these files, so **don't rename or move run folders**.

## Start here

| File | What |
|---|---|
| [`final/final_table.md`](final/final_table.md) | Every model: dev and test rates, dev P(IDK) AUC vs SFT |
| [`tradeoff/tradeoff.md`](tradeoff/tradeoff.md) | SFT's threshold curve and every model's distance from it (with [`tradeoff.png`](tradeoff/tradeoff.png)) |
| [`final/padding.md`](final/padding.md) | GRPO answer-length growth |
| [`final/reward_groups.md`](final/reward_groups.md) | Why GRPO's penalties had no effect (group reward scaling) |
| [`dpo-samples/analysis.md`](dpo-samples/analysis.md) | Hand-coded SFT error types |
| [`grader-audit/hyphens.md`](grader-audit/hyphens.md) | Known grader limitation and its effect on rankings |
| `compare/<a>__vs__<b>.json` | Paired comparisons written by `abstention.compare` |

`final/`, `tradeoff/` and `grader-audit/` are regenerated on CPU by `abstention/analysis/`.

## Run names

| Pattern | Example | What |
|---|---|---|
| `<model>` | `baseline`, `sft`, `dpo-v2`, `ceiling` | **Test** split eval of a selected model (run once) |
| `<model>-dev` | `baseline-dev`, `ceiling-dev` | Dev eval of a model with no training candidates |
| `sft-dev-e<n>` | `sft-dev-e2` | Dev eval of SFT epoch n (epoch 2 was selected) |
| `<variant>-dev-lr<lr>-e<n>` | `dpo-v2-dev-lr2e-05-e1` | Dev eval of a DPO or twin-DPO candidate (learning rate, epoch) |
| `grpo-<run>-dev-s<step>` | `grpo-G-default-dev-s200` | Dev eval of a GRPO checkpoint |
| `<stage>-train` | `sft-train`, `grpo-G-equal-train` | Training logs and `train_summary.json` (holds the selected candidate) |
| `dpo-samples`, `grpo-samples` | | SFT sampled on the DPO / GRPO split, every sample graded |
| `grpo-<run>-best-samples` | | Fresh samples from a GRPO run's best checkpoint (reward-group shapes at the end) |
| `grpo-pool`, `twins` | | The GRPO question pool; question twins for twin DPO |
| `idk-probe/<model>.json` | | P("I don't know") per dev question |
| `baseline-0.6b*` | | Qwen3-0.6B, the debug model |

Selected candidates: SFT epoch 2, DPO v1 `lr5e-06-e1`, DPO v2 `lr2e-05-e1`, GRPO step 200 for both runs, twin DPO `lr5e-06-e2` for both variants.

## Files in an eval run

| File | What |
|---|---|
| `metrics.json` | 2×2 rates with bootstrap 95% CIs, counts, `grader_version`, and the run `config` (including `eval_fingerprint`) |
| `outputs.jsonl` | One row per question: output, gold, grade |
| `handcheck.jsonl` | 50 random rows for reading by hand |
