# RUN.md

Everything runs on a RunPod GPU pod. The laptop only edits code and `config.json` and runs the CPU tests.

## Layout

| Where | What |
| --- | --- |
| `config.json` | Every setting you might change: models, prompt, data sizes, eval decoding, vLLM, LoRA, SFT/DPO/GRPO hyperparameters, GRPO sweep |
| `config.py` | Loads the config; `--config path` on every script |
| `prompts.py` | The single prompt template used by every stage |
| `grader.py` | Frozen grader + GRPO reward (`GRADER_VERSION`); not configurable on purpose |
| `metrics.py` | 2×2 rates, bootstrap intervals, strict vs lenient, reward; paired bootstrap + McNemar for model-vs-model |
| `compare.py` | Paired comparison of two runs on the same questions (CPU only) |
| `data.py` | Builds the frozen splits + `manifest.json` (sha256 per split) |
| `eval.py` | vLLM evaluation of a base model or base + LoRA adapter |
| `freeze_grader.py`, `fixtures/` | Golden fixture that freezes the grader |
| `test_grader.py`, `test_pipeline.py` | CPU tests, run locally and on the pod |
| `scripts/pod_setup.sh` | Pod setup: clone, venv, GPU check, tests |
| `sft.py` | Stage 2: LoRA SFT with TRL `SFTTrainer`, per-epoch adapters, best-epoch selection on dev |
| `sample_dpo.py` | Samples SFT-best on a split and grades every sample (DPO pairs; GRPO question filter) |
| `dpo.py` | Stage 3: DPO variants from the SFT samples |
| `twins.py`, `dpo_twin.py` | Twin DPO: SQuAD question twins (real question vs its edited, unanswerable twin) and DPO on them, input-side or output-side pairs |
| `grpo.py` | Stage 4: GRPO question pool (disagreeing samples only), training with per-eval logging, dev gate |
| `helpers/` | Laptop-side scripts that drive the pod over RunPod's proxy SSH (`pod.sh`, `pod_job.sh`, `pod_log.sh`, `pod_wait.sh`, `pull_results.sh`) and the on-pod run scripts they ship (`run_baselines.sh`, `run_ceiling.sh`, `run_sft.sh`) |
| `FUTURE.md` | Parked experiments, designed but not run: GRPO with batch/no reward scaling and an exact-match reward; twin DPO on hand-checked hard twins |
| `results/` | Copies of every pod run (`metrics.json`, `handcheck.jsonl`, `outputs.jsonl`), pulled with `helpers/pull_results.sh` |
| `rehearsal/` | Qwen3-0.6B rehearsal results (local MPS, transformers backend) and split hashes |

On the pod, `/workspace` is the 50 GB network volume:

```
/workspace/code         repo checkout
/workspace/venv         Python env (uv); uv's download cache stays on the container disk (/root/.cache/uv)
/workspace/hf           Hugging Face model cache
/workspace/abstention/  paths.root: data/, results/<run>/, checkpoints/
```

## What's frozen and what's editable

- **Editable:** everything in `config.json`.
- **Must not change mid-experiment:** `prompt.*`, `eval.temperature`, `eval.max_new_tokens`. Each `metrics.json` records an `eval_fingerprint` hash of these; rows with different fingerprints aren't comparable.
- **Frozen in code:** grader logic, the format penalty (−0.5 over 30 words), the 20-word containment cap. Changing them needs a `GRADER_VERSION` bump and `uv run python freeze_grader.py`.
- `grpo.runs` may only use `(h, r)` pairs covered by the grader fixture; `test_pipeline.py` checks this.

## 0. Pod setup (once per pod)

1. RunPod console → Storage → create a **50 GB network volume** in a datacenter that has RTX 4090s.
2. Deploy a pod: **RTX 4090**, **RunPod PyTorch** template, attach the volume at `/workspace`.
3. Open the web terminal or SSH in, then:

```bash
curl -fsSL https://raw.githubusercontent.com/pranavpage/small-model-abstention/main/scripts/pod_setup.sh | bash
# a branch other than main:  ... | BRANCH=my-branch bash
source ~/.bashrc
```

The script prints the GPU name and vLLM version and runs the tests. Re-run it after a pod restart, or to pull new code.

## 1. Data (once)

```bash
uv run python data.py
cat /workspace/abstention/data/manifest.json
```

The sha256 values must equal `rehearsal/data_manifest.json`; the splits are the same as the rehearsal.

## 2. Baseline and ceiling

```bash
uv run python eval.py --model main --split dev  --run baseline-dev     # tune/inspect here
uv run python eval.py --model main --split test --run baseline         # once
uv run python eval.py --model ceiling --split test --run ceiling       # once
```

Quick smoke test first if anything changed: `uv run python eval.py --model debug --split dev --limit 20 --run smoke`.

`eval.py` refuses a second test-set run into the same folder (`--force` to override) and refuses `--limit` on test.

Done when: the 2×2 for both models exists and 50 rows of `handcheck.jsonl` were read by hand with at most 2 disagreements.

## 3. SFT

`sft.py` trains a LoRA adapter with TRL's `SFTTrainer` on a prompt/completion dataset. The prompt is the exact eval string from `prompts.render()`, and the loss covers only the target plus `<|im_end|>`. Targets are the gold span, or `I don't know` for unanswerable questions. Rows whose answer is longer than `sft.max_answer_words` (10) are dropped, along with the same number of random unanswerable rows, so the set stays 50/50 (3,000 → 2,870). An adapter is saved after each epoch to `checkpoints/sft/epoch-<n>/`.

```bash
uv run python sft.py --dry-run     # prints the tokenized examples and the loss mask, no training
```

From the laptop, the whole stage: train → dev eval per epoch (`sft-dev-e<n>`) → `sft.py --select` copies the best dev balanced score to `checkpoints/sft/best` → test eval once (`sft`):

```bash
git push origin <branch>                              # code reaches the pod only through git
helpers/pod.sh 'BRANCH=<branch> bash /workspace/code/scripts/pod_setup.sh'
helpers/pod_job.sh sft "$(cat helpers/run_sft.sh)"
helpers/pod_wait.sh sft
helpers/pull_results.sh sft-train sft-dev-e1 sft-dev-e2 sft
```

## Comparing models

Every model answers the same frozen questions, so compare two runs question by question rather than checking whether their intervals overlap:

```bash
uv run python compare.py sft dpo                       # on the pod (paths.results_dir)
uv run python compare.py sft-dev-e1 sft-dev-e2 --dir results   # on the laptop, from the tracked copies
```

For each rate, `compare.py` prints the difference (B − A) with a **paired bootstrap** 95% interval. Both models are resampled with the same question indices, stratified by answerable/unanswerable. For per-question yes/no outcomes it also prints how many questions flipped each way and an **exact McNemar** p-value on those flips. It refuses runs with different splits, question sets, eval fingerprints or grader versions, and writes `results/compare/<a>__vs__<b>.json`. Claim a winner only when the paired interval excludes 0 (or p < 0.05). `report.py` should use this for every pairwise claim.

## 4. Later stages

Before DPO, `sample_dpo.py` samples the best SFT adapter on the DPO split (1 greedy + `dpo.samples_per_question` at `dpo.sample_temperature`), grades every output and counts the preference pairs (`results/dpo-samples/`): `helpers/pod_job.sh dpo-samples "$(cat helpers/run_dpo_samples.sh)"`.


DPO and GRPO start from `checkpoints/sft/best`. `report.py`: not written yet.

### Twin DPO (input-side pairs)

DPO v1/v2 and GRPO only moved the abstain threshold: every trained model sits on SFT's P("I don't know") threshold curve and the AUC barely moves. Twin DPO gives the model pairs that differ only in the question.

1. `twins.py` matches each unanswerable SQuAD-train question to the answerable question in the same paragraph it was edited from. A match needs word-sequence similarity ≥ 0.8 that beats the runner-up by ≥ 0.1, one-to-one, with a gold answer of ≤ 10 words. For example, "What did Saarinen use for inspiration?" and "What did Saarinen not use for inspiration?". Training twins have a question in the DPO split and none in sft/grpo/dev, and no dev paragraph: 387 twins. 200 held-out twins come from paragraphs with no question in any split. Copies without paragraphs go in `results/twins/`.
2. Each twin gives four sequences, (q, a), (q, IDK), (q′, a), (q′, IDK), scored against the SFT adapter as reference. `dpo-twin` pairs them **input-side**: the same response after the real question vs after the twin (a: q ≻ q′; IDK: q′ ≻ q). That loss ignores any change that moves a response equally after both questions, so a threshold shift cannot lower it. `dpo-twin-out` is the control: the same sequences paired output-side (standard DPO: q: a ≻ IDK; q′: IDK ≻ a).
3. Probes log dev P(IDK) (means, AUC) and held-out twin separation at step 0, every 10 steps and every epoch end. Each variant's best (lr, epoch) on dev must pass the GRPO dev gate before test runs.

```bash
helpers/pod.sh 'cd /workspace/code && uv run --no-sync python twins.py && uv run --no-sync python dpo_twin.py --variant dpo-twin --dry-run'
helpers/pod_job.sh dpo-twin "$(cat helpers/run_dpo_twin.sh)"
helpers/pull_results.sh twins dpo-twin-train dpo-twin-out-train idk-probe compare $(ls -d results/dpo-twin*-dev-* 2>/dev/null)
```

### GRPO

A GRPO group whose 8 rewards are all equal has zero advantage, and the SFT model gets most questions right every time. So `grpo.py` trains only on questions where the SFT model's own samples disagree:

1. `helpers/run_grpo_samples.sh` samples SFT-best on the GRPO split (greedy + 8 at T=1.0, plus an independent "recheck" draw of 8 that is used only to measure the filter) → `results/grpo-samples/`. Then it builds the pool → `results/grpo-pool/` (`pool.jsonl`, `pool_stats.json`).
2. Pool = GRPO-split questions whose 8 rewards (G-default reward) are not all equal. That includes partial answers (0.67 vs 1.0). Answerable questions also need at least one correct sample: with only "I don't know" (−0.5) vs a wrong answer (0), the only lesson is to guess. The varying unanswerable questions from the DPO split (`dpo-samples`, 4 samples each; SFT-best never trained on them) are added. Then unanswerable questions are repeated only as needed to reach 50/50. Pool: 342 answerable + 269 unanswerable (repeat factor 1.27) = 684 rows. Every `(h, r)` run uses the same pool.
3. Each step is 4 questions × 8 rollouts. `max_steps = ceil(passes × pool / 4)`. KL is measured against a frozen copy of the SFT adapter.
4. Logging: every rollout batch logs TRL's metrics plus the share of zero-variance groups per class (`train_log.jsonl`). At step 0 and every `eval_every_steps` (`evals.jsonl`), it also logs the dev balanced score (HF greedy, as a live monitor), P("I don't know") on answerable vs unanswerable dev questions, mean answer length, and the per-class zero-variance share since the last eval. It also logs the AUC of P("I don't know") for unanswerable vs answerable dev questions; the per-question values are saved in `p_idk.jsonl`. The adapter is saved at each eval. A higher AUC means better judgment. A moved abstain rate with the same AUC is only a threshold shift.
5. `helpers/run_grpo.sh` evaluates every checkpoint on dev with vLLM (one engine), selects the best, and runs test only if it passes the dev gate against `sft-dev-e2`: paired p < 0.1, or ≥ +2 points with more questions gained than lost.

```bash
helpers/pod_job.sh grpo-samples "$(cat helpers/run_grpo_samples.sh)"
helpers/pod.sh 'cd /workspace/code && uv run --no-sync python grpo.py --run G-default --dry-run'   # checks, no training
helpers/pod_job.sh grpo-G-default "RUN=G-default $(cat helpers/run_grpo.sh)"
```

## Driving the pod from the laptop

RunPod's proxy SSH (`<pod>@ssh.runpod.io`) only offers an interactive terminal; `helpers/pod.sh` works around that.

```bash
export POD_HOST=<pod-id>-<suffix>@ssh.runpod.io          # from the pod's Connect tab
helpers/pod.sh 'nvidia-smi'                              # short command
helpers/pod_job.sh ceiling "$(cat helpers/run_ceiling.sh)"   # long job, detached, log in /workspace/abstention/logs/
helpers/pod_wait.sh ceiling                              # block until it stops, print the log tail
```

## Copying results back

`helpers/pull_results.sh [run ...]` copies `metrics.json`, `handcheck.jsonl` and `outputs.jsonl` for each run into `results/<run>/` in the repo. Commit them.

Stop the pod between steps; the volume keeps everything.
