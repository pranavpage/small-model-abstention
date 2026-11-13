# RUN.md

Everything runs on a RunPod GPU pod. The laptop only edits code and `config.json` and runs the CPU tests.

## Layout

| Where | What |
| --- | --- |
| `config.json` | Every setting you might change: models, prompt, data sizes, eval decoding, vLLM, LoRA, SFT/DPO/GRPO hyperparameters, GRPO sweep |
| `config.py` | Loads the config; `--config path` on every script |
| `prompts.py` | The single prompt template used by every stage |
| `grader.py` | Frozen grader + GRPO reward (`GRADER_VERSION`); not configurable on purpose |
| `metrics.py` | 2×2 rates, bootstrap intervals, strict vs lenient, reward |
| `data.py` | Builds the frozen splits + `manifest.json` (sha256 per split) |
| `eval.py` | vLLM evaluation of a base model or base + LoRA adapter |
| `freeze_grader.py`, `fixtures/` | Golden fixture that freezes the grader |
| `test_grader.py`, `test_pipeline.py` | CPU tests, run locally and on the pod |
| `scripts/pod_setup.sh` | Pod setup: clone, venv, GPU check, tests |
| `sft.py` | Stage 2: LoRA SFT with TRL `SFTTrainer`, per-epoch adapters, best-epoch selection on dev |
| `helpers/` | Laptop-side scripts that drive the pod over RunPod's proxy SSH (`pod.sh`, `pod_job.sh`, `pod_log.sh`, `pod_wait.sh`, `pull_results.sh`, `sync_code.sh`) and the on-pod run scripts they ship (`run_baselines.sh`, `run_ceiling.sh`, `run_sft.sh`) |
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
helpers/sync_code.sh                                  # ship this checkout's code to /workspace/code
helpers/pod_job.sh sft "$(cat helpers/run_sft.sh)"
helpers/pod_wait.sh sft
helpers/pull_results.sh sft-train sft-dev-e1 sft-dev-e2 sft
```

## 4. Later stages

`dpo.py`, `grpo.py`, `report.py`: not written yet. DPO and GRPO start from `checkpoints/sft/best`.

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
