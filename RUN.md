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
| `rehearsal/` | Qwen3-0.6B rehearsal results (local MPS, transformers backend) and split hashes |

On the pod, `/workspace` is the 50 GB network volume:

```
/workspace/code         repo checkout
/workspace/venv         Python env (uv)
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

## 3. Later stages

`sft.py`, `dpo.py`, `grpo.py`, `report.py`: not written yet. They read their sections of `config.json` and write adapters to `checkpoints/<stage>/`, evaluated with:

```bash
uv run python eval.py --model main --adapter checkpoints/sft/best --split test --run sft
```

## Copying results back

Results live on the network volume. To pull them to the laptop, from the pod: `runpodctl send /workspace/abstention/results` and on the laptop `runpodctl receive <code>`. Or commit the small `metrics.json`/`handcheck.jsonl` files.

Stop the pod between steps; the volume keeps everything.
