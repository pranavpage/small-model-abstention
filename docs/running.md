# Running the experiment

GPU work runs on a RunPod pod. The laptop edits code and `config.json`, runs the CPU tests and analysis, and drives the pod over SSH with the scripts in `pod/`. Code reaches the pod only through git: push a branch, then pull it on the pod.

How each stage works (pair construction, the GRPO pool, twin pairing, selection) is documented in the module's docstring. Start at [`abstention/__init__.py`](../abstention/__init__.py) for the map.

## What may change and what may not

- **Editable:** anything in `config.json`.
- **Frozen for the whole experiment:** `prompt.*`, `eval.temperature` and `eval.max_new_tokens`. Each `metrics.json` records an `eval_fingerprint` hash of these, and `compare` refuses runs whose fingerprints differ.
- **Frozen in code:** the grader logic, the format penalty (−0.5 over 30 words) and the 20-word containment cap. To change them, bump `GRADER_VERSION` in `abstention/grader.py`, then run `uv run python -m abstention.freeze_grader`.
- `grpo.runs` may only use `(h, r)` pairs covered by the grader fixture. `tests/test_config.py` checks this.
- **Test is run once per selected model.** `abstention.eval` refuses a second test run into the same folder (`--force` overrides) and refuses `--limit` on test.

## 1. Pod setup (once per pod)

1. RunPod console → Storage: create a **50 GB network volume**.
2. Deploy a pod with a 24 GB GPU (an RTX A5000 was used) on the **RunPod PyTorch** template, with the volume mounted at `/workspace`.
3. On the pod (web terminal):

   ```bash
   curl -fsSL https://raw.githubusercontent.com/pranavpage/small-model-abstention/main/pod/setup.sh | bash
   ```

`pod/setup.sh` clones the repo to `/workspace/code`, installs the venv with the GPU extras, prints the GPU and vLLM version, runs the tests, and makes new shells source `pod/env.sh`. Re-run it after a pod restart or to pull a branch:

```bash
pod/exec.sh 'BRANCH=<branch> bash /workspace/code/pod/setup.sh'
```

Everything that must survive a restart lives on the volume:

```text
/workspace/code         repo checkout
/workspace/venv         Python env (uv)
/workspace/hf           Hugging Face model cache
/workspace/abstention/  config.json paths.root: data/, results/<run>/, checkpoints/, logs/
```

## 2. Driving the pod from the laptop

RunPod's proxy SSH (`<pod>@ssh.runpod.io`) only offers an interactive terminal. `pod/exec.sh` works around that by typing a base64-encoded script into the remote shell.

```bash
export POD_HOST=<pod-id>-<suffix>@ssh.runpod.io     # from the pod's Connect tab
pod/exec.sh 'nvidia-smi'                            # short command, output printed locally
pod/job.sh sft "$(cat pod/jobs/sft.sh)"             # long job: detached, logs to /workspace/abstention/logs/sft.log
pod/log.sh sft                                      # status + log tail
pod/wait.sh sft                                     # block until it stops
pod/pull_results.sh sft-train sft-dev-e1 sft        # copy runs into results/ (no args: every run)
```

Commit what `pull_results.sh` copies. Stop the pod between stages; the volume keeps everything.

## 3. Stages

Each stage is one script in `pod/jobs/`, started with `pod/job.sh <name> "$(cat pod/jobs/<script>)"`. Prefix the command with `VARIANT=…` or `RUN=…` where the table says so.

| # | Job script | Does | Writes |
|---|---|---|---|
| 1 | `baselines.sh` | Builds the data and checks split hashes against `rehearsal/data_manifest.json`, then runs a smoke eval and the 0.6B/1.7B dev and test baselines | `baseline-dev`, `baseline`, `baseline-0.6b*` |
| 1 | `ceiling.sh` | Qwen3-8B on test (needs about 16 GB free in `/workspace/hf`) | `ceiling` |
| 2 | `sft.sh` | Trains LoRA SFT, evals each epoch on dev, selects the best, runs test once | `sft-train`, `sft-dev-e<n>`, `sft` |
| 3 | `dpo_samples.sh` | Samples SFT on the dpo split, grades each sample, counts pairs | `dpo-samples` |
| 3 | `dpo.sh` (`VARIANT=dpo-v2`) | Trains each learning rate, evals each epoch on dev, runs test only if it beats SFT on dev | `<variant>-train`, `<variant>-dev-*`, `<variant>` |
| 3 | `dpo_twin.sh` (`VARIANTS=…` optional) | Builds the twins, trains both variants, evals on dev, runs test only if the dev gate passes | `twins`, `dpo-twin*-train`, `dpo-twin*-dev-*` |
| 4 | `grpo_samples.sh` | Samples SFT on the grpo split (8 + an 8-sample recheck draw), builds the pool | `grpo-samples`, `grpo-pool` |
| 4 | `grpo.sh` (`RUN=G-default`) | Trains, evals every checkpoint on dev, runs test only if the dev gate passes | `grpo-<run>-train`, `grpo-<run>-dev-s*` |
| 5 | `final_analysis.sh` | Finishes the stopped G-equal run, runs the 8B on dev, P(IDK) probes, end-of-training GRPO samples | `ceiling-dev`, `idk-probe`, `grpo-*-best-samples` |

**Dev gate** (GRPO and twin DPO): the best dev candidate goes to test only if, paired against `sft-dev-e2`, p < 0.1, or it gains at least 2 points with more questions gained than lost. `--select` exits 10 when the gate fails, and the job script then skips test.

**Dry runs** before any training stage build the trainer, check that tokens match eval, and train nothing:

```bash
pod/exec.sh 'cd /workspace/code && uv run --no-sync python -m abstention.sft --dry-run'
pod/exec.sh 'cd /workspace/code && uv run --no-sync python -m abstention.grpo --run G-default --dry-run'
```

Afterwards, on the laptop, rebuild the reports from the pulled results:

```bash
uv run python -m abstention.analysis.final_analysis              # results/final/
uv run --with matplotlib python -m abstention.analysis.tradeoff  # results/tradeoff/
uv run python -m abstention.analysis.audit_hyphens               # results/grader-audit/
```

## 4. Comparing models

Every model answers the same frozen questions, so compare two runs question by question rather than checking whether their intervals overlap:

```bash
uv run python -m abstention.compare sft dpo-v2 --dir results     # laptop, committed copies
uv run python -m abstention.compare sft dpo-v2                   # pod, paths.results_dir
```

For each rate, `compare` prints:

- the difference (B − A) with a **paired bootstrap** 95% interval, stratified by answerable/unanswerable;
- how many questions flipped each way;
- an **exact McNemar** p-value on those flips.

It refuses runs with different splits, question sets, eval fingerprints or grader versions, and writes `results/compare/<a>__vs__<b>.json`. Claim a winner only when the paired interval excludes 0 (or p < 0.05).
