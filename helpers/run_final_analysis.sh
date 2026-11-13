#!/usr/bin/env bash
# Runs ON THE POD, no training: wrap up the stopped G-equal run (dev evals, selection, gate, test only if it passes),
# the 8B ceiling on dev, per-question P("I don't know") for every model on dev, and fresh 8-sample rollouts from each
# GRPO run's best checkpoint on the GRPO split (to measure reward-group shapes at the end of training).
#   helpers/pod_job.sh final-analysis "$(cat helpers/run_final_analysis.sh)"
set -euo pipefail
cd /workspace/code
export HF_HOME=/workspace/hf UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
PY="uv run --no-sync python"
RUN=G-equal

$PY grpo.py --run $RUN --finalize-stopped
ADAPTERS=(); RUNS=()
for s in $($PY grpo.py --run $RUN --candidates); do ADAPTERS+=("checkpoints/grpo/$RUN/step-$s"); RUNS+=("grpo-$RUN-dev-s$s"); done
$PY eval.py --model main --split dev --adapter "${ADAPTERS[@]}" --run "${RUNS[@]}"
rc=0; $PY grpo.py --run $RUN --select || rc=$?
if [ $rc -eq 0 ]; then
  $PY eval.py --model main --adapter "checkpoints/grpo/$RUN/best" --split test --run "grpo-$RUN"
  $PY compare.py sft "grpo-$RUN"
elif [ $rc -eq 10 ]; then
  echo "== grpo-$RUN: best checkpoint does not pass the dev gate; test NOT run"
else
  exit $rc
fi

$PY eval.py --model ceiling --split dev --run ceiling-dev
$PY probe_idk.py Baseline=main SFT=main:checkpoints/sft/best "DPO-v1=main:checkpoints/dpo/best" \
  "DPO-v2=main:checkpoints/dpo-v2/best" "G-default=main:checkpoints/grpo/G-default/best" \
  "G-equal=main:checkpoints/grpo/G-equal/best" Ceiling=ceiling
rm -rf /workspace/hf/hub/models--Qwen--Qwen3-8B   # 16 GB; re-download if the ceiling is ever needed again

for r in G-default G-equal; do
  $PY sample_dpo.py --adapter checkpoints/grpo/$r/best --split grpo --n 8 --temperature 1.0 --run grpo-$r-best-samples
done
echo "== final analysis done"
