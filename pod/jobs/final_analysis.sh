#!/usr/bin/env bash
# Runs ON THE POD, no training: wrap up the stopped G-equal run (dev evals, selection, gate, test only if it passes),
# the 8B ceiling on dev, per-question P("I don't know") for every model on dev, and fresh 8-sample rollouts from each
# GRPO run's best checkpoint on the GRPO split (to measure reward-group shapes at the end of training).
#   pod/job.sh final-analysis "$(cat pod/jobs/final_analysis.sh)"
set -euo pipefail
source /workspace/code/pod/env.sh
RUN=G-equal

$PY abstention.grpo --run $RUN --finalize-stopped
ADAPTERS=(); RUNS=()
for s in $($PY abstention.grpo --run $RUN --candidates); do ADAPTERS+=("checkpoints/grpo/$RUN/step-$s"); RUNS+=("grpo-$RUN-dev-s$s"); done
$PY abstention.eval --model main --split dev --adapter "${ADAPTERS[@]}" --run "${RUNS[@]}"
rc=0; $PY abstention.grpo --run $RUN --select || rc=$?
if [ $rc -eq 0 ]; then
  $PY abstention.eval --model main --adapter "checkpoints/grpo/$RUN/best" --split test --run "grpo-$RUN"
  $PY abstention.compare sft "grpo-$RUN"
elif [ $rc -eq 10 ]; then
  echo "== grpo-$RUN: best checkpoint does not pass the dev gate; test NOT run"
else
  exit $rc
fi

$PY abstention.eval --model ceiling --split dev --run ceiling-dev
$PY abstention.probe_idk Baseline=main SFT=main:checkpoints/sft/best "DPO-v1=main:checkpoints/dpo/best" \
  "DPO-v2=main:checkpoints/dpo-v2/best" "G-default=main:checkpoints/grpo/G-default/best" \
  "G-equal=main:checkpoints/grpo/G-equal/best" Ceiling=ceiling
rm -rf /workspace/hf/hub/models--Qwen--Qwen3-8B   # 16 GB; re-download if the ceiling is ever needed again

for r in G-default G-equal; do
  $PY abstention.sample --adapter checkpoints/grpo/$r/best --split grpo --n 8 --temperature 1.0 --run grpo-$r-best-samples
done
echo "== final analysis done"
