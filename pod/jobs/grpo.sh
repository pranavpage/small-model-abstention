#!/usr/bin/env bash
# Runs ON THE POD: one GRPO run (a key of grpo.runs) from checkpoints/sft/best. Trains, evals every saved checkpoint
# on dev with vLLM (one engine for all of them), keeps the best, and evals it on test ONLY if it passes the dev gate
# against SFT (grpo.gate).
#   git push, then: pod/exec.sh 'BRANCH=<branch> bash /workspace/code/pod/setup.sh'
#   pod/job.sh grpo-G-default "RUN=G-default $(cat pod/jobs/grpo.sh)"   # from the laptop
set -euo pipefail
: "${RUN:?set RUN, e.g. G-default}"
source /workspace/code/pod/env.sh

$PY abstention.grpo --run "$RUN"
STEPS=$($PY abstention.grpo --run "$RUN" --candidates)
ADAPTERS=(); RUNS=()
for s in $STEPS; do ADAPTERS+=("checkpoints/grpo/$RUN/step-$s"); RUNS+=("grpo-$RUN-dev-s$s"); done
$PY abstention.eval --model main --split dev --adapter "${ADAPTERS[@]}" --run "${RUNS[@]}"
rc=0; $PY abstention.grpo --run "$RUN" --select || rc=$?
if [ $rc -eq 0 ]; then
  $PY abstention.eval --model main --adapter "checkpoints/grpo/$RUN/best" --split test --run "grpo-$RUN"
  $PY abstention.compare sft "grpo-$RUN"
elif [ $rc -eq 10 ]; then
  echo "== grpo-$RUN: best checkpoint does not pass the dev gate; test NOT run"
else
  exit $rc
fi
