#!/usr/bin/env bash
# Runs ON THE POD: one GRPO run (a key of grpo.runs) from checkpoints/sft/best. Trains, evals every saved checkpoint
# on dev with vLLM (one engine for all of them), keeps the best, and evals it on test ONLY if it passes the dev gate
# against SFT (grpo.gate).
#   git push, then: helpers/pod.sh 'BRANCH=<branch> bash /workspace/code/scripts/pod_setup.sh'
#   helpers/pod_job.sh grpo-G-default "RUN=G-default $(cat helpers/run_grpo.sh)"   # from the laptop
set -euo pipefail
: "${RUN:?set RUN, e.g. G-default}"
cd /workspace/code
export HF_HOME=/workspace/hf UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

uv run --no-sync python grpo.py --run "$RUN"
STEPS=$(uv run --no-sync python grpo.py --run "$RUN" --candidates)
ADAPTERS=(); RUNS=()
for s in $STEPS; do ADAPTERS+=("checkpoints/grpo/$RUN/step-$s"); RUNS+=("grpo-$RUN-dev-s$s"); done
uv run --no-sync python eval.py --model main --split dev --adapter "${ADAPTERS[@]}" --run "${RUNS[@]}"
rc=0; uv run --no-sync python grpo.py --run "$RUN" --select || rc=$?
if [ $rc -eq 0 ]; then
  uv run --no-sync python eval.py --model main --adapter "checkpoints/grpo/$RUN/best" --split test --run "grpo-$RUN"
  uv run --no-sync python compare.py sft "grpo-$RUN"
elif [ $rc -eq 10 ]; then
  echo "== grpo-$RUN: best checkpoint does not pass the dev gate; test NOT run"
else
  exit $rc
fi
