#!/usr/bin/env bash
# Runs ON THE POD: SFT the main model, eval each epoch's adapter on dev, keep the best epoch, eval it on test once.
#   git push, then: pod/exec.sh 'BRANCH=<branch> bash /workspace/code/pod/setup.sh'
#   pod/job.sh sft "$(cat pod/jobs/sft.sh)"   # from the laptop
set -euo pipefail
source /workspace/code/pod/env.sh
EPOCHS=$(uv run --no-sync python -c "from abstention.config import load_config; print(load_config()['sft']['epochs'])")

$PY abstention.sft
for e in $(seq 1 "$EPOCHS"); do
  $PY abstention.eval --model main --adapter checkpoints/sft/epoch-$e --split dev --run sft-dev-e$e
done
$PY abstention.sft --select
$PY abstention.eval --model main --adapter checkpoints/sft/best --split test --run sft
