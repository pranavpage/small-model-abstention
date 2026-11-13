#!/usr/bin/env bash
# Runs ON THE POD: SFT the main model, eval each epoch's adapter on dev, keep the best epoch, eval it on test once.
#   git push, then: helpers/pod.sh 'BRANCH=<branch> bash /workspace/code/scripts/pod_setup.sh'
#   helpers/pod_job.sh sft "$(cat helpers/run_sft.sh)"   # from the laptop
set -euo pipefail
cd /workspace/code
export HF_HOME=/workspace/hf UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
EPOCHS=$(uv run --no-sync python -c "from config import load_config; print(load_config()['sft']['epochs'])")

uv run --no-sync python sft.py
for e in $(seq 1 "$EPOCHS"); do
  uv run --no-sync python eval.py --model main --adapter checkpoints/sft/epoch-$e --split dev --run sft-dev-e$e
done
uv run --no-sync python sft.py --select
uv run --no-sync python eval.py --model main --adapter checkpoints/sft/best --split test --run sft
