#!/usr/bin/env bash
# Runs ON THE POD: sample the best SFT adapter on the GRPO split (1 greedy + 8 at T=1.0 per question, plus an
# independent recheck draw of 8 that is never used for filtering), then build the GRPO question pool.
# Analysis only, no training.
#   git push, then: helpers/pod.sh 'BRANCH=<branch> bash /workspace/code/scripts/pod_setup.sh'
#   helpers/pod_job.sh grpo-samples "$(cat helpers/run_grpo_samples.sh)"   # from the laptop
set -euo pipefail
cd /workspace/code
export HF_HOME=/workspace/hf UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
uv run --no-sync python sample_dpo.py --adapter checkpoints/sft/best --split grpo --n 8 --recheck-n 8 \
  --temperature 1.0 --run grpo-samples
uv run --no-sync python grpo.py --pool
