#!/usr/bin/env bash
# Runs ON THE POD: sample the best SFT adapter on the DPO split (1 greedy + dpo.samples_per_question at
# dpo.sample_temperature per question) and count the preference pairs. Analysis only, no training.
#   git push, then: helpers/pod.sh 'BRANCH=<branch> bash /workspace/code/scripts/pod_setup.sh'
#   helpers/pod_job.sh dpo-samples "$(cat helpers/run_dpo_samples.sh)"   # from the laptop
set -euo pipefail
cd /workspace/code
export HF_HOME=/workspace/hf UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
uv run --no-sync python sample_dpo.py --adapter checkpoints/sft/best --split dpo --run dpo-samples
