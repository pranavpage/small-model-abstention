#!/usr/bin/env bash
# Runs ON THE POD: Qwen3-8B ceiling on the test split (once). Needs ~16 GB free under HF_HOME for the weights.
#   helpers/pod_job.sh ceiling "$(cat helpers/run_ceiling.sh)"   # from the laptop; ships this file over SSH
set -euo pipefail
cd /workspace/code
export HF_HOME=/workspace/hf UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
df -h /workspace
uv run --no-sync python eval.py --model ceiling --split test --run ceiling
