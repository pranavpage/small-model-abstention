#!/usr/bin/env bash
# Runs ON THE POD: Qwen3-8B ceiling on the test split (once). Needs ~16 GB free under HF_HOME for the weights.
#   pod/job.sh ceiling "$(cat pod/jobs/ceiling.sh)"   # from the laptop; ships this file over SSH
set -euo pipefail
source /workspace/code/pod/env.sh
df -h /workspace
$PY abstention.eval --model ceiling --split test --run ceiling
