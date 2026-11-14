#!/usr/bin/env bash
# Runs ON THE POD: sample the best SFT adapter on the GRPO split (1 greedy + 8 at T=1.0 per question, plus an
# independent recheck draw of 8 that is never used for filtering), then build the GRPO question pool.
# Analysis only, no training.
#   git push, then: pod/exec.sh 'BRANCH=<branch> bash /workspace/code/pod/setup.sh'
#   pod/job.sh grpo-samples "$(cat pod/jobs/grpo_samples.sh)"   # from the laptop
set -euo pipefail
source /workspace/code/pod/env.sh
$PY abstention.sample --adapter checkpoints/sft/best --split grpo --n 8 --recheck-n 8 \
  --temperature 1.0 --run grpo-samples
$PY abstention.grpo --pool
