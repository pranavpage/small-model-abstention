#!/usr/bin/env bash
# Runs ON THE POD: sample the best SFT adapter on the DPO split (1 greedy + dpo.samples_per_question at
# dpo.sample_temperature per question) and count the preference pairs. Analysis only, no training.
#   git push, then: pod/exec.sh 'BRANCH=<branch> bash /workspace/code/pod/setup.sh'
#   pod/job.sh dpo-samples "$(cat pod/jobs/dpo_samples.sh)"   # from the laptop
set -euo pipefail
source /workspace/code/pod/env.sh
$PY abstention.sample --adapter checkpoints/sft/best --split dpo --run dpo-samples
