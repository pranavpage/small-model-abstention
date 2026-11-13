#!/usr/bin/env bash
# Runs ON THE POD: DPO from checkpoints/sft/best for every dpo.learning_rates entry, eval each epoch's adapter on dev,
# keep the best (lr, epoch), eval it on test once.
#   git push, then: helpers/pod.sh 'BRANCH=<branch> bash /workspace/code/scripts/pod_setup.sh'
#   helpers/pod_job.sh dpo "$(cat helpers/run_dpo.sh)"   # from the laptop
set -euo pipefail
cd /workspace/code
export HF_HOME=/workspace/hf UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

uv run --no-sync python dpo.py
uv run --no-sync python dpo.py --candidates | while read -r tag e; do
  uv run --no-sync python eval.py --model main --adapter checkpoints/dpo/$tag/epoch-$e --split dev --run dpo-dev-$tag-e$e < /dev/null
done
uv run --no-sync python dpo.py --select
uv run --no-sync python eval.py --model main --adapter checkpoints/dpo/best --split test --run dpo
