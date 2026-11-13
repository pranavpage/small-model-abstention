#!/usr/bin/env bash
# Runs ON THE POD: one DPO variant (a key of dpo.variants) from checkpoints/sft/best. Trains every learning rate,
# evals each epoch's adapter on dev, keeps the best (lr, epoch), and evals it on test ONLY if it beats SFT's dev
# balanced score.
#   git push, then: helpers/pod.sh 'BRANCH=<branch> bash /workspace/code/scripts/pod_setup.sh'
#   helpers/pod_job.sh dpo-v2 "VARIANT=dpo-v2 $(cat helpers/run_dpo.sh)"   # from the laptop
set -euo pipefail
: "${VARIANT:?set VARIANT, e.g. dpo-v2}"
cd /workspace/code
export HF_HOME=/workspace/hf UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

uv run --no-sync python dpo.py --variant "$VARIANT"
uv run --no-sync python dpo.py --variant "$VARIANT" --candidates | while read -r tag e; do
  uv run --no-sync python eval.py --model main --adapter checkpoints/$VARIANT/$tag/epoch-$e --split dev \
    --run $VARIANT-dev-$tag-e$e < /dev/null
done
rc=0; uv run --no-sync python dpo.py --variant "$VARIANT" --select || rc=$?
if [ $rc -eq 0 ]; then
  uv run --no-sync python eval.py --model main --adapter checkpoints/$VARIANT/best --split test --run "$VARIANT"
elif [ $rc -eq 10 ]; then
  echo "== $VARIANT: best candidate does not beat SFT on dev; test NOT run"
else
  exit $rc
fi
