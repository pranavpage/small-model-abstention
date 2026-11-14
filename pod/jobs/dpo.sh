#!/usr/bin/env bash
# Runs ON THE POD: one DPO variant (a key of dpo.variants) from checkpoints/sft/best. Trains every learning rate,
# evals each epoch's adapter on dev, keeps the best (lr, epoch), and evals it on test ONLY if it beats SFT's dev
# balanced score.
#   git push, then: pod/exec.sh 'BRANCH=<branch> bash /workspace/code/pod/setup.sh'
#   pod/job.sh dpo-v2 "VARIANT=dpo-v2 $(cat pod/jobs/dpo.sh)"   # from the laptop
set -euo pipefail
: "${VARIANT:?set VARIANT, e.g. dpo-v2}"
source /workspace/code/pod/env.sh

$PY abstention.dpo --variant "$VARIANT"
$PY abstention.dpo --variant "$VARIANT" --candidates | while read -r tag e; do
  $PY abstention.eval --model main --adapter checkpoints/$VARIANT/$tag/epoch-$e --split dev \
    --run $VARIANT-dev-$tag-e$e < /dev/null
done
rc=0; $PY abstention.dpo --variant "$VARIANT" --select || rc=$?
if [ $rc -eq 0 ]; then
  $PY abstention.eval --model main --adapter checkpoints/$VARIANT/best --split test --run "$VARIANT"
elif [ $rc -eq 10 ]; then
  echo "== $VARIANT: best candidate does not beat SFT on dev; test NOT run"
else
  exit $rc
fi
