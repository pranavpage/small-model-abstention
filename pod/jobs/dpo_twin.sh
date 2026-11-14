#!/usr/bin/env bash
# Runs ON THE POD: twin DPO (abstention/dpo_twin.py) from checkpoints/sft/best. Builds the twin set, trains every variant and
# learning rate, evals every epoch's adapter on dev with one vLLM engine, keeps each variant's best (lr, epoch), and
# evals it on test ONLY if it passes the dev gate against SFT (dpo_twin.gate).
#   git push, then: pod/exec.sh 'BRANCH=<branch> bash /workspace/code/pod/setup.sh'
#   pod/job.sh dpo-twin "$(cat pod/jobs/dpo_twin.sh)"                          # both variants
#   pod/job.sh dpo-twin "VARIANTS=dpo-twin $(cat pod/jobs/dpo_twin.sh)"         # one variant
set -euo pipefail
source /workspace/code/pod/env.sh
VARIANTS="${VARIANTS:-dpo-twin dpo-twin-out}"

$PY abstention.twins
ADAPTERS=(); RUNS=()
for V in $VARIANTS; do
  $PY abstention.dpo_twin --variant "$V"
  while read -r tag e; do
    ADAPTERS+=("checkpoints/$V/$tag/epoch-$e"); RUNS+=("$V-dev-$tag-e$e")
  done < <($PY abstention.dpo_twin --variant "$V" --candidates)
done
$PY abstention.eval --model main --split dev --adapter "${ADAPTERS[@]}" --run "${RUNS[@]}"
for V in $VARIANTS; do
  echo "== select $V"
  rc=0; $PY abstention.dpo_twin --variant "$V" --select || rc=$?
  if [ $rc -eq 0 ]; then
    $PY abstention.eval --model main --adapter "checkpoints/$V/best" --split test --run "$V"
    $PY abstention.compare sft "$V"
  elif [ $rc -eq 10 ]; then
    echo "== $V: best candidate does not pass the dev gate; test NOT run"
  else
    exit $rc
  fi
done
echo "== dpo-twin done"
