#!/usr/bin/env bash
# Runs ON THE POD: twin DPO (dpo_twin.py) from checkpoints/sft/best. Builds the twin set, trains every variant and
# learning rate, evals every epoch's adapter on dev with one vLLM engine, keeps each variant's best (lr, epoch), and
# evals it on test ONLY if it passes the dev gate against SFT (dpo_twin.gate).
#   git push, then: helpers/pod.sh 'BRANCH=<branch> bash /workspace/code/scripts/pod_setup.sh'
#   helpers/pod_job.sh dpo-twin "$(cat helpers/run_dpo_twin.sh)"                          # both variants
#   helpers/pod_job.sh dpo-twin "VARIANTS=dpo-twin $(cat helpers/run_dpo_twin.sh)"         # one variant
set -euo pipefail
cd /workspace/code
export HF_HOME=/workspace/hf UV_PROJECT_ENVIRONMENT=/workspace/venv UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
PY="uv run --no-sync python"
VARIANTS="${VARIANTS:-dpo-twin dpo-twin-out}"

$PY twins.py
ADAPTERS=(); RUNS=()
for V in $VARIANTS; do
  $PY dpo_twin.py --variant "$V"
  while read -r tag e; do
    ADAPTERS+=("checkpoints/$V/$tag/epoch-$e"); RUNS+=("$V-dev-$tag-e$e")
  done < <($PY dpo_twin.py --variant "$V" --candidates)
done
$PY eval.py --model main --split dev --adapter "${ADAPTERS[@]}" --run "${RUNS[@]}"
for V in $VARIANTS; do
  echo "== select $V"
  rc=0; $PY dpo_twin.py --variant "$V" --select || rc=$?
  if [ $rc -eq 0 ]; then
    $PY eval.py --model main --adapter "checkpoints/$V/best" --split test --run "$V"
    $PY compare.py sft "$V"
  elif [ $rc -eq 10 ]; then
    echo "== $V: best candidate does not pass the dev gate; test NOT run"
  else
    exit $rc
  fi
done
echo "== dpo-twin done"
