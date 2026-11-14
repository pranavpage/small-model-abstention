#!/usr/bin/env bash
# Runs ON THE POD (after pod/setup.sh): build data, check split hashes against the rehearsal,
# smoke test, then dev + test baselines for the main (1.7B) and debug (0.6B) models.
#   pod/job.sh baselines "$(cat pod/jobs/baselines.sh)"   # from the laptop; ships this file over SSH
set -euo pipefail
source /workspace/code/pod/env.sh

[ -f $ROOT/data/manifest.json ] || $PY abstention.data
uv run --no-sync python - <<'PY'
import json
a = json.load(open("/workspace/abstention/data/manifest.json"))["splits"]
b = json.load(open("rehearsal/data_manifest.json"))["splits"]
bad = [k for k in b if a.get(k, {}).get("sha256") != b[k]["sha256"]]
print("split hashes match rehearsal" if not bad else f"HASH MISMATCH: {bad}")
raise SystemExit(1 if bad else 0)
PY

run() {  # run <name> <eval args...>; skip if metrics already exist
  local name=$1; shift
  if [ -f $ROOT/results/$name/metrics.json ]; then echo "== skip $name (done)"; return; fi
  echo "== $name: $*"; $PY abstention.eval "$@" --run "$name"
}
run smoke-0.6b         --model debug --split dev --limit 20
run baseline-0.6b-dev  --model debug --split dev
run baseline-dev       --model main  --split dev
run baseline-0.6b      --model debug --split test
run baseline           --model main  --split test
echo "all baselines done"
