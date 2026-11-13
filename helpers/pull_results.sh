#!/usr/bin/env bash
# Copy results from the pod (/workspace/abstention/results/<run>/{metrics.json,handcheck.jsonl,outputs.jsonl,train_*})
# into results/<run>/ in the repo, which is tracked. Smoke runs (smoke-*) are skipped.
#   helpers/pull_results.sh            # every run
#   helpers/pull_results.sh ceiling    # selected runs
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; DEST="$HERE/../results"; mkdir -p "$DEST"
RUNS="${*:-}"
"$HERE/pod.sh" "
cd /workspace/abstention/results
RUNS=\"$RUNS\"; [ -n \"\$RUNS\" ] || RUNS=\$(ls -d */ | tr -d / | grep -v '^smoke-')
FILES=\$(for r in \$RUNS; do ls \$r/metrics.json \$r/handcheck.jsonl \$r/outputs.jsonl \$r/train_summary.json \$r/train_log.jsonl 2>/dev/null; done)
tar czf - \$FILES | base64 -w0; echo
" | grep -v '^__END__' | tr -d '\n' | base64 -d | tar xzf - -C "$DEST"
ls "$DEST"
