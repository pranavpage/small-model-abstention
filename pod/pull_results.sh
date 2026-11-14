#!/usr/bin/env bash
# Copy results from the pod (/workspace/abstention/results/<run>/*.json, *.jsonl, *.md)
# into results/<run>/ in the repo, which is tracked. Smoke runs (smoke-*) are skipped.
#   pod/pull_results.sh            # every run
#   pod/pull_results.sh ceiling    # selected runs
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; DEST="$HERE/../results"; mkdir -p "$DEST"
RUNS="${*:-}"
"$HERE/exec.sh" "
cd /workspace/abstention/results
RUNS=\"$RUNS\"; [ -n \"\$RUNS\" ] || RUNS=\$(ls -d */ | tr -d / | grep -v '^smoke-')
FILES=\$(for r in \$RUNS; do ls \$r/*.json \$r/*.jsonl \$r/*.md 2>/dev/null; done)
tar czf - \$FILES | base64 -w0; echo
" | grep -v '^__END__' | tr -d '\n' | base64 -d | tar xzf - -C "$DEST"
ls "$DEST"
