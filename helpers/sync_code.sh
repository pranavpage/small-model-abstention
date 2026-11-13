#!/usr/bin/env bash
# Copy this checkout's code (tracked + new non-ignored files, results/ and rehearsal/ excluded) into
# /workspace/code on the pod, so a branch can run there without pushing it first. The pod's git checkout
# shows these files as modified until the branch is merged and pulled (scripts/pod_setup.sh).
#   helpers/sync_code.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/.."
B64=$(git ls-files -co --exclude-standard | grep -v '^results/\|^rehearsal/' \
      | COPYFILE_DISABLE=1 tar czf - --no-xattrs -T - | base64 | tr -d '\n')
"$HERE/pod.sh" "
cd /workspace/code
echo $B64 | base64 -d | tar xzf - --no-same-owner --warning=no-unknown-keyword
find . -name '._*' -not -path './.git/*' -delete
echo synced to /workspace/code; git status --short
"
