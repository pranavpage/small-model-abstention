#!/usr/bin/env bash
# Show status and the tail of a job started with helpers/pod_job.sh.   helpers/pod_log.sh <name> [lines]
set -euo pipefail
NAME="${1:?usage: pod_log.sh <name> [lines]}"; N="${2:-30}"
HERE="$(cd "$(dirname "$0")" && pwd)"
"$HERE/pod.sh" "
L=/workspace/abstention/logs
if kill -0 \$(cat \$L/$NAME.pid 2>/dev/null) 2>/dev/null; then echo STATUS: running; else echo STATUS: stopped; fi
tail -n $N \$L/$NAME.log | tr '\r' '\n' | tail -n $N
"
