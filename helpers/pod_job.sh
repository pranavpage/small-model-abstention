#!/usr/bin/env bash
# Start a long-running command on the pod, detached (survives SSH disconnects), logging to
# /workspace/abstention/logs/<name>.log. Check on it with helpers/pod_log.sh <name>.
#
#   helpers/pod_job.sh baseline-dev 'cd /workspace/code && uv run python eval.py --model main --split dev --run baseline-dev'
set -euo pipefail
NAME="${1:?usage: pod_job.sh <name> '<command>'}"; CMD="${2:?usage: pod_job.sh <name> '<command>'}"
HERE="$(cd "$(dirname "$0")" && pwd)"
B64=$(printf '%s\n' "$CMD" | base64 | tr -d '\n')
"$HERE/pod.sh" "
LOGS=/workspace/abstention/logs; mkdir -p \$LOGS
echo $B64 | base64 -d > \$LOGS/$NAME.cmd.sh
if [ -f \$LOGS/$NAME.pid ] && kill -0 \$(cat \$LOGS/$NAME.pid) 2>/dev/null; then echo 'already running: $NAME'; exit 1; fi
setsid nohup bash -lc \"source ~/.bashrc 2>/dev/null; bash \$LOGS/$NAME.cmd.sh; echo __JOB_EXIT__ rc=\\\$?\" > \$LOGS/$NAME.log 2>&1 < /dev/null &
echo \$! > \$LOGS/$NAME.pid
echo started $NAME pid \$(cat \$LOGS/$NAME.pid) log \$LOGS/$NAME.log
"
