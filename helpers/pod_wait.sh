#!/usr/bin/env bash
# Block until a job started with helpers/pod_job.sh stops, then print its log tail.
#   helpers/pod_wait.sh <name> [poll_seconds] [lines]
set -euo pipefail
NAME="${1:?usage: pod_wait.sh <name> [poll_seconds] [lines]}"; POLL="${2:-60}"; N="${3:-40}"
HERE="$(cd "$(dirname "$0")" && pwd)"
until "$HERE/pod_log.sh" "$NAME" 1 | grep -q '^STATUS: stopped'; do sleep "$POLL"; done
"$HERE/pod_log.sh" "$NAME" "$N"
