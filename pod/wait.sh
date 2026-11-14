#!/usr/bin/env bash
# Block until a job started with pod/job.sh stops, then print its log tail.
#   pod/wait.sh <name> [poll_seconds] [lines]
set -euo pipefail
NAME="${1:?usage: wait.sh <name> [poll_seconds] [lines]}"; POLL="${2:-60}"; N="${3:-40}"
HERE="$(cd "$(dirname "$0")" && pwd)"
until "$HERE/log.sh" "$NAME" 1 | grep -q '^STATUS: stopped'; do sleep "$POLL"; done
"$HERE/log.sh" "$NAME" "$N"
