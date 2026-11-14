#!/usr/bin/env bash
# Run a shell snippet on a RunPod pod through the RunPod SSH proxy (<pod>@ssh.runpod.io),
# which only offers an interactive PTY: the snippet is base64-encoded, typed into the
# remote shell, and the output between markers is printed locally.
#
#   export POD_HOST=<pod-id>-<suffix>@ssh.runpod.io     # from the pod's "Connect" tab
#   pod/exec.sh 'nvidia-smi; ls /workspace'
#   pod/exec.sh - < script.sh                          # snippet from stdin
#
# Keep snippets short-lived; start long jobs with pod/job.sh.
set -euo pipefail
: "${POD_HOST:?set POD_HOST, e.g. abc123-64410000@ssh.runpod.io}"
KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
if [ "${1:-}" = "-" ]; then SNIPPET=$(cat); else SNIPPET="${1:?usage: exec.sh 'commands' | pod.sh - < file}"; fi
B64=$(printf '%s\n' "$SNIPPET" | base64 | tr -d '\n')

# The payload is sent in 1000-char lines (a TTY line holds at most 4096 chars).
# Markers are assembled remotely ("__BE''GIN__") so the PTY echo of our input never matches.
send() {
  sleep 2
  echo "stty -echo; rm -f /tmp/_pod_snippet.b64"
  printf '%s\n' "$B64" | fold -w 1000 | sed 's|^|echo |; s|$| >> /tmp/_pod_snippet.b64|'
  echo "base64 -d /tmp/_pod_snippet.b64 > /tmp/_pod_snippet.sh; echo __BE''GIN__; bash -l /tmp/_pod_snippet.sh 2>&1 </dev/null; echo __E''ND__ rc=\$?; exit"
  while kill -0 $$ 2>/dev/null; do sleep 5; done
}

ssh -tt -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" "$POD_HOST" < <(send) 2>/dev/null \
  | tr -d '\r' | sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' \
  | awk '/^__BEGIN__$/{p=1;next} /^__END__ rc=/{print; rc=$2; sub("rc=","",rc); exit rc+0} p'
