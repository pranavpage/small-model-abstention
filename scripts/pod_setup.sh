#!/usr/bin/env bash
# One-time (and after every pod restart) setup on a RunPod pod with the network volume at /workspace.
# Everything that must survive a restart (code, venv, HF cache, data, results, checkpoints) lives on /workspace.
#
#   curl -fsSL https://raw.githubusercontent.com/pranavpage/small-model-abstention/main/scripts/pod_setup.sh | BRANCH=main bash
#   # or, if the repo is already cloned:  bash /workspace/code/scripts/pod_setup.sh
set -euo pipefail

REPO="${REPO:-https://github.com/pranavpage/small-model-abstention.git}"
BRANCH="${BRANCH:-main}"
CODE="${CODE:-/workspace/code}"

export HF_HOME=/workspace/hf
export UV_CACHE_DIR=/root/.cache/uv   # container disk: a download cache, not worth network-volume space
export UV_PROJECT_ENVIRONMENT=/workspace/venv
export UV_LINK_MODE=copy

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ -d "$CODE/.git" ]; then
  git -C "$CODE" fetch origin "$BRANCH" && git -C "$CODE" checkout "$BRANCH" && git -C "$CODE" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO" "$CODE"
fi

cd "$CODE"
uv sync --extra gpu
uv run python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('vllm', vllm.__version__)"
uv run python -m pytest -q

# Persist env for new shells on this pod
grep -q "abstention env" ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'RC'
# abstention env
export HF_HOME=/workspace/hf UV_CACHE_DIR=/root/.cache/uv UV_PROJECT_ENVIRONMENT=/workspace/venv UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/code
RC
echo "setup done: code at $CODE, venv at /workspace/venv, outputs under /workspace/abstention"
