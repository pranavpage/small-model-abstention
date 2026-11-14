#!/usr/bin/env bash
# Runs ON THE POD, once per pod and after every restart: clone or update the code, install the venv, check the GPU,
# run the CPU tests. Needs the network volume mounted at /workspace (see pod/env.sh for what lives where).
#
#   curl -fsSL https://raw.githubusercontent.com/pranavpage/small-model-abstention/main/pod/setup.sh | BRANCH=main bash
#   pod/exec.sh 'BRANCH=<branch> bash /workspace/code/pod/setup.sh'      # from the laptop, repo already cloned
set -euo pipefail

REPO="${REPO:-https://github.com/pranavpage/small-model-abstention.git}"
BRANCH="${BRANCH:-main}"
CODE="${CODE:-/workspace/code}"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ -d "$CODE/.git" ]; then
  git -C "$CODE" fetch origin "$BRANCH" && git -C "$CODE" checkout "$BRANCH" && git -C "$CODE" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO" "$CODE"
fi

source "$CODE/pod/env.sh"
uv sync --extra gpu
uv run python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('vllm', vllm.__version__)"
uv run python -m pytest -q

# New shells on this pod get the same environment.
grep -q "pod/env.sh" ~/.bashrc 2>/dev/null || echo "source $CODE/pod/env.sh" >> ~/.bashrc
echo "setup done: code at $CODE, venv at $UV_PROJECT_ENVIRONMENT, outputs under $ROOT"
