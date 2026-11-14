# Sourced ON THE POD by pod/setup.sh, every job in pod/jobs/, and ~/.bashrc.
# Everything that must survive a pod restart (code, venv, HF cache, data, results, checkpoints) lives on /workspace.
export CODE="${CODE:-/workspace/code}"
export HF_HOME=/workspace/hf
export UV_PROJECT_ENVIRONMENT=/workspace/venv
export UV_CACHE_DIR=/root/.cache/uv   # container disk: a download cache, not worth network-volume space
export UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
ROOT=/workspace/abstention             # config.json paths.root: data/, results/, checkpoints/, logs/
PY="uv run --no-sync python -m"        # $PY abstention.eval ...   (setup.sh already ran uv sync --extra gpu)
cd "$CODE"
