#!/usr/bin/env bash
set -e

cd /workspace

# Tegra/Jetson containers run as root over host-owned bind mounts
git config --global --add safe.directory '*' 2>/dev/null || true

if [[ -z "${HF_TOKEN}" && -f .huggingfacekey ]]; then
  export HF_TOKEN=$(cat .huggingfacekey | tr -d '[:space:]')
  export HF_HOME="${HF_HOME:-/data/huggingface}"
  echo "[entrypoint] Loaded HF_TOKEN from .huggingfacekey"
fi

if [[ -f pyproject.toml ]]; then
  echo "[entrypoint] Re-installing cap-x in editable mode..."
  if ! uv pip install --system -e ".[contactgraspnet,curobo]" --no-build-isolation 2>&1; then
    echo "[entrypoint] WARNING: editable reinstall had errors; CUDA extensions may be stale"
  fi
fi

# If a command was passed (e.g. "bash"), run it instead of servers
if [[ $# -gt 0 ]]; then
  exec "$@"
fi

if [[ -f .openrouterkey ]]; then
  echo "[entrypoint] Starting OpenRouter LLM proxy on port 8110..."
  python -m capx.serving.openrouter_server \
    --key-file .openrouterkey \
    --port 8110 \
    --host "${CAPX_HOST:-0.0.0.0}" &
fi

echo "[entrypoint] Starting cap-x servers (profile: ${CAPX_PROFILE:-default})..."
exec python -m capx.serving.launch_servers \
  --profile "${CAPX_PROFILE:-default}" \
  --host "${CAPX_HOST:-0.0.0.0}"
