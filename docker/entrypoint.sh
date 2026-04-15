#!/usr/bin/env bash
set -e

cd /workspace

# Verify OpenCV was built with CUDA (libcuda.so.1 requires live GPU — cannot check at build time)
python3 -c "
import cv2, sys
info = cv2.getBuildInformation()
if 'CUDA' not in info:
    print('[entrypoint] WARNING: OpenCV was NOT built with CUDA support', file=sys.stderr)
else:
    cuda_lines = [l.strip() for l in info.splitlines() if any(k in l for k in ('CUDA','cuDNN','NVCUVID','NvVideoCodec'))]
    print('[entrypoint] OpenCV', cv2.__version__, '— CUDA build OK:', ', '.join(cuda_lines[:3]))
" 2>/dev/null || true

# Tegra/Jetson containers run as root over host-owned bind mounts
git config --global --add safe.directory '*' 2>/dev/null || true

if [[ -z "${HF_TOKEN}" && -f .huggingfacekey ]]; then
  export HF_TOKEN=$(cat .huggingfacekey | tr -d '[:space:]')
  export HF_HOME="${HF_HOME:-/data/huggingface}"
  echo "[entrypoint] Loaded HF_TOKEN from .huggingfacekey"
fi

if [[ -f pyproject.toml ]]; then
  # CAPX_INSTALL_EXTRAS is set per image (base="", default="contactgraspnet,curobo", etc.)
  # so this editable reinstall only re-links extensions already compiled in the image.
  _EXTRAS="${CAPX_INSTALL_EXTRAS:-}"
  _TARGET="${_EXTRAS:+.[${_EXTRAS}]}"
  _TARGET="${_TARGET:-"."}"
  echo "[entrypoint] Re-installing cap-x in editable mode (extras: ${_EXTRAS:-none})..."
  if ! uv pip install --system -e "${_TARGET}" --no-build-isolation 2>&1; then
    echo "[entrypoint] WARNING: editable reinstall had errors; CUDA extensions may be stale"
  fi
fi

# DemoGrasp checkpoint hint (first run only)
if [ -d /opt/demograsp ] && [ ! -f /opt/demograsp/ckpt/inspire.pt ]; then
  echo "[entrypoint] DemoGrasp checkpoints not found. Download with:"
  echo "  bash scripts/download_demograsp_ckpts.sh"
  echo "  Source: https://github.com/BeingBeyond/DemoGrasp#requirements"
fi

# If a command was passed (e.g. "bash"), run it instead of servers
if [[ $# -gt 0 ]]; then
  exec "$@"
fi

rm -f /tmp/.capx-ready

if [[ -f .openrouterkey ]]; then
  echo "[entrypoint] Starting OpenRouter LLM proxy on port 8110..."
  python -m capx.serving.openrouter_server \
    --key-file .openrouterkey \
    --port 8110 \
    --host "${CAPX_HOST:-0.0.0.0}" &
fi

echo "[entrypoint] Starting cap-x servers (profile: ${CAPX_PROFILE:-default})..."
python -m capx.serving.launch_servers \
  --profile "${CAPX_PROFILE:-default}" \
  --host "${CAPX_HOST:-0.0.0.0}" &
SERVER_PID=$!

# Forward SIGTERM/SIGINT to the server process so tini → bash → python gets graceful shutdown.
# Without this, tini sends SIGTERM to bash (PID 1's child), bash exits, and the kernel
# SIGKILL's the orphaned python process with no chance for connection draining or cleanup.
trap 'kill -TERM $SERVER_PID 2>/dev/null; wait $SERVER_PID' TERM INT

# Wait for gateway to bind (max 60s), then signal readiness for Docker healthcheck
for _i in $(seq 1 60); do
  if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 8100)); s.close()" 2>/dev/null; then
    touch /tmp/.capx-ready
    echo "[entrypoint] Servers ready (gateway on :8100)"
    break
  fi
  sleep 1
done

if [ ! -f /tmp/.capx-ready ]; then
  echo "[entrypoint] WARNING: servers did not become ready within 60s" >&2
fi

wait $SERVER_PID
