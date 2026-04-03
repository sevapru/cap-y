#!/usr/bin/env bash
# cap-y: one-line installer for Jetson Thor
# Usage: curl -sSL https://raw.githubusercontent.com/sevapru/cap-y/main/scripts/install.sh | bash
set -euo pipefail

echo ""
echo "  cap-y — CUDA robotics container for Jetson Thor"
echo "  by sobaka.dev  🐕"
echo ""

# Check for NVIDIA runtime
if ! docker info 2>/dev/null | grep -qE 'Runtimes.*nvidia|nvidia.*Runtimes'; then
  echo "ERROR: NVIDIA Container Toolkit not found."
  echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  exit 1
fi

# Detect GPU
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]') || true
if [[ -z "$ARCH" ]]; then
  echo "WARNING: Could not detect GPU. Defaulting to sm_110 (Jetson Thor)"
  ARCH="11.0"
fi
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown') (sm_${ARCH//./_})"

# Pull or build
echo ""
echo "  Pulling cap-y:latest..."
if docker pull ghcr.io/sevapru/cap-y:latest 2>/dev/null; then
  docker tag ghcr.io/sevapru/cap-y:latest cap-y:latest
  echo "  Pulled from ghcr.io ✓"
else
  echo "  Pre-built image not available for your platform. Building from source..."
  TMPDIR=$(mktemp -d)
  trap "rm -rf $TMPDIR" EXIT
  git clone --recurse-submodules https://github.com/sevapru/cap-y "$TMPDIR/cap-y"
  cd "$TMPDIR/cap-y/docker"
  ./build.sh --arch "$ARCH"
fi

# Verify
echo ""
echo "  Verifying CUDA..."
if ! docker run --rm --runtime nvidia --entrypoint python cap-y -c "import torch; print(f'  PyTorch {torch.__version__} — {torch.cuda.get_device_name(0)}')" 2>&1; then
  echo "  WARNING: CUDA verification failed. Container may still work -- check manually."
fi

echo ""
echo "  ✓ cap-y installed. Run:"
echo ""
echo "    docker run -it --rm --runtime nvidia --entrypoint bash -v .:/workspace cap-y"
echo ""
echo "  Or start perception servers:"
echo ""
echo "    git clone https://github.com/sevapru/cap-y && cd cap-y/docker"
echo "    docker compose -f docker-compose.capx.yml up -d"
echo ""
