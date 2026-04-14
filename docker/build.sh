#!/usr/bin/env bash
# Build cap-y Docker images with automatic platform and GPU architecture detection.
#
# Platform (PLATFORM_ARCH) maps directly to the Dockerfile folder:
#   aarch64  → docker/aarch64/  (Jetson Thor — source-compiled CUDA libs)
#   x86_64   → docker/x86_64/  (workstations/servers — pre-built GPU wheels)
#
# CUDA_ARCH_BIN is the GPU compute capability for source-compiled components:
#   11.0  Jetson Thor / GB10 (default on aarch64)
#   8.9   RTX 4080/4090 (default on x86_64)
#   9.0   H100 / A100 (x86_64)
#   12.0  RTX 5090 Blackwell desktop (x86_64)
#
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker/docker-compose.capx.yml}"
PLATFORM_ARCH=""
CUDA_ARCH_BIN=""
BUILD_ALL=0
BUILD_DEV=0

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --platform ARCH   Force platform (aarch64 or x86_64). Default: auto (uname -m)
  --cuda-arch CAP   GPU compute capability (e.g. 8.9, 9.0, 11.0, 12.0)
                    Default: auto-detect via nvidia-smi, else platform default
  --file FILE       Compose file (default: docker/docker-compose.capx.yml)
  --all             Build all images: base → default, base → open → nvidia
  --dev             Build cap-y:dev after base (all extras + sim venvs)
  -h, --help        Show this help

Image hierarchy:
  cap-y:base    MIT/Apache clean foundation (mink, DemoGrasp, rh56, OpenCV CUDA, JAX)
  cap-y:default + cuRobo + ContactGraspNet (NVIDIA NC — research only)
  cap-y:open    + ROS2 full stack + nvblox + Drake (from cap-y:base)
  cap-y:nvidia  + Isaac ROS + cuMotion + cuVSLAM (from cap-y:open)

Examples:
  # Auto-detect everything (recommended)
  ./docker/build.sh

  # Build full stack on RTX 4090
  ./docker/build.sh --all --cuda-arch 8.9

  # Force aarch64 build on x86_64 host (cross-build)
  ./docker/build.sh --platform aarch64 --cuda-arch 11.0
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --platform)     PLATFORM_ARCH="$2"; shift 2 ;;
    --cuda-arch)    CUDA_ARCH_BIN="$2"; shift 2 ;;
    # legacy flag alias
    --arch)         CUDA_ARCH_BIN="$2"; shift 2 ;;
    --file|-f)      COMPOSE_FILE="$2"; shift 2 ;;
    --all)          BUILD_ALL=1; shift ;;
    --dev)          BUILD_DEV=1; shift ;;
    -h|--help)      usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# ── Platform detection ────────────────────────────────────────────────────────
if [[ -z "$PLATFORM_ARCH" ]]; then
  PLATFORM_ARCH=$(uname -m)
fi

if [[ "$PLATFORM_ARCH" != "aarch64" && "$PLATFORM_ARCH" != "x86_64" ]]; then
  echo "ERROR: Unsupported platform '${PLATFORM_ARCH}'. Expected aarch64 or x86_64."
  exit 1
fi

echo "Platform: ${PLATFORM_ARCH}  →  using docker/${PLATFORM_ARCH}/ Dockerfiles"

# ── CUDA architecture detection ───────────────────────────────────────────────
if [[ -z "$CUDA_ARCH_BIN" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    CUDA_ARCH_BIN=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | head -1 | tr -d '[:space:]') || true
  fi
  if [[ -z "$CUDA_ARCH_BIN" ]]; then
    # Platform-specific defaults
    if [[ "$PLATFORM_ARCH" == "aarch64" ]]; then
      CUDA_ARCH_BIN="11.0"   # Jetson Thor (SM 110)
    else
      CUDA_ARCH_BIN="8.9"    # RTX 40-series (SM 89)
    fi
    echo "CUDA compute capability: not detected — using default ${CUDA_ARCH_BIN} for ${PLATFORM_ARCH}"
  else
    echo "CUDA compute capability: auto-detected ${CUDA_ARCH_BIN}"
  fi
fi

# ── Build ─────────────────────────────────────────────────────────────────────
export PLATFORM_ARCH
export CUDA_ARCH_BIN

echo ""
echo "Building cap-y:base  [platform=${PLATFORM_ARCH}, cuda_arch=${CUDA_ARCH_BIN}]"
docker compose -f "${COMPOSE_FILE}" build \
  --build-arg CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
  cap-y-base

if [[ "$BUILD_ALL" = "1" ]]; then
  echo ""
  echo "Building cap-y:default (+ cuRobo + ContactGraspNet — NVIDIA NC)"
  docker compose -f "${COMPOSE_FILE}" build cap-y-default

  echo ""
  echo "Building cap-y:open (+ ROS2 full stack + nvblox + Drake)"
  docker compose -f "${COMPOSE_FILE}" build \
    --build-arg CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
    cap-y-open

  echo ""
  echo "Building cap-y:nvidia (+ Isaac ROS + cuMotion + cuVSLAM)"
  docker compose -f "${COMPOSE_FILE}" build cap-y-nvidia

  echo ""
  echo "All images built:"
  docker image ls --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' \
    --filter reference='cap-y:*'
fi

if [[ "$BUILD_DEV" = "1" ]]; then
  echo ""
  echo "Building cap-y:dev (all extras + sim venvs — dev only)"
  docker compose -f "${COMPOSE_FILE}" build cap-y-dev
fi
