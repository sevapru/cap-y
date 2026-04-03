#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="docker-compose.capx.yml"
ARCH=""
BUILD_ALL=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --arch) ARCH="$2"; shift 2 ;;
    --file|-f) COMPOSE_FILE="$2"; shift 2 ;;
    --all) BUILD_ALL=1; shift ;;
    *) echo "Usage: $0 [--arch CUDA_ARCH_BIN] [--file COMPOSE_FILE] [--all]"
       echo ""
       echo "  --arch   GPU compute capability (e.g. 11.0 for Thor, 8.9 for RTX 4080)"
       echo "  --all    Build full chain: cap-y → cap-y-open → cap-y-nvidia"
       exit 1 ;;
  esac
done

if [[ -z "$ARCH" ]]; then
  if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Specify --arch manually (e.g. --arch 11.0)"
    exit 1
  fi
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]') || true
  if [[ -z "$ARCH" ]]; then
    echo "ERROR: Could not detect GPU compute capability. Specify --arch manually."
    exit 1
  fi
  echo "Detected GPU compute capability: ${ARCH}"
fi

echo "Building cap-y (base) with CUDA_ARCH_BIN=${ARCH}"
docker compose -f "${COMPOSE_FILE}" build \
  --build-arg CUDA_ARCH_BIN="${ARCH}" \
  capx-serving

if [[ "$BUILD_ALL" = "1" ]]; then
  echo ""
  echo "Building cap-y-open (ROS 2 + Nav2 + MoveIt 2 + Drake + LiveKit)"
  docker compose -f "${COMPOSE_FILE}" build capx-open

  echo ""
  echo "Building cap-y-nvidia (Isaac ROS + cuMotion + cuVSLAM)"
  docker compose -f "${COMPOSE_FILE}" build capx-nvidia

  echo ""
  echo "All images built:"
  docker image ls --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' \
    --filter reference='cap-y*'
fi
