#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="docker-compose.capx.yml"
ARCH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --arch) ARCH="$2"; shift 2 ;;
    --file|-f) COMPOSE_FILE="$2"; shift 2 ;;
    *) echo "Usage: $0 [--arch CUDA_ARCH_BIN] [--file COMPOSE_FILE]"; exit 1 ;;
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

echo "Building capx-serving image with CUDA_ARCH_BIN=${ARCH}"
docker compose -f "${COMPOSE_FILE}" build \
  --build-arg CUDA_ARCH_BIN="${ARCH}"
