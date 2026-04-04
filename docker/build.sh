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
       echo "  --all    Build all 4 images: base → default, base → open → nvidia"
       echo ""
       echo "Image hierarchy:"
       echo "  cap-y:base    MIT/Apache clean foundation (mink, DemoGrasp, rh56)"
       echo "  cap-y:default + cuRobo + ContactGraspNet (NVIDIA NC — research only)"
       echo "  cap-y:open    + ROS2 full stack + dfx_inspire (from cap-y:base)"
       echo "  cap-y:nvidia  + Isaac ROS + cuMotion + cuVSLAM (from cap-y:open)"
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

echo "Building cap-y:base (MIT/Apache clean foundation) with CUDA_ARCH_BIN=${ARCH}"
docker compose -f "${COMPOSE_FILE}" build \
  --build-arg CUDA_ARCH_BIN="${ARCH}" \
  cap-y-base

if [[ "$BUILD_ALL" = "1" ]]; then
  echo ""
  echo "Building cap-y:default (+ cuRobo + ContactGraspNet — NVIDIA NC)"
  docker compose -f "${COMPOSE_FILE}" build cap-y-default

  echo ""
  echo "Building cap-y:open (+ ROS2 + dfx_inspire — from cap-y:base)"
  docker compose -f "${COMPOSE_FILE}" build cap-y-open

  echo ""
  echo "Building cap-y:nvidia (+ Isaac ROS + cuMotion + cuVSLAM — from cap-y:open)"
  docker compose -f "${COMPOSE_FILE}" build cap-y-nvidia

  echo ""
  echo "All images built:"
  docker image ls --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' \
    --filter reference='cap-y:*'
fi
