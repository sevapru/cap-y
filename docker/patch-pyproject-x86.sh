#!/bin/bash
# Patch pyproject.toml for x86_64 images (docker/x86_64/).
#
# Same intent as patch-pyproject.sh (aarch64) but adapted for wheel-based installs:
#   - open3d: suppressed (already installed from PyPI before this runs)
#   - opencv-python-headless: suppressed (cudawarped CUDA wheel + fake dist-info in Dockerfile)
#   - jax / jaxlib version ceilings: removed (jax[cuda13] installs a newer version)
#   - numpy pin: removed (same as aarch64)
#   - Editable installs: stripped (meaningless inside Docker build layer)
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: patch-pyproject-x86.sh <pyproject.toml>" >&2
  exit 1
fi

sed -i \
  -e '/"open3d<=0.18.0 ; platform_machine/d' \
  -e 's/"open3d",/"open3d ; sys_platform == '\''never'\''",/' \
  -e '/"jax<0.4.30"/d' \
  -e '/"jaxlib<0.4.30"/d' \
  -e '/"numpy==1.26.4"/d' \
  -e '/"opencv-python-headless<4.13"/d' \
  -e 's/"opencv-python-headless",/"opencv-python-headless ; sys_platform == '\''never'\''",/' \
  -e 's/, editable = true//g' \
  -e 's/editable = true, //g' \
  "$1"
