#!/bin/bash
# Neutralize pyproject.toml entries that conflict with Docker source builds.
# Source-built packages (Open3D, JAX) are already installed system-wide;
# uv must not try to resolve them from PyPI (no aarch64 wheels exist).
# Editable installs are meaningless during Docker build -- strip them.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: patch-pyproject.sh <pyproject.toml>" >&2
  exit 1
fi

sed -i \
  -e '/"open3d<=0.18.0 ; platform_machine/d' \
  -e 's/"open3d",/"open3d ; sys_platform == '\''never'\''",/' \
  -e '/"jax<0.4.30"/d' \
  -e '/"jaxlib<0.4.30"/d' \
  -e '/"numpy==1.26.4"/d' \
  -e 's/, editable = true//g' \
  -e 's/editable = true, //g' \
  "$1"
