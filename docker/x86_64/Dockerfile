# cap-y:default (x86_64) — extends cap-y:base with cuRobo and ContactGraspNet
#
# ── Dependency chain ─────────────────────────────────────────────────────────
#   nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
#   └── cap-y:base   (x86_64/Dockerfile.base  — GPU wheels, mink, DemoGrasp, JAX)
#       └── cap-y:default  ← YOU ARE HERE
#
# ── What this adds over cap-y:base ──────────────────────────────────────────
#   - cuRobo CUDA extensions (NVIDIA NC Research License — no commercial use)
#   - ContactGraspNet / PointNet2 CUDA extensions (NVlabs — no license)
#   - cmake + ninja added here (not in base) only for compiling CUDA extensions
#
# ── x86_64 vs aarch64 difference ────────────────────────────────────────────
#   Both arches compile CUDA extensions from source (no pip wheels for cuRobo).
#   Difference: x86_64 default SM target is 8.9 (RTX 40-series) vs 11.0 (Jetson Thor).
#
# ── License ──────────────────────────────────────────────────────────────────
#   NVIDIA NC License (cuRobo) + NVlabs no-license (ContactGraspNet)
#   Commercial use: NOT permitted without separate NVIDIA licensing.

ARG BASE_IMAGE=cap-y:base
FROM ${BASE_IMAGE}

# CUDA_ARCH_BIN is inherited from cap-y:base ENV — no re-declaration needed.
# Override the base default so the entrypoint re-links the CUDA extensions on source mounts.
ENV CAPX_INSTALL_EXTRAS="contactgraspnet,curobo"

# cmake + ninja are required to compile cuRobo and PointNet2/ContactGraspNet CUDA extensions.
# Not included in x86_64/Dockerfile.base to keep the base image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build ccache \
    && rm -rf /var/lib/apt/lists/*

# [D1] cuRobo + ContactGraspNet CUDA extensions
# Built from source against the CUDA/PyTorch already in the base image.
WORKDIR /tmp/capx-default-install
COPY pyproject.toml .
COPY capx/third_party/sam3 capx/third_party/sam3
COPY capx/third_party/contact_graspnet_pytorch capx/third_party/contact_graspnet_pytorch
COPY capx/third_party/curobo capx/third_party/curobo

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.ccache \
    mkdir -p capx && touch capx/__init__.py && \
    patch-pyproject-x86.sh pyproject.toml && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.7.0 \
    uv pip install --system --no-build-isolation \
      ".[contactgraspnet,curobo]"

RUN rm -rf /tmp/capx-default-install

WORKDIR /workspace

EXPOSE 8110 8113 8114 8115 8116 8117 8118

ENTRYPOINT ["/entrypoint.sh"]
