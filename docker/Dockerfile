# cap-y:default — extends cap-y:base with cuRobo and ContactGraspNet
#
# ── Dependency chain ─────────────────────────────────────────────────────────
#   nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
#   └── cap-y:base   (docker/Dockerfile.base — auto-selects arch at build time)
#       └── cap-y:default
#
# ── What this adds over cap-y:base ──────────────────────────────────────────
#   - cuRobo CUDA extensions (NVIDIA NC Research License — no commercial use)
#   - ContactGraspNet / PointNet2 CUDA extensions (NVlabs — no license)
#   - cmake + ninja added here (x86_64 only — aarch64 base already has them)
#
# ── License ──────────────────────────────────────────────────────────────────
#   NVIDIA NC License (cuRobo) + NVlabs no-license (ContactGraspNet)
#   Commercial use: NOT permitted without separate NVIDIA licensing.

ARG BASE_IMAGE=cap-y:base
FROM ${BASE_IMAGE}

# CUDA_ARCH_BIN is inherited from cap-y:base ENV — no re-declaration needed.
# Override the base default so the entrypoint re-links CUDA extensions on source mounts.
ENV CAPX_INSTALL_EXTRAS="contactgraspnet,curobo"

# cmake + ninja: required for CUDA extension compilation.
# aarch64 base already includes them; install only on x86_64.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build ccache \
      && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /tmp/capx-default-install
COPY pyproject.toml .
COPY capx/third_party/sam3 capx/third_party/sam3
COPY capx/third_party/contact_graspnet_pytorch capx/third_party/contact_graspnet_pytorch
COPY capx/third_party/curobo capx/third_party/curobo

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.ccache \
    mkdir -p capx && touch capx/__init__.py && \
    if [ "$(uname -m)" = "x86_64" ]; then \
      patch-pyproject-x86.sh pyproject.toml; \
    else \
      patch-pyproject.sh pyproject.toml; \
    fi && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.7.0 \
    uv pip install --system --no-build-isolation ".[contactgraspnet,curobo]"

RUN rm -rf /tmp/capx-default-install

WORKDIR /workspace

EXPOSE 8110 8113 8114 8115 8116 8117 8118
ENTRYPOINT ["/entrypoint.sh"]
