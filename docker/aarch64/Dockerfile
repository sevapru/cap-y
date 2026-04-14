# cap-y:default — extends cap-y:base with cuRobo and ContactGraspNet
#
# License: NVIDIA NC License (cuRobo) + no-license (ContactGraspNet/NVlabs)
# Commercial use: NOT permitted without separate NVIDIA licensing.
# See NOTICE for full component inventory.
#
# Build:
#   docker compose -f docker/docker-compose.capx.yml build cap-y-default
# Or standalone:
#   docker build -f docker/Dockerfile --build-arg BASE_IMAGE=cap-y:base -t cap-y:default .

ARG BASE_IMAGE=cap-y:base
FROM ${BASE_IMAGE}

# [D1] cuRobo + ContactGraspNet CUDA extensions
# These are built from source against the CUDA/PyTorch already in the base image.
WORKDIR /tmp/capx-default-install
COPY pyproject.toml .
COPY capx/third_party/sam3 capx/third_party/sam3
COPY capx/third_party/contact_graspnet_pytorch capx/third_party/contact_graspnet_pytorch
COPY capx/third_party/curobo capx/third_party/curobo

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.ccache \
    mkdir -p capx && touch capx/__init__.py && \
    patch-pyproject.sh pyproject.toml && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.7.0 \
    uv pip install --system --no-build-isolation \
      ".[contactgraspnet,curobo]"

RUN rm -rf /tmp/capx-default-install

WORKDIR /workspace

EXPOSE 8110 8113 8114 8115 8116 8117 8118

ENTRYPOINT ["/entrypoint.sh"]
