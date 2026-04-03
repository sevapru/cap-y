ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG CUDA_ARCH_BIN="11.0"
ARG OPENCV_VERSION="4.13.0"
ARG PYTHON_VERSION="3.12"
ARG TORCH_VERSION="2.11.0"

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_BIN}" \
    CCACHE_DIR=/root/.ccache \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    build-essential cmake ninja-build git ccache pkg-config \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev libv4l-dev \
    libeigen3-dev libtbb-dev \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev \
    libegl1 libgl1 libglvnd-dev \
    libgtk-3-dev \
    curl wget unzip file \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv

RUN uv pip install --system numpy==1.26.4

RUN mkdir -p /opt/opencv-build && cd /opt/opencv-build && \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git

RUN ln -sfnv /usr/include/$(uname -m)-linux-gnu/cudnn_version_v*.h \
    /usr/include/$(uname -m)-linux-gnu/cudnn_version.h 2>/dev/null || true

RUN cd /opt/opencv-build && \
    sed -i 's|weight != 1.0|(float)weight != 1.0f|' \
      opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp && \
    sed -i 's|nms_iou_threshold > 0|(float)nms_iou_threshold > 0.0f|' \
      opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp
# ---------------------------------------------------------------------------
# OpenCV from source
# Build with CUDA on Jetson
# https://docs.opencv.org/4.13.0/d6/d00/tutorial_py_root.html
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.ccache \
    cd /opt/opencv-build && mkdir -p build && cd build && \
    cmake -G Ninja ../opencv \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
      -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
      -DWITH_CUDA=ON \
      -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
      -DCUDA_ARCH_PTX= \
      -DCUDA_FAST_MATH=ON \
      -DWITH_CUBLAS=ON \
      -DWITH_CUDNN=ON \
      -DOPENCV_DNN_CUDA=ON \
      -DCUDNN_INCLUDE_DIR=/usr/include/$(uname -m)-linux-gnu \
      -DWITH_EIGEN=ON \
      -DEIGEN_INCLUDE_PATH=/usr/include/eigen3 \
      -DWITH_TBB=ON \
      -DWITH_GSTREAMER=ON \
      -DWITH_FFMPEG=ON \
      -DWITH_V4L=ON \
      -DWITH_LIBV4L=ON \
      -DWITH_OPENGL=ON \
      -DOpenGL_GL_PREFERENCE=GLVND \
      -DWITH_GTK=ON \
      -DOPENCV_ENABLE_NONFREE=ON \
      -DOPENCV_GENERATE_PKGCONFIG=ON \
      -DWITH_OPENCL=OFF \
      -DWITH_IPP=OFF \
      -DBUILD_opencv_rgbd=OFF \
      -DBUILD_opencv_python2=OFF \
      -DBUILD_opencv_python3=ON \
      -DPYTHON3_EXECUTABLE=/usr/bin/python3.12 \
      -DPYTHON3_INCLUDE_DIR=$(python3.12 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
      -DPYTHON3_LIBRARY=$(find /usr/lib -name 'libpython3.12*.so*' | head -1) \
      -DPYTHON3_PACKAGES_PATH=/usr/local/lib/python3.12/dist-packages \
      -DPYTHON3_NUMPY_INCLUDE_DIRS=$(python3.12 -c "import numpy; print(numpy.get_include())" 2>/dev/null || echo "/usr/lib/python3/dist-packages/numpy/core/include") \
      -DBUILD_opencv_java=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_DOCS=OFF \
      $([ "$(uname -m)" = "aarch64" ] && echo "-DENABLE_NEON=ON") \
    && cmake --build . --parallel $(nproc) \
    && cmake --install . \
    && ldconfig \
    && OPENCV_DIST="/usr/local/lib/python3.12/dist-packages/opencv_python_headless-${OPENCV_VERSION}.dist-info" \
    && mkdir -p "$OPENCV_DIST" \
    && printf "Metadata-Version: 2.1\nName: opencv-python-headless\nVersion: ${OPENCV_VERSION}\n" > "$OPENCV_DIST/METADATA" \
    && echo "opencv-python-headless" > "$OPENCV_DIST/top_level.txt" \
    && echo "cv2/__init__.py,," > "$OPENCV_DIST/RECORD" \
    && rm -rf /opt/opencv-build

RUN python3 -c "import cv2; print('OpenCV', cv2.__version__); assert 'CUDA' in cv2.getBuildInformation(), 'NOT built with CUDA'; print('CUDA build: OK')" \
    && python3 -c "import cv2; [print(l) for l in cv2.getBuildInformation().splitlines() if any(k in l for k in ('CUDA','cuDNN','GStreamer','NEON','NVIDIA'))]"

ENV UV_HTTP_TIMEOUT=300
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    torch==${TORCH_VERSION} torchvision \
    --index-url https://download.pytorch.org/whl/cu130

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system hatchling setuptools wheel scikit-build-core yapf

# ---------------------------------------------------------------------------
# Open3D from source
# Build with CUDA on Jetson, GUI on (full OpenGL available)
# https://www.open3d.org/docs/release/arm.html
# ---------------------------------------------------------------------------
# Use main branch: v0.19.0 fails with CUDA 13.0 due to outdated stdgpu
# Fixed in main via PRs #7083, #7398 (issues #7376, #7397)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libusb-1.0-0-dev liblzf-dev libfmt-dev pybind11-dev \
    clang libc++-dev libc++abi-dev libomp-dev libglu1-mesa-dev libssl-dev \
    libcurl4-openssl-dev \
    gfortran nasm apt-transport-https \
    && mkdir -p /etc/apt/keyrings \
    && curl -sSf https://librealsense.realsenseai.com/Debian/librealsenseai.asc | \
       gpg --dearmor > /etc/apt/keyrings/librealsenseai.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo noble main" > \
       /etc/apt/sources.list.d/librealsense.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends librealsense2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch main --recursive https://github.com/isl-org/Open3D.git /opt/open3d && \
    git clone --depth 1 --branch main https://github.com/isl-org/Open3D-ML.git /opt/open3d/Open3D-ML

RUN --mount=type=cache,target=/root/.ccache \
    export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_BIN}" && \
    cd /opt/open3d && \
    sed -i '/set(TORCH_CUDA_ARCH_LIST "")/,/endforeach()/c\        set(TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")' 3rdparty/cmake/FindPytorch.cmake && \
    mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CUDA_STANDARD=17 \
      -DCMAKE_CUDA_FLAGS="--generate-code=arch=compute_$(echo ${CUDA_ARCH_BIN} | tr -d '.'),code=sm_$(echo ${CUDA_ARCH_BIN} | tr -d '.')" \
      -DTORCH_CUDA_ARCH_LIST="${CUDA_ARCH_BIN}" \
      -DBUILD_CUDA_MODULE=ON \
      -DBUILD_GUI=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DENABLE_HEADLESS_RENDERING=OFF \
      -DBUILD_WEBRTC=OFF \
      -DBUILD_LIBREALSENSE=ON \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_OPEN3D_ML=ON \
      -DOPEN3D_ML_ROOT=/opt/open3d/Open3D-ML \
      -DUSE_BLAS=ON \
      -DUSE_SYSTEM_CURL=ON \
      -DBUILD_UNIT_TESTS=OFF \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DPYTHON_EXECUTABLE=/usr/bin/python3.12 \
    && make -j$(nproc) \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && mkdir -p /opt/wheels \
    && LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" make install-pip-package -j$(nproc) \
    && cp /opt/open3d/build/lib/python_package/dist/open3d-*.whl /opt/wheels/ 2>/dev/null || true \
    && rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && ldconfig \
    && rm -rf /opt/open3d

RUN python3 -c "import importlib.util; assert importlib.util.find_spec('open3d') is not None, 'open3d not installed'; print('Open3D package found')"

# ---------------------------------------------------------------------------
# cap-x dependencies + CUDA extensions (CuRobo, PointNet2)
# ---------------------------------------------------------------------------
WORKDIR /tmp/capx-install
COPY pyproject.toml .
COPY capx/third_party/sam3 capx/third_party/sam3
COPY capx/third_party/contact_graspnet_pytorch capx/third_party/contact_graspnet_pytorch
COPY capx/third_party/curobo capx/third_party/curobo

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.ccache \
    mkdir -p capx && touch capx/__init__.py && \
    sed -i 's|"open3d<=0.18.0 ; platform_machine == '"'"'aarch64'"'"'",||' pyproject.toml && \
    sed -i 's|"open3d",|"open3d ; sys_platform == '"'"'never'"'"'",|' pyproject.toml && \
    sed -i 's|"jax<0.4.30",||' pyproject.toml && \
    sed -i 's|"jaxlib<0.4.30",||' pyproject.toml && \
    sed -i 's|"numpy==1.26.4",||' pyproject.toml && \
    sed -i 's|, editable = true||g' pyproject.toml && \
    sed -i 's|editable = true, ||g' pyproject.toml && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.7.0 \
    uv pip install --system --no-build-isolation \
      ".[contactgraspnet,curobo]"

RUN rm -rf /tmp/capx-install

WORKDIR /tmp/libero-install
COPY pyproject.toml .
COPY capx/third_party/sam3 capx/third_party/sam3
COPY capx/third_party/contact_graspnet_pytorch capx/third_party/contact_graspnet_pytorch
COPY capx/third_party/curobo capx/third_party/curobo
COPY capx/third_party/libero_dependencies capx/third_party/libero_dependencies
COPY capx/third_party/LIBERO-PRO capx/third_party/LIBERO-PRO

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.ccache \
    mkdir -p capx && touch capx/__init__.py && \
    uv venv /opt/venv-libero --python python${PYTHON_VERSION} --system-site-packages && \
    sed -i 's|"open3d<=0.18.0 ; platform_machine == '"'"'aarch64'"'"'",||' pyproject.toml && \
    sed -i 's|"open3d",|"open3d ; sys_platform == '"'"'never'"'"'",|' pyproject.toml && \
    sed -i 's|"jax<0.4.30",||' pyproject.toml && \
    sed -i 's|"jaxlib<0.4.30",||' pyproject.toml && \
    sed -i 's|"numpy==1.26.4",||' pyproject.toml && \
    sed -i 's|, editable = true||g' pyproject.toml && \
    sed -i 's|editable = true, ||g' pyproject.toml && \
    uv pip install --python /opt/venv-libero/bin/python \
      --no-build-isolation \
      ".[libero,contactgraspnet]" || \
    echo "WARNING: LIBERO venv install had errors (may need runtime install)"

RUN rm -rf /tmp/libero-install

# JAX with CUDA 13 -- installed last so numpy upgrade is not reverted
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "jax[cuda13]"

WORKDIR /workspace

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8110 8113 8114 8115 8116 8117 8118

ENTRYPOINT ["/entrypoint.sh"]
