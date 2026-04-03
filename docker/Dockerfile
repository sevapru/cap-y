ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG CUDA_ARCH_BIN="11.0"
ARG OPENCV_VERSION="4.13.0"
ARG PYTHON_VERSION="3.12"
ARG TORCH_VERSION="2.11.0"
ARG OPEN3D_BRANCH=main
ARG JAX_VERSION=0.9.2

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_BIN}" \
    CCACHE_DIR=/root/.ccache \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    UV_HTTP_TIMEOUT=300

# [1] System deps (single layer: base + OpenCV + Open3D + RealSense)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python3-pip \
    build-essential cmake ninja-build git ccache pkg-config \
    curl wget unzip file \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev libv4l-dev \
    libeigen3-dev libtbb-dev libgtk-3-dev \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev \
    libegl1 libgl1 libglvnd-dev libglu1-mesa-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libusb-1.0-0-dev liblzf-dev libfmt-dev pybind11-dev \
    clang libc++-dev libc++abi-dev libomp-dev \
    libssl-dev libcurl4-openssl-dev \
    gfortran nasm apt-transport-https \
    && mkdir -p /etc/apt/keyrings \
    && curl -sSf https://librealsense.realsenseai.com/Debian/librealsenseai.asc | \
       gpg --dearmor > /etc/apt/keyrings/librealsenseai.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo noble main" > \
       /etc/apt/sources.list.d/librealsense.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends librealsense2-dev \
    && rm -rf /var/lib/apt/lists/*

# [2] Python + uv
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    rm -f /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv && \
    uv pip install --system "numpy>=2.0,<3"

# [3] OpenCV 4.13 CUDA
RUN mkdir -p /opt/wheels /opt/opencv-build && cd /opt/opencv-build && \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git

RUN ln -sfnv /usr/include/$(uname -m)-linux-gnu/cudnn_version_v*.h \
    /usr/include/$(uname -m)-linux-gnu/cudnn_version.h 2>/dev/null || true

# FP16 half-cast patches from jetson-containers
RUN cd /opt/opencv-build && \
    sed -i 's|weight != 1.0|(float)weight != 1.0f|' \
      opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp && \
    sed -i 's|nms_iou_threshold > 0|(float)nms_iou_threshold > 0.0f|' \
      opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp

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
      -DPYTHON3_EXECUTABLE=/usr/bin/python${PYTHON_VERSION} \
      -DPYTHON3_INCLUDE_DIR=$(python${PYTHON_VERSION} -c "from sysconfig import get_paths; print(get_paths()['include'])") \
      -DPYTHON3_LIBRARY=$(find /usr/lib -name 'libpython${PYTHON_VERSION}*.so*' | head -1) \
      -DPYTHON3_PACKAGES_PATH=/usr/local/lib/python${PYTHON_VERSION}/dist-packages \
      -DPYTHON3_NUMPY_INCLUDE_DIRS=$(python${PYTHON_VERSION} -c "import numpy; print(numpy.get_include())" 2>/dev/null || echo "/usr/lib/python3/dist-packages/numpy/core/include") \
      -DBUILD_opencv_java=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_DOCS=OFF \
      $([ "$(uname -m)" = "aarch64" ] && echo "-DENABLE_NEON=ON") \
    && cmake --build . --parallel $(nproc) \
    && cmake --install . \
    && ldconfig \
    && OPENCV_DIST="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/opencv_python_headless-${OPENCV_VERSION}.dist-info" \
    && mkdir -p "$OPENCV_DIST" \
    && printf "Metadata-Version: 2.1\nName: opencv-python-headless\nVersion: ${OPENCV_VERSION}\n" > "$OPENCV_DIST/METADATA" \
    && echo "opencv-python-headless" > "$OPENCV_DIST/top_level.txt" \
    && echo "cv2/__init__.py,," > "$OPENCV_DIST/RECORD" \
    && rm -rf /opt/opencv-build

RUN python3 -c "import cv2; print('OpenCV', cv2.__version__); assert 'CUDA' in cv2.getBuildInformation(), 'NOT built with CUDA'; print('CUDA build: OK')" \
    && python3 -c "import cv2; [print(l) for l in cv2.getBuildInformation().splitlines() if any(k in l for k in ('CUDA','cuDNN','GStreamer','NEON','NVIDIA'))]"

# [4] PyTorch 2.11 cu130
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    torch==${TORCH_VERSION} torchvision \
    --index-url https://download.pytorch.org/whl/cu130

# [5] Build tools
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system hatchling setuptools wheel scikit-build-core yapf

# [6] Open3D CUDA (from source, Clang)
RUN git clone --depth 1 --branch ${OPEN3D_BRANCH} --recursive https://github.com/isl-org/Open3D.git /opt/open3d && \
    git clone --depth 1 --branch main https://github.com/isl-org/Open3D-ML.git /opt/open3d/Open3D-ML

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

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
      -DPYTHON_EXECUTABLE=/usr/bin/python${PYTHON_VERSION} \
    && make -j$(nproc) \
    && LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" make install-pip-package -j$(nproc) \
    && cp /opt/open3d/build/lib/python_package/pip_package/open3d-*.whl /opt/wheels/ \
    && echo "Open3D wheel saved: $(ls /opt/wheels/open3d-*.whl)" \
    && ldconfig \
    && rm -rf /opt/open3d

RUN rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN python3 -c "import importlib.util; assert importlib.util.find_spec('open3d') is not None, 'open3d not installed'"

# [7] cap-x deps + CUDA extensions (CuRobo, PointNet2)
COPY docker/patch-pyproject.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/patch-pyproject.sh

WORKDIR /tmp/capx-install
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

RUN rm -rf /tmp/capx-install

# [8] LIBERO evaluation venv (optional)
ARG WITH_LIBERO=0
RUN if [ "${WITH_LIBERO}" = "1" ]; then \
      mkdir -p /tmp/libero-install; \
    fi

WORKDIR /tmp/libero-install
COPY pyproject.toml .
COPY capx/third_party/sam3 capx/third_party/sam3
COPY capx/third_party/contact_graspnet_pytorch capx/third_party/contact_graspnet_pytorch
COPY capx/third_party/curobo capx/third_party/curobo
COPY capx/third_party/libero_dependencies capx/third_party/libero_dependencies
COPY capx/third_party/LIBERO-PRO capx/third_party/LIBERO-PRO

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.ccache \
    if [ "${WITH_LIBERO}" = "1" ]; then \
      mkdir -p capx && touch capx/__init__.py && \
      uv venv /opt/venv-libero --python python${PYTHON_VERSION} --system-site-packages && \
      patch-pyproject.sh pyproject.toml && \
      uv pip install --python /opt/venv-libero/bin/python \
        --no-build-isolation \
        ".[libero,contactgraspnet]" || \
      echo "WARNING: LIBERO venv install had errors (may need runtime install)"; \
    else \
      echo "Skipping LIBERO venv (WITH_LIBERO=0)"; \
    fi

RUN rm -rf /tmp/libero-install

# [9] JAX with CUDA 13 (built from source for SM 110, no PTX JIT)
RUN --mount=type=cache,target=/root/.cache/bazel \
    git clone --branch "jax-v${JAX_VERSION}" --depth=1 --recursive https://github.com/google/jax /opt/jax && \
    cd /opt/jax && \
    mkdir -p /opt/jax/wheels && \
    python3 build/build.py build \
      --clang_path=/usr/bin/clang \
      --output_path=/opt/jax/wheels/ \
      --cuda_compute_capabilities="sm_110" \
      --cuda_version=13.0.0 \
      --cudnn_version=9.12.0 \
      --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt && \
    uv pip install --system \
      /opt/jax/wheels/jaxlib-*.whl \
      /opt/jax/wheels/jax_cuda13_plugin-*.whl \
      /opt/jax/wheels/jax_cuda13_pjrt-*.whl \
      opt_einsum && \
    uv pip install --system --no-deps jax==${JAX_VERSION} && \
    cp /opt/jax/wheels/*.whl /opt/wheels/ && \
    rm -rf /opt/jax

RUN --mount=type=cache,target=/root/.cache/uv \
    for whl in /opt/wheels/*selfbuilt*.whl; do \
      [ -f "$whl" ] || continue; \
      uv tool run change_wheel_version "$whl" --version ${JAX_VERSION} --delete-old-wheel; \
    done

# [10] ROS 2 Jazzy client
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu noble main" > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-rclpy \
      ros-jazzy-action-msgs \
      ros-jazzy-nav-msgs \
      ros-jazzy-trajectory-msgs \
      ros-jazzy-geometry-msgs \
      ros-jazzy-sensor-msgs \
      ros-jazzy-std-msgs \
    && rm -rf /var/lib/apt/lists/* \
    && echo '. /opt/ros/jazzy/setup.bash' >> /root/.bashrc

ENV ROS_DISTRO=jazzy

ARG CLEAN_CACHES=0
RUN if [ "${CLEAN_CACHES}" = "1" ]; then \
      echo "Cleaning build caches..." && \
      rm -rf /root/.cache/bazel /root/.cache/uv /root/.cache/pip /root/.ccache /tmp/* && \
      echo "Caches cleaned."; \
    fi

WORKDIR /workspace

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8110 8113 8114 8115 8116 8117 8118

ENTRYPOINT ["/entrypoint.sh"]
