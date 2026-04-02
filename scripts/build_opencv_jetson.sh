#!/usr/bin/env bash
# Build OpenCV 4.13.0 with CUDA 13.0 + cuDNN 9 for Jetson Thor (sm_110, aarch64)
# and install the Python bindings into the cap-x venv.
#
# Usage (from repo root):
#   bash scripts/build_opencv_jetson.sh

set -euo pipefail

OPENCV_VERSION="4.13.0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
VENV_PYTHON="${VENV}/bin/python"
VENV_SITE="$("${VENV_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
NUMPY_INCLUDE="$("${VENV_PYTHON}" -c 'import numpy; print(numpy.get_include())')"
BUILD_DIR="${REPO_ROOT}/build/opencv"
SRC_DIR="${BUILD_DIR}/src"
CMAKE_BUILD="${BUILD_DIR}/cmake_build"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# Jetson Thor GPU compute capability (hardware arch, not toolkit version)
CUDA_ARCH="11.0"

echo "=== OpenCV ${OPENCV_VERSION} Jetson Thor build ==="
echo "  CUDA toolkit: ${CUDA_HOME} (v13.0)"
echo "  GPU arch:     sm_110 (CC ${CUDA_ARCH})"
echo "  Venv:         ${VENV}"
echo "  Site-pkgs:    ${VENV_SITE}"
echo "  Build dir:    ${BUILD_DIR}"

# ── 1. System dependencies ──────────────────────────────────────────────────
echo ""
echo "── [1/5] Checking system dependencies..."

MISSING_PKGS=()
for pkg in \
    libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13 \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libeigen3-dev libv4l-dev \
    cmake ninja-build; do
    dpkg -s "$pkg" &>/dev/null || MISSING_PKGS+=("$pkg")
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    echo "  Installing: ${MISSING_PKGS[*]}"
    sudo apt-get install -y "${MISSING_PKGS[@]}"
fi

# cuDNN headers: on aarch64 Ubuntu, apt installs to /usr/include/aarch64-linux-gnu/
CUDNN_INCLUDE_DIR="$(dpkg -L libcudnn9-headers-cuda-13 2>/dev/null | grep 'cudnn\.h$' | head -1 | xargs dirname)"
if [[ -z "${CUDNN_INCLUDE_DIR}" || ! -f "${CUDNN_INCLUDE_DIR}/cudnn.h" ]]; then
    echo "ERROR: cudnn.h not found after installing libcudnn9-headers-cuda-13"
    exit 1
fi
echo "  cuDNN headers: ${CUDNN_INCLUDE_DIR}"

# ── 2. Source download ──────────────────────────────────────────────────────
echo ""
echo "── [2/5] Fetching OpenCV ${OPENCV_VERSION} sources..."

mkdir -p "${SRC_DIR}"

clone_or_update() {
    local url="$1" dest="$2" tag="$3"
    if [[ -d "${dest}/.git" ]]; then
        echo "  ${dest##*/} already cloned, checking out ${tag}"
        git -C "${dest}" fetch --tags -q
        git -C "${dest}" checkout "${tag}" -q
    else
        echo "  Cloning ${url}"
        git clone --depth 1 --branch "${tag}" "${url}" "${dest}"
    fi
}

clone_or_update \
    "https://github.com/opencv/opencv.git" \
    "${SRC_DIR}/opencv" \
    "${OPENCV_VERSION}"

clone_or_update \
    "https://github.com/opencv/opencv_contrib.git" \
    "${SRC_DIR}/opencv_contrib" \
    "${OPENCV_VERSION}"

# ── 3. CMake configure ──────────────────────────────────────────────────────
echo ""
echo "── [3/5] CMake configure..."

mkdir -p "${CMAKE_BUILD}"

PYTHON3_LIB="$(find /usr/lib/aarch64-linux-gnu -name 'libpython3.10*.so*' | head -1)"
PYTHON3_INCLUDE="$("${VENV_PYTHON}" -c 'from sysconfig import get_paths; print(get_paths()["include"])')"

cmake -S "${SRC_DIR}/opencv" \
      -B "${CMAKE_BUILD}" \
      -G Ninja \
      \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${CMAKE_BUILD}/install" \
      \
      -DOPENCV_EXTRA_MODULES_PATH="${SRC_DIR}/opencv_contrib/modules" \
      \
      -DWITH_CUDA=ON \
      -DCUDA_ARCH_BIN="${CUDA_ARCH}" \
      -DCUDA_ARCH_PTX="" \
      -DCUDA_HOME="${CUDA_HOME}" \
      -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
      \
      -DWITH_CUDNN=ON \
      -DOPENCV_DNN_CUDA=ON \
      -DCUDNN_INCLUDE_DIR="${CUDNN_INCLUDE_DIR}" \
      \
      -DWITH_NVCUVID=OFF \
      -DWITH_NVCUVENC=OFF \
      \
      -DWITH_GSTREAMER=ON \
      -DWITH_FFMPEG=ON \
      -DWITH_V4L=ON \
      -DWITH_LIBV4L=ON \
      \
      -DWITH_OPENGL=OFF \
      -DWITH_QT=OFF \
      -DBUILD_opencv_python2=OFF \
      -DBUILD_opencv_python3=ON \
      -DPYTHON3_EXECUTABLE="${VENV_PYTHON}" \
      -DPYTHON3_INCLUDE_DIR="${PYTHON3_INCLUDE}" \
      -DPYTHON3_LIBRARY="${PYTHON3_LIB}" \
      -DPYTHON3_NUMPY_INCLUDE_DIRS="${NUMPY_INCLUDE}" \
      -DPYTHON3_PACKAGES_PATH="${VENV_SITE}" \
      \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_DOCS=OFF \
      -DBUILD_opencv_java=OFF \
      -DBUILD_opencv_js=OFF \
      \
      -DENABLE_NEON=ON \
      -DCPU_BASELINE="NEON" \
      2>&1 | tee "${BUILD_DIR}/cmake_configure.log"

echo ""
echo "── CMake CUDA/cuDNN/GStreamer summary:"
grep -E "CUDA|cuDNN|GStreamer|Python 3" "${BUILD_DIR}/cmake_configure.log" \
    | grep -E "YES|NO|version|path" | sort -u | head -30

# ── 4. Build ─────────────────────────────────────────────────────────────────
echo ""
echo "── [4/5] Building with $(nproc) cores (~30-60 min on Jetson)..."
cmake --build "${CMAKE_BUILD}" --parallel "$(nproc)" 2>&1 | tee "${BUILD_DIR}/build.log"

# ── 5. Install into venv ─────────────────────────────────────────────────────
echo ""
echo "── [5/5] Installing Python bindings into venv..."

# Remove the CPU-only PyPI wheel
"${VENV}/bin/pip" uninstall -y opencv-python-headless 2>/dev/null || true

# cmake install copies cv2.*.so + stubs to PYTHON3_PACKAGES_PATH
cmake --install "${CMAKE_BUILD}" --component python

echo ""
echo "=== Build complete. Verifying ==="
"${VENV_PYTHON}" -c "
import cv2
print('OpenCV version:', cv2.__version__)
for line in cv2.getBuildInformation().splitlines():
    if any(k in line for k in ('CUDA', 'cuDNN', 'GStreamer', 'NEON')):
        print(line)
print('CUDA-enabled devices:', cv2.cuda.getCudaEnabledDeviceCount())
"
