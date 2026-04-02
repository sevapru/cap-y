# cap-x -- AI Stack Architecture

## Overview

Robotics AI serving stack: perception (SAM3, SAM2, OWL-ViT), grasping (ContactGraspNet), motion planning (PyRoKi IK, CuRobo), and 3D processing (Open3D). Containerized for NVIDIA Jetson Thor with full CUDA acceleration. Portable to any NVIDIA GPU via `CUDA_ARCH_BIN` build arg.

---

## Docker Ecosystem

### Base Image

`nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04` -- NVIDIA's recommended container for Jetson Thor development ([setup guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html)). Multi-arch: pulls arm64 on Jetson, amd64 on x86 laptops. The `-devel` variant includes CUDA toolkit headers needed for building OpenCV and CuRobo CUDA extensions.

### Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Single-stage build: OpenCV CUDA + Open3D CUDA + PyTorch 2.11 + cap-x deps |
| `docker-compose.capx.yml` | Service definition: runtime nvidia, profiles, healthcheck, cache volumes |
| `build.sh` | Auto-detects GPU compute capability, passes to docker compose build |
| `entrypoint.sh` | Loads secrets from files, editable reinstall, starts servers |
| `.dockerignore` | Excludes secrets, build artifacts, large unused third_party dirs |

### Build

```bash
./build.sh 2>&1 | tee /tmp/open3d-build.log   # auto-detect GPU, log to file
./build.sh --arch 8.9 2>&1 | tee /tmp/open3d-build.log   # manual (RTX 4080)
```

### Build Log

All build output should go to `/tmp/open3d-build.log`. Before starting the container or debugging failures, always check the log:

```bash
# Check for errors (excluding warnings)
grep -i "error:" /tmp/open3d-build.log | grep -v warning | tail -20

# Check last 30 lines for final status
tail -30 /tmp/open3d-build.log

# Check OpenCV verification
grep -A5 "OpenCV.*CUDA build" /tmp/open3d-build.log

# Check Open3D verification
grep "Open3D package found" /tmp/open3d-build.log
```

### Run

```bash
CAPX_PROFILE=default docker compose -f docker-compose.capx.yml up -d   # 3 servers
CAPX_PROFILE=full docker compose -f docker-compose.capx.yml up -d      # 5 servers
CAPX_PROFILE=minimal docker compose -f docker-compose.capx.yml up -d   # PyRoKi only
```

### Server Profiles

| Profile | Servers | Ports |
|---------|---------|-------|
| `default` | SAM3, ContactGraspNet, PyRoKi | 8114, 8115, 8116 |
| `full` | + OWL-ViT, SAM2 | + 8117, 8113 |
| `minimal` | PyRoKi only | 8116 |

OpenRouter LLM proxy (port 8110) starts automatically if `.openrouterkey` exists.

### Portability

Same Dockerfile builds on Jetson Thor (aarch64) and laptops (x86_64). Only `CUDA_ARCH_BIN` changes:

| Device | CUDA_ARCH_BIN | Architecture |
|--------|--------------|--------------|
| Jetson Thor | 11.0 | aarch64, sm_110 |
| RTX 4080 | 8.9 | x86_64, sm_89 |
| RTX 3090 | 8.6 | x86_64, sm_86 |

### Volumes

| Volume | Purpose |
|--------|---------|
| `.:/workspace` | Live source code mount (editable install at startup) |
| `${HOME}/.cache/huggingface:/data/huggingface` | Model weights cache |
| `capx-uv-cache` | Python package download cache across rebuilds |
| `capx-ccache` | C++ compilation cache for OpenCV/Open3D/CUDA rebuilds |

### Secrets

Never baked into the image. Loaded at runtime from mounted workspace:
- `.openrouterkey` -- OpenRouter API key (LLM proxy on port 8110)
- `.huggingfacekey` -- HuggingFace token (model downloads)
- `.env` -- Docker Compose environment variables

All three are in `.dockerignore`.

---

## Two Python Environments

The container has two Python environments due to a robosuite version conflict:

| Environment | Location | Purpose |
|-------------|----------|---------|
| System Python | `/usr/bin/python3.12` | Serving APIs (SAM3, GraspNet, PyRoKi, OWL-ViT, SAM2) |
| LIBERO venv | `/opt/venv-libero` | LIBERO evaluation (robosuite 1.4 fork) |

The LIBERO venv uses `--system-site-packages` so it inherits CUDA OpenCV, PyTorch, and Open3D from the system install.

---

## Source-Built Libraries

Three libraries are built from source in the Dockerfile (no pre-built aarch64 wheels):

1. **OpenCV 4.13** -- CUDA DNN, cuBLAS, cuDNN, GStreamer, FFMPEG, NEON
2. **Open3D 0.19+** -- CUDA, GUI, PyTorch ops, RealSense, Open3D-ML, BLAS
3. **CuRobo** -- CUDA motion planning extensions (built via `uv pip install`)

See `OPTIMISATIONS.md` for detailed flag explanations.

### OpenCV CUDA Shadow Prevention

After building OpenCV from source, a dummy `dist-info` is created so `uv pip install` doesn't overwrite the CUDA build with a CPU-only PyPI wheel. This is critical -- without it, the entire OpenCV CUDA build becomes dead code.

### Open3D Jetson Thor Patches

- **stdgpu fix**: v0.19.0 fails with CUDA 13.0 ([#7376](https://github.com/isl-org/Open3D/issues/7376)). Built from `main` branch which includes the fix.
- **FindPytorch.cmake patch**: Open3D's arch conversion (`110` -> `1.1+PTX0`) breaks for compute capability >= 10. Patched via `sed` to read `TORCH_CUDA_ARCH_LIST` from env instead.
- **Clang compiler**: Filament (GUI rendering engine) requires Clang >= 7.
- **System curl**: Open3D's bundled BoringSSL conflicts with system OpenSSL. Fixed with `-DUSE_SYSTEM_CURL=ON`.

### FP16 Half-Cast Patches

OpenCV DNN CUDA has implicit `double`-to-`half` comparisons that cause incorrect results with FP16 inference. Patched from [jetson-containers](https://github.com/dusty-nv/jetson-containers/blob/master/packages/cv/opencv/build.sh):
```cpp
// Before (buggy): weight != 1.0  (double vs half)
// After (fixed):  (float)weight != 1.0f
```

---

## JAX / PyRoKi

PyRoKi uses JAX for IK solving. cap-x pins `jax<0.4.30` in `pyproject.toml` override-dependencies (compatibility with jaxlie/jaxls). JAX 0.4.29 has no CUDA 13.0 aarch64 wheels, so PyRoKi runs on CPU. This is acceptable: `SERVER_REGISTRY["pyroki"]["gpu_required"] = False`.

**Future upgrade path**: update jaxlie/jaxls/pyroki to support JAX 0.8+, then install `jax[cuda13]`.

---

## Simulation (Separate)

MuJoCo/robosuite/LIBERO simulation is NOT in the serving container's primary workflow. For headless evaluation inside the container:

```bash
docker exec capx-serving bash -c '
  source /opt/venv-libero/bin/activate &&
  MUJOCO_GL=egl bash scripts/test_libero_privileged.sh'
```

For GUI visualization: mount X11 socket with `DISPLAY=$DISPLAY` and `-v /tmp/.X11-unix:/tmp/.X11-unix`.

---

## Tooling

### Package Management
- **`uv`** (latest, installed via `curl -LsSf https://astral.sh/uv/install.sh | sh`): fast Python package manager. Use instead of pip/venv. Supports `--find-links` for local wheels, `--override` for dependency resolution, `--no-build-isolation` for CUDA extension builds.
- **`--find-links /opt/wheels`**: standard uv/pip mechanism for prioritising locally-built packages (Open3D CUDA wheel). Preferred over `--override` hacks.

### Build Tools
- **`ccache`**: C++ compilation cache. Persisted via Docker named volume `capx-ccache` AND BuildKit `--mount=type=cache,target=/root/.ccache`. Reduces OpenCV/Open3D rebuilds from ~60min to ~5min on cache hit. **Use it**: add `-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache` to any cmake build.
- **`clang` 18**: used for Open3D build (required by Filament rendering engine). Often produces faster ARM64 code than GCC.
- **`gcc` 13**: used for OpenCV build and system packages.
- **`ninja`**: fast build system for OpenCV cmake (faster than Make for incremental builds).
- **BuildKit cache mounts** (`--mount=type=cache,target=...`): persist caches across Docker layer invalidations. Added to every heavy RUN step. **Always use them** when adding new source builds to the Dockerfile.

### Docker
- **`docker compose` v2** (not docker-compose v1)
- **`runtime: nvidia`**: exposes GPU to container via NVIDIA Container Toolkit
- **`init: true`**: proper signal forwarding and zombie reaping (tini as PID 1)
- **`build.sh`**: auto-detects GPU compute capability via `nvidia-smi`, passes to `docker compose build`

### Debugging
- **Build log**: always `./build.sh 2>&1 | tee /tmp/open3d-build.log`. Check with `grep -i "error:" /tmp/open3d-build.log | grep -v warning`
- **`docker exec`**: run commands inside running container (`docker exec capx-serving bash -c '...'`)
- **Healthcheck**: SAM3 at `:8114/health`, auto-probed every 30s by docker compose

### Recommendations for Agents
- **Use `uv` over `pip`** everywhere: faster resolution, better error messages, supports overrides and find-links natively.
- **Use `ccache`** for any C/C++/CUDA compilation: add `CMAKE_C_COMPILER_LAUNCHER=ccache` and `CMAKE_CXX_COMPILER_LAUNCHER=ccache` to cmake. The cache volume persists across builds.
- **Use `--mount=type=cache`** in Dockerfile RUN steps that download or compile anything: `target=/root/.cache/uv` for Python packages, `target=/root/.ccache` for C++ objects.
- **Use `--find-links /opt/wheels`** when installing packages that have locally-built wheels (Open3D). This is the standard pip/uv mechanism -- preferred over `--override` or dummy dist-info.
- **Use `--no-build-isolation`** only when a package needs `torch` or other heavy dependencies at build time (CuRobo, PointNet2). Pre-install build backends (`hatchling`, `setuptools`, `wheel`) before using it.
- **Check build logs** before claiming success: `grep -i "error:"` in the log file. Parallel builds hide errors in interleaved output.
- **Use `importlib.util.find_spec`** instead of full `import` for build-time package verification when GPU is unavailable (Docker build has no GPU access).
- **Pipe to `tee`** all long-running commands to capture output for debugging: `command 2>&1 | tee /tmp/logfile.log`

---

## Known Issues

- `opencv_rgbd` module disabled: uses legacy `glRenderbufferStorageEXT` calls incompatible with GLVND on aarch64. Use Open3D for 3D reconstruction (TSDF, ICP, KinFu equivalents).
- Open3D WebRTC not supported on ARM Linux. LiveKit handles WebRTC transport independently.
- Open3D headless rendering conflicts with GUI mode. Use `o3d.visualization.rendering.OffscreenRenderer` API instead of the cmake flag.
- `UV_SYSTEM_TARGET` is not a real uv env var. `--system` flag is passed explicitly to all `uv pip install` calls.
- Port 8117: shared between OWL-ViT (in `full` profile) and CuRobo (in registry). Don't run both simultaneously.

---

## Observations

Agents: append findings here with date and branch name.

### 2026-04-02 docker-capx
- Open3D v0.19.0 cannot compile with CUDA 13.0 on Jetson Thor -- must use `main` branch (stdgpu fix)
- Open3D's `FindPytorch.cmake` `translate_arch_string` macro fails for 3-digit compute capabilities (110 -> 1.1+PTX0). Patched via sed.
- Filament requires Clang; cannot build with GCC. Added `clang libc++-dev libc++abi-dev` to Open3D deps only.
- Open3D bundled curl uses BoringSSL which conflicts with system OpenSSL. Fixed with `-DUSE_SYSTEM_CURL=ON`.
- `libglu1-mesa-dev` required for GLEW's `GL/glu.h` in Open3D build.
- `yapf` Python package required by Open3D's `generate_torch_ops_wrapper.py` during pip package generation.
- `file` command required by Filament's `combine-static-libs.sh`.
- PyTorch 2.11.0 cu130 wheels available for aarch64 on PyPI since March 23, 2026. jetson-containers only supports up to 2.10.
- Ubuntu 24.04 Python 3.12 has PEP 668 `EXTERNALLY-MANAGED` marker -- must remove for system-wide pip installs in Docker.
- NVIDIA PyPI mirror (`pypi.nvidia.com`) can be slow; `UV_HTTP_TIMEOUT=300` prevents download timeouts.
