<p align="center">
  <img src="./docker/bird.png" alt="cap-y" width="400">
</p>

# cap-y (Agentic Robot Manipulation)


[Get Started](#get-started) · [What's Inside](#whats-inside) · [Wheels](#pre-built-wheels) · [Containers](#container-family) · [vs CaP-X](#vs-cap-x-upstream)

[Docs](#docs) · [LinkedIn](https://linkedin.com/in/sevapru) · [wheels.sobaka.dev](https://wheels.sobaka.dev)

**CUDA-optimized Docker runtime for robot control on NVIDIA Jetson Thor.** by [sobaka.dev](https://sobaka.dev) 🐕

Dedicated environemnt for Agentic humanoid robot operation designed to provide best performance and acceleration available.

Fork of [CaP-X](https://github.com/capgym/cap-x). Everything that can run on GPU -- runs on GPU. Pull the container, plug in your robot, go.

### One-line Install

```bash
curl -sSL https://raw.githubusercontent.com/sevapru/cap-y/main/scripts/install.sh | bash
```

## What's Inside

| Module | Version | CUDA | Notes | On PyPI for aarch64? | cap-y |
|--------|---------|------|-------|:---:|:---:|
| OpenCV | 4.13 | cuDNN 9.12, cuBLAS, FAST_MATH | GStreamer, NEON FP16/BF16 🏃 | no CUDA wheel | ✅ |
| Open3D | 0.19+ | CUDA tensors, PyTorch ops | RealSense D455, Open3D-ML, GUI | no wheel at all | ✅ |
| JAX | 0.9.2 | SM 110 native kernels | Built from source, no PTX JIT  | no SM 110 kernels | ✅ |
| CuRobo | 0.7 | 5 CUDA extensions | Collision-free trajectories | needs CUDA + torch | ✅ |
| ContactGraspNet | - | PointNet2 CUDA ops | 6-DOF grasps from depth | needs CUDA build | ✅ |
| PyTorch | 2.11 cu130 | FlashAttention-4, sm_110 | aarch64 native | ✅ | ✅ |
| MuJoCo | 3.6 | EGL headless | LIBERO evaluation | ✅ | ✅ |
| ROS 2 | Jazzy | - | rclpy + msg types  | ✅ | ✅ |

**5 out of 8 modules have no pre-built aarch64 CUDA packages anywhere.** All verified with `test_container_base.py` (9/9 PASS).


If you'd actually compare it with [jetson-containers](https://github.com/dusty-nv/jetson-containers/tree/master?tab=readme-ov-file)  - you'll understand how great this set is for 03.04.2026. 
There is, mainly, no on-time cu130 support for containers and, as I would tell you: not enough developers who can provide this builds. 


## Get Started

## Pre-built Wheels

Index: [wheels.sobaka.dev](https://wheels.sobaka.dev)

Pre-built wheels for Jetson Platform
Tested on **Jetson Thor aarch64, Python 3.12, CUDA 13.0, SM 110**.

```bash
uv pip install --extra-index-url https://wheels.sobaka.dev/ \
  jaxlib jax-cuda13-plugin jax-cuda13-pjrt
```

### Pull & Run

All three containers are available from GHCR:

| Image | Pull command |
|-------|-------------|
| `cap-y` | `docker pull ghcr.io/sevapru/cap-y:latest` |
| `cap-y-open` | `docker pull ghcr.io/sevapru/cap-y-open:latest` |
| `cap-y-nvidia` | `docker pull ghcr.io/sevapru/cap-y-nvidia:latest` |

```bash
# Base container — perception + manipulation
docker pull ghcr.io/sevapru/cap-y:latest
docker run -it --rm --runtime nvidia --entrypoint bash -v .:/workspace ghcr.io/sevapru/cap-y:latest

# Open container — add ROS 2 Jazzy, Nav2, MoveIt 2, nvblox, LiveKit
docker pull ghcr.io/sevapru/cap-y-open:latest
docker run -it --rm --runtime nvidia --entrypoint bash -v .:/workspace ghcr.io/sevapru/cap-y-open:latest

# NVIDIA container — add Isaac ROS cuMotion, cuVSLAM, ScheduleStream
docker pull ghcr.io/sevapru/cap-y-nvidia:latest
docker run -it --rm --runtime nvidia --entrypoint bash -v .:/workspace ghcr.io/sevapru/cap-y-nvidia:latest
```

Verify after pull (runs base + layer-specific checks):
```bash
cd cap-y/docker

docker compose -f docker-compose.capx.yml --profile test run --rm capx-test
# --profile test-open run --rm capx-test-open
# --profile test-nvidia run --rm capx-test-nvidia
```

### Build from Source (like... 3 hours?)

> ⚡ Tested on **Jetson AGX Thor** (sm_110, CUDA 13.0, 128 GB unified memory)

```bash
git clone --recurse-submodules https://github.com/sevapru/cap-y && cd cap-y/docker

# Auto-detect GPU and build (nvidia-smi detects sm_110 on Thor)
./build.sh # Or specify GPU architecture manually with --arch 11.0 or --arch 8.9 (for RTX 4080)
```
Subsequent builds use ccache (~10 min)
Build takes ~2-3 hours first time (OpenCV CUDA + Open3D CUDA + JAX from source)

Check log for errors: `/tmp/cap-y-build.log`
```bash
grep -i "error:" /tmp/cap-y-build.log | grep -v warning
```

### Build with Docker Compose

```bash
cd cap-y/docker
docker compose -f docker-compose.capx.yml up -d --build
```

Build args:

| Arg | Effect | Example |
|-----|--------|---------|
| `CUDA_ARCH_BIN` | GPU compute capability | `--build-arg CUDA_ARCH_BIN=8.9` (RTX 4080) |
| `WITH_LIBERO=1` | Add LIBERO evaluation venv (robosuite/MuJoCo) | `--build-arg WITH_LIBERO=1` |
| `CLEAN_CACHES=1` | Remove bazel/uv/ccache for smaller image | `--build-arg CLEAN_CACHES=1` |
| `--all` (build.sh) | Build full chain: cap-y → cap-y-open → cap-y-nvidia | `./build.sh --all` |

```bash
# Publishing: full image with LIBERO + clean caches
docker compose -f docker-compose.capx.yml build \
  --build-arg WITH_LIBERO=1 --build-arg CLEAN_CACHES=1
```

### Start Perception Servers

```bash
cd cap-y/docker
docker compose -f docker-compose.capx.yml up -d                         # default: SAM3 + GraspNet + PyRoKi
CAPX_PROFILE=full docker compose -f docker-compose.capx.yml up -d       # + OWL-ViT + SAM2
CAPX_PROFILE=minimal docker compose -f docker-compose.capx.yml up -d    # PyRoKi only (IK go brrrr 🏎️)
```

Interactive shell:
```bash
docker run -it --rm --runtime nvidia --entrypoint bash -v .:/workspace cap-y
```

Or add to `~/.bashrc` for one-word access:
```bash
alias cap-y='docker run -it --rm --runtime nvidia --entrypoint bash -v .:/workspace cap-y'
# then just: cap-y
```

### Verify Everything Works

```bash
cd cap-y/docker
docker compose -f docker-compose.capx.yml --profile test run --rm capx-test  # Expected: 9/9
```

## Pre-built Wheels

Index: [wheels.sobaka.dev](https://wheels.sobaka.dev)

Pre-built for **Jetson Thor aarch64, Python 3.12, CUDA 13.0, SM 110**. None of these exist on PyPI 🎁

```bash
uv pip install --extra-index-url https://wheels.sobaka.dev/simple/ \
  jaxlib jax-cuda13-plugin jax-cuda13-pjrt
```

| Package | Build time saved | What's special |
|---------|-----------------|----------------|
| jaxlib 0.9.2 | ~2 hours | SM 110 native kernels, no PTX JIT |
| jax-cuda13-plugin | (same build) | SM 110 pre-compiled XLA |
| jax-cuda13-pjrt | (same build) | CUDA 13 runtime |

OpenCV CUDA and Open3D CUDA are available via the `cap-y` container (cmake/source builds don't produce portable wheels).
 


## Container Family

Split by license -- pick what you need:


| Container      | Adds                                                     | License        | Vibe                                    |
| -------------- | -------------------------------------------------------- | -------------- | --------------------------------------- |
| `cap-y`        | OpenCV, Open3D, PyTorch, JAX, CuRobo, perception servers | Mixed          | "I see things and grab them" 👁️🤖      |
| `cap-y-open`   | + ROS 2, Nav2, MoveIt 2, Drake, LiveKit                  | Apache/BSD/MIT | "I plan and navigate, commercially" 🗺️ |
| `cap-y-nvidia` | + Isaac ROS cuMotion, cuVSLAM, NITROS                    | NVIDIA license | "I have enterprise friends" 🏢          |

<details>
<summary><b>cap-y-open</b> — verified components</summary>

| Component | Version / Count | Status | Notes |
|-----------|----------------|--------|-------|
| ROS 2 Jazzy | `ros-base` | Verified | `rclpy` import, `ros2` CLI |
| Nav2 | 34 packages | Verified | Full navigation stack via `nav2-bringup` |
| MoveIt 2 | 27 packages | Verified | Motion planning, `moveit_core`, `moveit_ros_planning` |
| ros2_control | 6 packages | Verified | `controller_manager` + controllers |
| CycloneDDS | `rmw_cyclonedds_cpp` | Verified | `RMW_IMPLEMENTATION` set, package installed |
| nvblox | C++ core lib | Verified | `libnvblox` + headers at `/usr/local/` (Apache 2.0, NOT Isaac ROS wrapper) |
| LiveKit | SDK + agents | Verified | `livekit`, `livekit-agents`, `livekit-plugins-silero` |
| Drake / GCS | — | Skipped on aarch64 | No aarch64 wheel; available on x86_64 |
| Bimanual readiness | MoveIt dual-arm | Verified | `moveit_core` + dual-arm URDF loading via `robot_descriptions` |

</details>

<details>
<summary><b>cap-y-nvidia</b> — verified components</summary>

| Component | Status | Notes |
|-----------|--------|-------|
| Isaac ROS apt repo | Configured | `release-4.3` for Noble at `isaac.download.nvidia.com` |
| Isaac ROS common + NITROS | Pending | Jazzy packages not yet in NVIDIA repo (graceful skip) |
| cuMotion | Source cloned | Colcon build skipped — needs Isaac ROS NITROS deps |
| cuVSLAM | Built | Source cloned + `colcon build` installed |
| nvblox Isaac ROS wrapper | Built | Source cloned + `colcon build` installed |
| cuTAMP | Pending | Not on PyPI; install from NVIDIA NGC when available |
| ScheduleStream | Pending | NVlabs/ScheduleStream cloned; editable install needs cuRobo runtime |
| Isaac ROS workspace | 3 src / 5 built | `/opt/isaac_ros_ws/` |

**Bimanual stack** (from [Vorndamme et al.](https://schedulestream.github.io/) insights):

```
nvblox (scene SDF)  →  cuMotion (per-arm MoveIt)  →  ScheduleStream (bimanual scheduling)
     ↓                                                        ↓
 obstacle avoidance                                  dual-arm coordination
                                                     (cooperative task-space)
```

cuMotion plans each arm independently through MoveIt 2. ScheduleStream adds the missing bimanual coordination layer — temporal scheduling with GPU-accelerated samplers that produces parallel dual-arm motion instead of sequential single-arm plans. CycloneDDS provides the DDS transport for joint velocity commands at 1 kHz to the G1's native impedance controller.

</details>

```bash
docker build -f docker/Dockerfile.open -t cap-y-open:latest ..
docker build -f docker/Dockerfile.nvidia -t cap-y-nvidia:latest ..
```



## Ports


| Port | Service                | When      |
| ---- | ---------------------- | --------- |
| 8110 | LLM proxy (OpenRouter) | always    |
| 8113 | SAM2                   | `full`    |
| 8114 | SAM3                   | `default` |
| 8115 | ContactGraspNet        | `default` |
| 8116 | PyRoKi IK              | `default` |
| 8117 | OWL-ViT                | `full`    |
| 8118 | CuRobo                 | custom    |




## vs CaP-X (upstream)


|                             | CaP-X                        | cap-y                               |
| --------------------------- | ---------------------------- | ----------------------------------- |
| Install                     | `uv sync` + CUDA (maybe)     | `docker pull` ☁️                    |
| OpenCV                      | CPU                          | ✅ CUDA (cuDNN, cuBLAS, FAST_MATH)     |
| Open3D                      | ❌ no aarch64 wheel           | ✅ CUDA + PyTorch ops + RealSense    |
| JAX                         | CPU (SM 110 kernels missing) | ✅ SM 110 native (built from source) |
| Time to first robot command | hours                        | minutes                             |




## Docs


| File                                                                   | What                                  |
| ---------------------------------------------------------------------- | ------------------------------------- |
| [docker/OPTIMISATIONS.md](docker/OPTIMISATIONS.md)                     | Optimisations relevant to Jetson Thor |
| [.claude/CLAUDE.md](.claude/CLAUDE.md)                                 | Agent/dev docs                        |
| [scripts/test_cuda_acceleration.py](scripts/test_cuda_acceleration.py) | CUDA test suite                       |


For upstream CaP-X docs (environments, APIs, RL training): [github.com/capgym/cap-x](https://github.com/capgym/cap-x)





## Citation

If cap-y saved you from compiling OpenCV at 3 AM, cite the original work that made it possible:

```bibtex
@article{fu2025capx,
  title     = {{CaP-X}: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation},
  author    = {Fu, Max and Yu, Justin and El-Refai, Karim and Kou, Ethan and Xue, Haoru and others},
  journal   = {arXiv preprint arXiv:2603.22435},
  year      = {2025}
}
```

## License

Upstream CaP-X: MIT. Docker additions: MIT. Individual packages: see container family table.

Hope you have a good day, Seva

Built with 🐾 by [sobaka.dev](https://sobaka.dev)