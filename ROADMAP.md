# cap-y Roadmap

Task list for delegation to subagents or contributors. Each item is scoped to be executable independently.

---

## CI/CD Pipeline

Build all four container images on a self-hosted Jetson Thor runner, push to GHCR, and run the test cascade on every push to `main`.

- Set up GitHub Actions self-hosted runner on Jetson Thor (`runs-on: [self-hosted, jetson-thor]`)
- Workflow: `build.sh --all` → push to `ghcr.io/sevapru/cap-y:{base,default,open,nvidia}` → run `capx-test`, `capx-test-default`, `capx-test-open`, `capx-test-nvidia` compose profiles
- Cache: mount ccache and uv-cache as named Docker volumes between runs
- Gate: fail PR merge if any test profile returns non-zero

**Subagent inputs needed:** GitHub repo access, runner registration token, GHCR push credentials.

---

## Multi-Arch Build Matrix ✓ DONE

Unified Dockerfiles with `TARGETARCH` multi-stage builds. `docker-bake.hcl` handles the dependency graph. Images are tagged `sevapru/cap-y:{variant}-{amd64,arm64}` and joined via `docker buildx imagetools create`. See "Development Notes" section below for details.

- amd64 (x86_64): validated on RTX 4080, `CUDA_ARCH_BIN=8.9`
- arm64 (aarch64): validated on Jetson Thor, `CUDA_ARCH_BIN=11.0`
- `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"` covers all current + future GPUs via PTX forward compat

---

## Wheel Index Expansion

Add Open3D CUDA and OpenCV CUDA wheels to [wheels.sobaka.dev](https://wheels.sobaka.dev). Currently only JAX wheels are published.

- Extract Open3D `.whl` from the build at `/opt/wheels/open3d-*.whl` and upload to the index
- OpenCV: cmake source builds don't produce portable wheels directly; investigate packaging via `delocate` or shipping as a fat binary wheel with bundled `.so` files
- Update Cloudflare Worker at `wheels-worker/` to serve new packages
- Add install instructions to README

**Subagent inputs needed:** R2 bucket write access, existing `wrangler.toml` config.

---

## Image Size Optimisation

Reduce the 21.6 GB base image size without sacrificing CUDA functionality.

- Audit layer sizes with `docker history cap-y:base --no-trunc`
- Evaluate multi-stage build: separate build stage (compilers, headers, bazel) from runtime stage (only `.so`, Python packages, wheels)
- Test `CLEAN_CACHES=1` build arg impact on final size
- Remove `/opt/opencv-build`, `/opt/open3d`, `/opt/jax` source trees after install (already done; verify nothing was missed)
- Investigate stripping debug symbols from CUDA kernels with `strip --strip-debug`
- Target: under 15 GB for base image

---

## LIBERO Evaluation Venv Verification

Verify the optional LIBERO venv (`WITH_LIBERO=1`) runs correctly on Jetson Thor.

- Build with `--build-arg WITH_LIBERO=1`
- Run `test_container_base.py` LIBERO check — it imports MuJoCo in the venv and tests EGL headless rendering
- Fix any MuJoCo 3.6 / robosuite dependency conflicts on aarch64 + Python 3.12
- Document required env vars (`MUJOCO_GL=egl`, `DISPLAY` not needed for headless)

---

## Isaac ROS Jazzy Package Tracking

Monitor NVIDIA's apt repo for `ros-jazzy-*` Isaac ROS packages becoming available and upgrade `Dockerfile.nvidia` accordingly.

- Poll `https://isaac.download.nvidia.com/apt/repos/ubuntu/noble/arm64/Packages` for new package listings
- When `ros-jazzy-isaac-ros-nitros` appears: replace the colcon-from-source build with an apt install
- Re-enable `colcon build` for cuMotion once NITROS is installable
- Run `test_container_nvidia.py` to validate the new build

**Subagent inputs needed:** Periodic polling task (cron or GitHub Actions schedule).

---

## Bimanual Stack End-to-End Test

Validate the full `nvblox → cuMotion → ScheduleStream` pipeline on the G1 URDF in simulation.

- Load G1 URDF in MuJoCo or Isaac Sim
- Run nvblox SDF generation on a static scene pointcloud
- Feed obstacle map to cuMotion for single-arm trajectory planning (left and right independently)
- Hand off plans to ScheduleStream for bimanual temporal scheduling
- Verify joint velocity commands reach the CycloneDDS topic at 1 kHz
- Document the full launch sequence in `docker/OPTIMISATIONS.md`

**Subagent inputs needed:** G1 URDF file, Isaac Sim license or MuJoCo scene setup.

---

## Commercial Profile

Create a `CAPX_PROFILE=commercial` server profile that runs only MIT/Apache/BSD-licensed servers (SAM3 + PyRoKi), excluding cuRobo and ContactGraspNet.

- Add `"commercial"` entry to `PROFILES` dict in `capx/serving/launch_servers.py`
- Document in README and NOTICE that this profile does not carry the NVIDIA NC License restriction
- Note: SAM3 Meta SAM License still applies (redistribution conditions, publication acknowledgment)

---

## CLI App (cap-x features)

Adapt cap-x capabilities into a `capx` CLI with the following commands,
wrapping the existing module entrypoints for discoverability:

- `capx envs list` — list registered environments (`capx.envs.list_envs()`)
- `capx envs run --config <yaml>` — start a trial (`capx/envs/launch.py`)
- `capx serve start [--profile default|open|full]` — launch all servers for profile
- `capx serve status` — hit gateway /status and pretty-print table
- `capx serve logs <server>` — tail logs for a named server
- `capx skills list` — list SkillLibrary entries
- `capx skills export --out <file>` — export skill library as Python
- `capx test [--mark imports|servers|api|simulation]` — run pytest with the right marks

Implementation: add `[project.scripts]` entry in `pyproject.toml`:
  `capx = "capx.cli.main:app"` — thin Tyro or Click app delegating to existing modules.

---

## NVIDIA Server Wrappers

Create FastAPI HTTP wrappers for Isaac ROS nodes so the gateway can proxy them:

- `capx/serving/launch_cumotion_server.py` (port 8122) — wraps cuMotion MoveIt plugin
  as HTTP: POST /plan with joint start + SE(3) goal → collision-free trajectory
- `capx/serving/launch_cuvslam_server.py` (port 8121) — wraps cuVSLAM node
  as HTTP: GET /pose → current estimated camera pose; POST /reset
- `capx/serving/launch_nvblox_server.py` (port 8123) — wraps nvblox SDF
  as HTTP: POST /query_sdf with 3D points → signed distances; GET /mesh → current mesh

Each wrapper subscribes to the ROS2 topic internally and exposes a stateless HTTP API.
Requires cap-y:nvidia image with ROS2 + Isaac ROS workspace.

---

## Development Notes

Accumulated findings, architecture decisions, and gotchas from building and deploying the Docker stack across aarch64 (Jetson Thor) and x86_64 (RTX 4080/4090, Blackwell).

### Unified Dockerfile architecture

A single set of Dockerfiles under `docker/` serves both architectures. `Dockerfile.base` uses Docker's built-in `TARGETARCH` (amd64/arm64) with multi-stage `FROM arch-${TARGETARCH}`:

- **amd64 stage**: pre-built GPU wheels (cudawarped OpenCV, `jax[cuda13]` pip, PyPI Open3D, PyTorch cu130). Build time ~5 min.
- **arm64 stage**: source compilation (OpenCV cmake, Open3D cmake+clang, JAX Bazel). Build time ~45 min.
- **final stage**: common packages (mink, rh56_controller, DemoGrasp, Newton, pymodbus, unitree_sdk2_python, capx deps, ROS 2 minimal).

No `PLATFORM_ARCH` variable needed. `docker buildx bake -f docker/docker-bake.hcl all` handles everything.

### Bake dependency graph

`docker-bake.hcl` declares image dependencies via `contexts = { "cap-y:base" = "target:cap-y-base" }` — bake resolves the full graph and builds in the correct order automatically. This replaces manual sequential `docker compose build` calls that failed with "image not found" race conditions.

Groups: `default` (base only), `all` (base + default + open + nvidia), `full` (all + dev).

### Per-image entrypoint behavior

| Image | ENTRYPOINT | CMD | Behavior |
|-------|-----------|-----|----------|
| `cap-y:base` | (NVIDIA default) | `["bash"]` | Interactive shell, no servers |
| `cap-y:default` | `/entrypoint.sh` | `[]` | Launches `CAPX_PROFILE=default` servers |
| `cap-y:open` | `/entrypoint.sh` | `[]` | Launches `CAPX_PROFILE=open` servers |
| `cap-y:nvidia` | `/entrypoint.sh` | `[]` | Launches `CAPX_PROFILE=nvidia` servers |
| `cap-y:dev` | `[]` | `["bash"]` | Interactive shell, no servers |

Server images write `/tmp/.capx-ready` after the gateway binds on port 8100. Compose healthchecks test this file.

### CUDA architecture strategy

`TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"` covers Turing through Blackwell desktop. `+PTX` on 12.0 provides forward compatibility — PTX JIT-compiles at runtime on any future SM (including SM 13.0 / GB300). Explicitly listing SM 13.0 causes `ValueError: Unknown CUDA arch` in PyTorch 2.11.0.

`CUDA_ARCH_BIN` (set via `build.sh` or `docker/.env`) targets cmake-based source builds (nvblox, cuRobo, ContactGraspNet) to a specific GPU. Auto-detected via `nvidia-smi --query-gpu=compute_cap`. Defaults: 8.9 for x86_64, 11.0 for aarch64. Exported to `ENV` in base so all child images inherit it.

### Jetson Thor compatibility

Jetson Thor (SM 110) runs the same `nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04` base as x86_64 — it is NOT an L4T image. This is intentional: Thor runs Ubuntu 24.04 natively (not JetPack/L4T). Builds for Thor also work on Orin devices when `CUDA_ARCH_BIN` is adjusted.

### OpenCV CUDA at build time

`libcuda.so.1` (the NVIDIA driver stub) is never present during `docker build`. OpenCV CUDA wheels import it at load time, so any `import cv2` validation in a `RUN` step will fail with `ImportError: libcuda.so.1: cannot open shared object file`. The CUDA build check is done in `entrypoint.sh` at container startup instead, with `|| true` so it doesn't abort if the container runs without a GPU.

### Shadow prevention for OpenCV

The cudawarped OpenCV wheel installs as `opencv_contrib_python_rolling`, but `pyproject.toml` depends on `opencv-python-headless`. Without intervention, `uv` installs the CPU-only PyPI wheel on top of the CUDA wheel. A fake `.dist-info` directory for `opencv-python-headless` is created in the Dockerfile to prevent this.

### Unitree hand control — two paths

- **RH56DFTP** (Ethernet Modbus TCP): `pymodbus>=3.12` is installed in `cap-y:base`. Register map: `ANGLE_SET=1486`, `ANGLE_ACT=1546`, `FORCE_SET=1498` (6x int16, 0-1000). No C++ compilation needed. See [ProsusAI/robot-teleop](https://github.com/ProsusAI/robot-teleop) for the Modbus TCP driver pattern.
- **RH56DFX** (DDS via dfx_inspire_service): requires `unitree_sdk2` C++ headers. Source is cloned to `/opt/dfx_inspire_service` in `cap-y:open` for reference but C++ colcon build is skipped (manual build instructions in the Dockerfile).

`unitree_sdk2_python` (BSD-3, Python-only DDS bindings) is installed in `cap-y:base` for programmatic robot control without C++ compilation.

### Docker content store corruption

The rootless Docker driver (`docker:rootless`) periodically loses blob references during parallel multi-image builds: `failed to get reader from content store: blob sha256:... not found`. Fix: `docker builder prune -f && sudo systemctl restart docker`. This is a known Docker/containerd issue, not a Dockerfile problem.

### uv hardlink warning in Docker

`warning: Failed to hardlink files; falling back to full copy` is expected — Docker overlayfs mounts the uv cache and the target on different filesystems. `UV_LINK_MODE=copy` is set in the base image ENV to suppress it. No performance impact.

### nvblox — no pre-built wheels

As of 2025, nvblox has no PyPI wheel (`TODO: Try re-enabling this once we have a pypi page` — upstream). Source build is required on both arches. `FETCHCONTENT_BASE_DIR=/root/.cache/nvblox-deps` caches cmake FetchContent dependencies (stdgpu, Eigen, etc.) between rebuilds.

### Multi-arch registry workflow

Build and push from each machine:
```bash
./docker/build.sh --push --registry sevapru/cap-y --all   # on each arch
```

Create unified manifests (from any machine):
```bash
for img in base default open nvidia; do
  docker buildx imagetools create -t sevapru/cap-y:${img} \
    sevapru/cap-y:${img}-amd64 sevapru/cap-y:${img}-arm64
done
```

After this, `docker pull sevapru/cap-y:base` auto-selects the correct architecture.

### Test coverage

Container smoke tests (`scripts/test_container_*.py`) validate:
- CUDA runtime, OpenCV CUDA, PyTorch CUDA, Open3D CUDA, JAX GPU
- pymodbus, unitree_sdk2_python, Newton (added April 2026)
- cuRobo + ContactGraspNet (required only on `cap-y:default` via `CAPY_REQUIRE_LICENSED_STACK=1`)
- ROS 2 Jazzy stack, CycloneDDS, Nav2, MoveIt 2, nvblox, Drake, LiveKit
- Isaac ROS packages (arch-aware: apt debs on x86_64, colcon dirs on aarch64)

Run via compose profiles:
```bash
docker compose -f docker/docker-compose.capx.yml --profile test run --rm capx-test
docker compose -f docker/docker-compose.capx.yml --profile test-default run --rm capx-test-default
```

---

## Port DemoGrasp Training from IsaacGym to Newton

IsaacGym Preview 4 is x86_64-only deprecated software. Newton (Apache 2.0)
is the GPU-accelerated replacement that works on Jetson Thor (aarch64, SM 110).

Steps:
1. Replace `import isaacgym` with Newton API (`newton.Model`, `newton.State`)
2. Replace `IsaacGymEnvs` task wrapper with Newton RL env (or Isaac Lab 3.0 Newton backend)
3. Port the Inspire hand URDF + grasp env to Newton's `ModelBuilder`
4. Verify GPU-accelerated training with `num_envs=7000` (original DemoGrasp config)
5. Validate checkpoint compatibility (policy network is pure PyTorch — should transfer)

References:
- Newton API: https://newton-physics.github.io/newton/api/newton.html
- Isaac Lab 3.0 Newton backend: isaaclab_newton (kit-less mode)
- DemoGrasp training script: /opt/demograsp/run_rl_grasp.py
