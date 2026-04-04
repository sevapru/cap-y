# cap-y Roadmap

Task list for delegation to subagents or contributors. Each item is scoped to be executable independently.

---

## CI/CD Pipeline

Build all four container images on a self-hosted Jetson Thor runner, push to GHCR, and run the test cascade on every push to `main`.

- Set up GitHub Actions self-hosted runner on Jetson Thor (`runs-on: [self-hosted, jetson-thor]`)
- Workflow: `build.sh --all` → push to `ghcr.io/sevapru/cap-y:{base,default,open,nvidia}` → run `capx-test`, `capx-test-open`, `capx-test-nvidia` profiles
- Cache: mount ccache and uv-cache as named Docker volumes between runs
- Gate: fail PR merge if any test profile returns non-zero

**Subagent inputs needed:** GitHub repo access, runner registration token, GHCR push credentials.

---

## Multi-Arch Build Matrix

Add `sm_89` (RTX 4080 / RTX 4090) as a second build target alongside `sm_110`.

- Extend `build.sh` to accept `--arch-list "11.0 8.9"` and build both in parallel
- Tag images as `cap-y:latest-sm110`, `cap-y:latest-sm89`, plus `cap-y:latest` pointing to sm110
- Validate on a desktop RTX 4080 with the existing test suite

**Subagent inputs needed:** Access to an RTX 4080 / 4090 machine for validation.

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
