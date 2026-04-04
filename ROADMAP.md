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
