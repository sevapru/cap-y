# Development Guide

## Testing

### Unit tests

```bash
uv run pytest tests/test_environments.py -q
```

Run a specific test:
```bash
uv run pytest tests/test_environments.py::test_franka_pick_place_code_env -q
uv run pytest tests/test_environments.py::test_franka_nut_assembly_code_env -q
```

### Oracle code testing

Check how oracle code performs on a specific environment:
```bash
uv run tests/test_environments.py --env_name YOUR_ENV_NAME
```

### Regression tests (expected rewards)

Before merging, run these environments and verify rewards match expectations:

```bash
uv run capx/envs/launch.py --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml
# Expected avg. reward: ~0.99

uv run capx/envs/launch.py --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml
# Expected avg. reward: ~0.90

uv run capx/envs/launch.py --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml
# Expected avg. reward: ~0.50

uv run capx/envs/launch.py --config-path env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged.yaml
# Expected avg. reward: ~0.15

uv run capx/envs/launch.py --config-path env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml
# Expected avg. reward: ~0.25

uv run capx/envs/launch.py --config-path env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml
# Expected avg. reward: ~0.20
```

## Docker / cap-y image stacks

Perception and evaluation often use the **cap-y** images defined in `docker/docker-compose.capx.yml`. Only **one** of `cap-y-base`, `cap-y-default`, `cap-y-open`, or `cap-y-nvidia` should run at a time on a host (overlapping host ports).

**Compose profiles** (enable with `--profile` or `COMPOSE_PROFILES`):

| Profile | Services |
|---------|----------|
| `base` | `cap-y-base` (servers) |
| `default` | `cap-y-default` |
| `open` | `cap-y-open` |
| `nvidia` | `cap-y-nvidia` |
| `shell` | `capx-base` — interactive bash, same image as base, no servers |
| `dev` | `cap-y-dev` |
| `test` / `test-default` / `test-open` / `test-nvidia` | One-shot verification containers |

From `docker/`:

```bash
docker compose -f docker-compose.capx.yml --profile base up -d cap-y-base
docker compose -f docker-compose.capx.yml --profile test run --rm capx-test
docker compose -f docker-compose.capx.yml --profile test-default run --rm capx-test-default
docker compose -f docker-compose.capx.yml --profile test-open run --rm capx-test-open
docker compose -f docker-compose.capx.yml --profile test-nvidia run --rm capx-test-nvidia
```

- **`CAPX_PROFILE`** (e.g. `open`, `default`, `full`, `minimal`) selects which **cap-x HTTP backends** the container **entrypoint** starts; it does not replace choosing the **image** (`base` / `default` / `open` / `nvidia`).
- **`capx-test`** on `cap-y:base` treats **cuRobo** and **ContactGraspNet** as optional (not installed on the base image). **`capx-test-default`** sets **`CAPY_REQUIRE_LICENSED_STACK=1`** so those packages must import on **`cap-y:default`**.

Full build/start tables: root [README.md](../README.md).

## Benchmark preflight (one-button safety)

Before long **LIBERO batch** runs, validate submodules, Python/`pyproject` constraints, `~/.libero/config.yaml`, YAML `api_servers` imports, LLM proxy reachability, MuJoCo/CUDA, and disk space:

```bash
uv run python scripts/benchmark_preflight.py \
  --suite libero \
  --config-path env_configs/libero/franka_libero_cap_agent0.yaml \
  --strict
```

- **`--strict`**: exit non-zero on any **WARN** as well as **FAIL** (useful for automation).
- **`--no-check-server`**: skip TCP check to `--server-url` (default `http://127.0.0.1:8110/chat/completions`).
- Preflight also prints **`tool.uv.conflicts`** and **`override-dependencies`** from `pyproject.toml` (dedicated **LIBERO** venv vs **robosuite** extra, **verl**/**molmo** x86_64-only pins, etc.).

**Wrapper** (preflight, then `run_libero_batch`; use a dedicated LIBERO venv when possible):

```bash
./scripts/benchmark_one_button.sh
# Optional: BENCHMARK_CONFIG, BENCHMARK_SERVER_URL, BENCHMARK_PREFLIGHT_STRICT=0|1
```

Details and `~/.libero` template: [libero-tasks.md](libero-tasks.md) (Preflight subsection).

## Linting

```bash
ruff check          # lint
ruff check --fix    # auto-fix
ruff format         # format
```

When contributing, please use ruff (automatically installed) for linting. See [ruff docs](https://docs.astral.sh/ruff/tutorial/#getting-started).

## SAM 3 / SAM 3.1 access

SAM3 by Meta is installed as a vendored package via `capx/third_party/sam3`. Before using SAM 3 or SAM 3.1, you must accept the model agreement on HuggingFace and authenticate:

```bash
hf auth login   # or: huggingface-cli login
```

### SAM 3 (open)

Checkpoint is downloaded automatically on first server start from `facebook/sam3` → `sam3.pt` (~2.4 GB), cached at `~/.cache/huggingface/hub/models--facebook--sam3/`.

### SAM 3.1 (gated — contact-info agreement required)

SAM 3.1 is a gated model. You must agree to share contact information at https://huggingface.co/facebook/sam3.1 before downloading.

Checkpoint location (after download): `~/.cache/huggingface/hub/models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/`

```bash
hf download facebook/sam3.1
```

> **Note:** `huggingface_hub` ≥ 1.10.2 is recommended (`pip install -U huggingface_hub`).

Start the server pointing at SAM 3.1:

```bash
# Via HF token (required for gated access)
HF_TOKEN=hf_xxx python -m capx.serving.launch_sam3_server \
    --hf-repo facebook/sam3.1 --device cuda --port 8114

# Or with a pre-downloaded checkpoint (no token needed at runtime)
python -m capx.serving.launch_sam3_server \
    --checkpoint-path ~/.cache/huggingface/hub/models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3_1.pt \
    --hf-repo facebook/sam3.1
```

SAM 3.1 is a drop-in replacement for SAM 3 — the `/segment` and `/segment_point` API endpoints are identical. Key improvement: ~7× faster multi-object inference (Object Multiplex) and better video object segmentation on 6/7 benchmarks.

## LIBERO-PRO installation

```bash
uv sync --extra libero --extra contactgraspnet
```

For headless servers, also install EGL rendering:
```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
```

See [libero-tasks.md](libero-tasks.md) for the full task reference, headless `~/.libero/config.yaml`, and **benchmark preflight** / one-button wrapper.

## BEHAVIOR installation (Isaac Sim)

BEHAVIOR tasks require NVIDIA Isaac Sim and OmniGibson. Use the provided install script:

```bash
cd capx/third_party/b1k
./uv_install.sh --dataset --accept-dataset-tos
cd ../../..
```

This installs:
- **BDDL** (Behavior Domain Definition Language)
- **OmniGibson** (simulator, editable install)
- **Isaac Sim 4.5.0** (downloaded as pip wheels from pypi.nvidia.com)
- **cuRobo** (GPU-accelerated motion planning, from StanfordVL fork)
- **PyRoKi** (IK solver)
- **SAM3 + ContactGraspNet dependencies** (perception server runtime deps)
- **Datasets** (robot assets, BEHAVIOR-1K assets, 2025 challenge task instances)

The script also fixes the known websockets conflict with Isaac Sim extscache.

### Post-install fix: cuRobo JIT headers

After running `uv_install.sh`, copy the cuRobo CUDA JIT headers (required for first-run kernel compilation). Run with the b1k venv active:

```bash
source capx/third_party/b1k/.venv/bin/activate
cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

> **Note:** On first run, cuRobo JIT-compiles CUDA kernels (3–5 min). Isaac Sim also does initial shader compilation on first run, adding another ~3 min to startup.

### Prerequisites

- Python 3.10 (Isaac Sim wheels are cp310-only)
- NVIDIA GPU with CUDA 12.x (driver 550+)
- `libegl1` and `libgl1` for headless rendering (see above)

### Environment variables

For headless (no display) servers, set before running:
```bash
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
```

See [behavior-tasks.md](behavior-tasks.md) for task configs and expected baselines.

## Vendored submodules

We vendor some upstream repos for reproducible, offline tests. Initialize submodules after cloning:

```bash
git submodule update --init --recursive
```

## Sharp bits / known issues

1. For MuJoCo, use `condim="4"` on bodies where collision matters. The default is 3, which may lead to slippage. See [MuJoCo docs](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-condim).
2. The sandbox for code execution is local and safe-ish. For stronger isolation, use Docker or nsjail.

### Running Newton examples on Kubuntu with an RTX 4080 (hybrid GPU laptop)

On Kubuntu with a Wayland session and a hybrid AMD iGPU + NVIDIA dGPU setup, the Newton GL viewer fails with `OpenGL.error.Error: Attempt to retrieve context when no valid context`. This happens because Newton sets `PYOPENGL_PLATFORM=glx`, but `glx` is not a registered plugin name in PyOpenGL 3.x — the correct name is `x11`. PyOpenGL silently falls through to the `wayland`/EGL platform, which cannot see the GLX context that pyglet creates. Additionally, without PRIME offload vars, the default GLX device is the AMD iGPU, not the NVIDIA dGPU.

Use the following prefix to run any Newton example correctly:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia PYOPENGL_PLATFORM=x11 python -m newton.examples robot_policy
```
