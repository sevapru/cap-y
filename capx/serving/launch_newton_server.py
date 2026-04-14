"""Newton GPU physics server for cap-y.

Wraps the Newton physics engine (Apache 2.0) as a FastAPI HTTP service so that
cap-y environments can offload GPU simulation to a remote host.

Newton is built on NVIDIA Warp and supports loading scenes from MJCF (MuJoCo XML),
URDF, or USD files.  The recommended solver for articulated robot arms is
``SolverMuJoCo`` (uses mujoco_warp internally), with ``SolverFeatherstone`` as a
fallback when mujoco_warp is unavailable.

Usage (standalone):
    uv run python -m capx.serving.launch_newton_server \\
        --mjcf-path /path/to/scene.xml --port 8124

Usage (via launch_servers.py):
    uv run python -m capx.serving.launch_servers --profile newton

API:
    GET  /health         — readiness check
    POST /load           — load/reload a scene from MJCF path or XML string
    POST /reset          — reset simulation to initial state
    POST /step           — step with joint position targets, returns state
    GET  /state          — current simulation state (no stepping)
    POST /set_joints     — teleport joints to specified positions
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("newton_server")

app = FastAPI(title="Newton Physics Server", version="1.0.0")

# ─────────────────────────────────── Global simulation state ─────────────────

_model: Any | None = None
_state_0: Any | None = None
_state_1: Any | None = None
_control: Any | None = None
_contacts: Any | None = None
_solver: Any | None = None
_sim_time: float = 0.0
_sim_dt: float = 1.0 / 600.0   # physics time step (600 Hz = 60 fps × 10 substeps)
_sim_substeps: int = 10         # substeps per control step
_body_names: list[str] = []
_joint_names: list[str] = []
_init_error: str | None = None

_async_lock = asyncio.Lock()

# ─────────────────────────────────── Pydantic models ─────────────────────────


class LoadRequest(BaseModel):
    """Request to load or reload a scene."""

    mjcf_path: str | None = None
    """Absolute path to an MJCF file on the server's filesystem."""

    mjcf_xml: str | None = None
    """Raw MJCF XML string (alternative to mjcf_path)."""

    sim_substeps: int = 10
    """Number of physics substeps per control step."""

    sim_fps: float = 60.0
    """Control frequency (Hz). Physics dt = 1 / (sim_fps * sim_substeps)."""

    solver: str = "mujoco"
    """Solver: "mujoco" (SolverMuJoCo, best for arms), "featherstone", or "xpbd"."""


class SimState(BaseModel):
    """Serialised simulation state returned by /reset, /step, /state."""

    body_names: list[str]
    body_positions: list[list[float]]      # (N, 3) world-frame XYZ
    body_quaternions: list[list[float]]    # (N, 4) WXYZ
    joint_names: list[str]
    joint_q: list[float]                   # joint DOF positions
    joint_qd: list[float]                  # joint DOF velocities
    sim_time: float


class ResetRequest(BaseModel):
    seed: int | None = None


class StepRequest(BaseModel):
    joint_targets: list[float]
    """Target joint positions (position control).  Length must match actuator count."""

    num_substeps: int | None = None
    """Override server default substeps for this step only."""


class SetJointsRequest(BaseModel):
    joint_q: list[float]
    joint_qd: list[float] | None = None


# ─────────────────────────────────── Helpers ─────────────────────────────────


def _extract_state() -> SimState:
    """Extract current simulation state as a Pydantic model (blocking)."""
    # body_q is a warp array of wp.transform, memory layout (N, 7):
    #   [px, py, pz, qx, qy, qz, qw]  per body (quaternion is xyzw in Warp)
    body_q_np = _state_0.body_q.numpy().reshape(-1, 7)
    positions = body_q_np[:, :3].tolist()
    # Convert quaternion xyzw → wxyz (cap-y convention)
    qxyzw = body_q_np[:, 3:]
    qwxyz = qxyzw[:, [3, 0, 1, 2]].tolist()

    joint_q = _state_0.joint_q.numpy().flatten().tolist()
    joint_qd = _state_0.joint_qd.numpy().flatten().tolist()

    return SimState(
        body_names=_body_names,
        body_positions=positions,
        body_quaternions=qwxyz,
        joint_names=_joint_names,
        joint_q=joint_q,
        joint_qd=joint_qd,
        sim_time=_sim_time,
    )


def _init_newton(
    mjcf_path: str | None,
    mjcf_xml: str | None,
    sim_substeps: int,
    sim_fps: float,
    solver_name: str,
) -> None:
    """Build the Newton model and solver (blocking, call from executor)."""
    global _model, _state_0, _state_1, _control, _contacts, _solver
    global _sim_time, _sim_dt, _sim_substeps, _body_names, _joint_names, _init_error

    try:
        import newton  # noqa: PLC0415 — lazy to keep module importable without Newton
        import warp as wp  # noqa: PLC0415
    except ImportError as exc:
        _init_error = f"newton / warp not installed: {exc}"
        raise RuntimeError(_init_error) from exc

    builder = newton.ModelBuilder()

    if mjcf_path is not None:
        path = Path(mjcf_path)
        if not path.exists():
            raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
        builder.add_mjcf(str(path))
        logger.info("Loaded MJCF from %s", mjcf_path)
    elif mjcf_xml is not None:
        import tempfile  # noqa: PLC0415

        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(mjcf_xml)
            tmp_path = f.name
        builder.add_mjcf(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        logger.info("Loaded MJCF from XML string (%d bytes)", len(mjcf_xml))
    else:
        # Minimal scene: flat ground so /reset and /state work without a robot
        builder.add_ground_plane()
        logger.info("No MJCF specified — using empty scene with ground plane")

    _body_names = list(getattr(builder, "body_name", []))
    _joint_names = list(getattr(builder, "joint_name", []))
    logger.info("Bodies (%d): %s...", len(_body_names), _body_names[:8])
    logger.info("Joints (%d): %s...", len(_joint_names), _joint_names[:8])

    _model = builder.finalize()
    _state_0 = _model.state()
    _state_1 = _model.state()
    _control = _model.control()
    _contacts = _model.contacts()

    _sim_substeps = sim_substeps
    _sim_dt = 1.0 / (sim_fps * sim_substeps)
    _sim_time = 0.0
    _init_error = None

    solver_lower = solver_name.lower()
    if solver_lower == "mujoco":
        try:
            _solver = newton.solvers.SolverMuJoCo(_model)
            logger.info("Using SolverMuJoCo (GPU-accelerated via mujoco_warp)")
        except Exception as exc:
            logger.warning(
                "SolverMuJoCo unavailable (%s); falling back to SolverFeatherstone", exc
            )
            _solver = newton.solvers.SolverFeatherstone(_model)
            logger.info("Using SolverFeatherstone")
    elif solver_lower == "featherstone":
        _solver = newton.solvers.SolverFeatherstone(_model)
        logger.info("Using SolverFeatherstone")
    else:
        _solver = newton.solvers.SolverXPBD(_model)
        logger.info("Using SolverXPBD")

    logger.info(
        "Newton ready — %d bodies, %d joints, dt=%.5f s, substeps=%d",
        _model.body_count,
        _model.joint_count,
        _sim_dt,
        _sim_substeps,
    )


def _do_step(joint_targets: list[float], num_substeps: int) -> SimState:
    """Apply joint position targets and advance the simulation (blocking)."""
    global _state_0, _state_1, _sim_time

    import warp as wp  # noqa: PLC0415

    # Write joint targets into the control buffer
    if joint_targets:
        ctrl_np = _control.joint_target.numpy().flatten()
        n_set = min(len(joint_targets), len(ctrl_np))
        ctrl_np[:n_set] = np.array(joint_targets[:n_set], dtype=np.float32)
        _control.joint_target.assign(wp.array(ctrl_np, dtype=wp.float32, device=_model.device))

    for _ in range(num_substeps):
        _state_0.clear_forces()
        _model.collide(_state_0, _contacts)
        _solver.step(
            state_in=_state_0,
            state_out=_state_1,
            control=_control,
            contacts=_contacts,
            dt=_sim_dt,
        )
        _state_0, _state_1 = _state_1, _state_0
        _sim_time += _sim_dt

    return _extract_state()


def _do_reset() -> SimState:
    """Reset simulation to initial state (blocking)."""
    global _state_0, _state_1, _control, _contacts, _sim_time

    _state_0 = _model.state()
    _state_1 = _model.state()
    _control = _model.control()
    _contacts = _model.contacts()
    _sim_time = 0.0
    return _extract_state()


def _do_set_joints(joint_q: list[float], joint_qd: list[float] | None) -> SimState:
    """Teleport joints to given positions (blocking)."""
    import warp as wp  # noqa: PLC0415

    cur_q = _state_0.joint_q.numpy().flatten()
    arr_q = np.array(joint_q, dtype=np.float32)
    n = min(len(arr_q), len(cur_q))
    cur_q[:n] = arr_q[:n]
    _state_0.joint_q.assign(wp.array(cur_q, dtype=wp.float32, device=_model.device))

    if joint_qd is not None:
        cur_qd = _state_0.joint_qd.numpy().flatten()
        arr_qd = np.array(joint_qd, dtype=np.float32)
        m = min(len(arr_qd), len(cur_qd))
        cur_qd[:m] = arr_qd[:m]
        _state_0.joint_qd.assign(wp.array(cur_qd, dtype=wp.float32, device=_model.device))

    return _extract_state()


async def _run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# ─────────────────────────────────── Routes ──────────────────────────────────


@app.get("/health")
async def health():
    loaded = _model is not None
    return {
        "status": "ready" if (loaded and _init_error is None) else "initializing",
        "loaded": loaded,
        "error": _init_error,
        "sim_time": _sim_time,
        "body_count": _model.body_count if loaded else 0,
        "joint_count": _model.joint_count if loaded else 0,
    }


@app.post("/load")
async def load_scene(req: LoadRequest):
    """Load or reload a physics scene from an MJCF file or XML string."""
    async with _async_lock:
        try:
            await _run_in_thread(
                _init_newton,
                req.mjcf_path,
                req.mjcf_xml,
                req.sim_substeps,
                req.sim_fps,
                req.solver,
            )
        except Exception as exc:
            logger.exception("Failed to load scene")
            raise HTTPException(500, f"Load failed: {exc}") from exc
    return {
        "status": "loaded",
        "body_count": _model.body_count,
        "joint_count": _model.joint_count,
        "body_names": _body_names,
        "joint_names": _joint_names,
    }


@app.post("/reset", response_model=SimState)
async def reset(req: ResetRequest):
    """Reset the simulation to the initial state and return the new state."""
    if _model is None:
        raise HTTPException(503, "Newton not loaded — call POST /load first")
    async with _async_lock:
        try:
            return await _run_in_thread(_do_reset)
        except Exception as exc:
            logger.exception("Reset failed")
            raise HTTPException(500, f"Reset failed: {exc}") from exc


@app.post("/step", response_model=SimState)
async def step(req: StepRequest):
    """Advance the simulation by applying joint position targets."""
    if _model is None:
        raise HTTPException(503, "Newton not loaded — call POST /load first")
    num_substeps = req.num_substeps if req.num_substeps is not None else _sim_substeps
    async with _async_lock:
        try:
            return await _run_in_thread(_do_step, req.joint_targets, num_substeps)
        except Exception as exc:
            logger.exception("Step failed")
            raise HTTPException(500, f"Step failed: {exc}") from exc


@app.get("/state", response_model=SimState)
async def get_state():
    """Return the current simulation state without stepping."""
    if _model is None:
        raise HTTPException(503, "Newton not loaded — call POST /load first")
    async with _async_lock:
        try:
            return await _run_in_thread(_extract_state)
        except Exception as exc:
            raise HTTPException(500, f"State extraction failed: {exc}") from exc


@app.post("/set_joints", response_model=SimState)
async def set_joints(req: SetJointsRequest):
    """Teleport robot joints to specified positions (zero velocity by default)."""
    if _model is None:
        raise HTTPException(503, "Newton not loaded — call POST /load first")
    async with _async_lock:
        try:
            return await _run_in_thread(_do_set_joints, req.joint_q, req.joint_qd)
        except Exception as exc:
            logger.exception("set_joints failed")
            raise HTTPException(500, f"set_joints failed: {exc}") from exc


# ─────────────────────────────────── Entrypoint ──────────────────────────────


def main(
    mjcf_path: str | None = None,
    solver: str = "mujoco",
    sim_substeps: int = 10,
    sim_fps: float = 60.0,
    port: int = 8124,
    host: str = "127.0.0.1",
) -> None:
    """Launch the Newton GPU physics server.

    Args:
        mjcf_path: MJCF file to pre-load at startup.  Can also be loaded later
            via POST /load.
        solver: Newton solver — "mujoco" (SolverMuJoCo, default), "featherstone",
            or "xpbd".
        sim_substeps: Physics substeps per control step.
        sim_fps: Control frequency (Hz).  Physics dt = 1 / (sim_fps * sim_substeps).
        port: HTTP port to bind.
        host: Host address to bind.
    """
    if mjcf_path is not None:
        logger.info("Pre-loading MJCF: %s", mjcf_path)
        t0 = time.monotonic()
        _init_newton(mjcf_path, None, sim_substeps, sim_fps, solver)
        logger.info("Scene loaded in %.1f s", time.monotonic() - t0)
    else:
        logger.info(
            "No --mjcf-path given; server will wait for POST /load before simulating"
        )

    logger.info("Newton server starting on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
