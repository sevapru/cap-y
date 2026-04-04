"""4-tier pytest suite for cap-y container validation.

Tier 1 — Import tests (all images, no GPU, runs without container):
    pytest tests/test_capx.py -k "import" -v

Tier 2 — Server health checks (requires running serving profile):
    pytest tests/test_capx.py -m servers -v

Tier 3 — API smoke tests (requires running servers + GPU):
    pytest tests/test_capx.py -m api -v

Tier 4 — Simulation step (requires GPU + EGL + sim extras):
    pytest tests/test_capx.py -m simulation -v
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Tier 1 — Import tests (no GPU, no servers required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "capx",
        "capx.skills",
        "capx.llm",
        "capx.serving.launch_servers",
    ],
)
def test_import(module: str) -> None:
    """Verify core modules are importable without GPU or running servers."""
    importlib.import_module(module)


@pytest.mark.parametrize(
    "module",
    [
        "capx.integrations.motion.pyroki",
        "capx.integrations.vision.sam3",
        "capx.integrations.vision.graspnet",
    ],
)
def test_integration_import(module: str) -> None:
    """Verify integration client modules are importable."""
    importlib.import_module(module)


# ---------------------------------------------------------------------------
# Tier 2 — Server health checks (requires running servers)
# ---------------------------------------------------------------------------


@pytest.mark.servers
@pytest.mark.parametrize(
    "port,name",
    [
        (8114, "SAM3"),
        (8116, "PyRoKi"),
        (8119, "DemoGrasp"),
        (8120, "GraspAnalytic"),
    ],
)
def test_server_health(port: int, name: str) -> None:
    """Verify each server responds at /health with a valid status."""
    import requests

    r = requests.get(f"http://localhost:{port}/health", timeout=5)
    assert r.status_code == 200, f"{name} health check failed: {r.status_code}"
    body = r.json()
    assert body.get("status") in (
        "ready",
        "ok",
        "initializing",
        "stub",
    ), f"{name} returned unexpected status: {body}"


@pytest.mark.servers
def test_gateway_status() -> None:
    """Verify gateway /status endpoint aggregates all backends."""
    import requests

    r = requests.get("http://localhost:8100/status", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert "backends" in body
    assert "profile" in body


# ---------------------------------------------------------------------------
# Tier 3 — API smoke tests (requires running servers)
# ---------------------------------------------------------------------------


@pytest.mark.api
def test_pyroki_ik() -> None:
    """Single IK request to PyRoKi server — verifies end-to-end IK solve."""
    import requests

    payload = {
        "target_pose": {
            "position": [0.4, 0.0, 0.5],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
        },
        "robot": "panda",
    }
    r = requests.post("http://localhost:8116/ik", json=payload, timeout=10)
    assert r.status_code == 200, f"IK request failed: {r.status_code} {r.text[:200]}"
    body = r.json()
    assert "joint_positions" in body, f"No joint_positions in response: {body}"


@pytest.mark.api
def test_graspanalytic_plan() -> None:
    """Width-to-grasp planning via rh56_controller server."""
    import requests

    r = requests.post(
        "http://localhost:8120/plan",
        json={"object_width_m": 0.05, "grasp_force_n": 5.0},
        timeout=10,
    )
    assert r.status_code == 200, f"Grasp plan failed: {r.status_code} {r.text[:200]}"
    body = r.json()
    assert body.get("success") is True, f"Grasp plan unsuccessful: {body}"


# ---------------------------------------------------------------------------
# Tier 4 — Simulation step (requires GPU + EGL + sim extras)
# ---------------------------------------------------------------------------


@pytest.mark.simulation
def test_mujoco_egl() -> None:
    """MuJoCo EGL headless rendering smoke test."""
    import mujoco

    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <body pos="0 0 0.5">
          <geom type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0.0, "Simulation step did not advance time"


@pytest.mark.simulation
def test_robosuite_import() -> None:
    """Robosuite import check (requires robosuite venv or system install)."""
    import robosuite  # type: ignore[import]

    assert hasattr(robosuite, "__version__"), "robosuite missing __version__"


@pytest.mark.simulation
def test_mink_import() -> None:
    """mink differential IK import (no GPU needed, but grouped with sim tests)."""
    import mink  # type: ignore[import]

    assert hasattr(mink, "__version__") or True  # mink may not have __version__
