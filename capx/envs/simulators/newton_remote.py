"""Newton remote environment — BaseEnv backed by a remote Newton physics server.

Instead of running MuJoCo/Robosuite in-process, this environment forwards all
simulation calls (reset, step, get_observation) to the Newton GPU physics server
(``capx/serving/launch_newton_server.py``) via HTTP.

The Newton server can run on a remote GPU machine (e.g. a Jetson Thor with SM 110),
while the policy / task logic runs locally.

Usage example
-------------
Start the Newton server on the GPU host::

    python -m capx.serving.launch_newton_server \\
        --mjcf-path /path/to/franka_scene.xml \\
        --host 0.0.0.0 --port 8124

Then in your Python code or YAML config::

    env = NewtonRemoteEnv(
        server_url="http://gpu-host:8124",
        robot_joint_names=["panda_joint1", ..., "panda_joint7"],
    )
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Observation keys
----------------
``robot_joint_pos``   – ndarray (n_robot_joints,)  joint positions
``robot_joint_vel``   – ndarray (n_robot_joints,)  joint velocities
``joint_q``           – ndarray (total_dofs,)       all joint positions
``joint_qd``          – ndarray (total_dofs,)       all joint velocities
``body_positions``    – ndarray (N, 3)              world-frame XYZ per body
``body_quaternions``  – ndarray (N, 4)              WXYZ quaternion per body
``body_names``        – list[str]                   body name for each row
``sim_time``          – float                       elapsed simulation time (s)
``body_{name}_pos``   – ndarray (3,)                convenience per-body position
``body_{name}_quat``  – ndarray (4,)                convenience per-body quaternion
"""

from __future__ import annotations

import logging
from typing import Any, SupportsFloat

import numpy as np
import requests

from capx.envs.base import BaseEnv

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8124"


class NewtonRemoteEnv(BaseEnv):
    """Gymnasium-compatible environment backed by a remote Newton physics server.

    Args:
        server_url: Base URL of the Newton server.
        mjcf_path: MJCF path *on the server's filesystem* to load at init time.
            Pass ``None`` if the server was pre-loaded with ``--mjcf-path``.
        mjcf_xml: Raw MJCF XML string to load (alternative to ``mjcf_path``).
        sim_substeps: Physics substeps sent with each ``/step`` request.
        sim_fps: Simulation frequency used when loading via ``/load``.
        solver: Newton solver (``"mujoco"``, ``"featherstone"``, ``"xpbd"``).
        robot_joint_names: If given, ``robot_joint_pos`` / ``robot_joint_vel``
            will be filtered to only these joint names (in order).  If ``None``,
            all joint DOFs are returned.
        max_steps: Episode truncation length.
        request_timeout: Per-request HTTP timeout in seconds.
        privileged: Passed through from register_env; unused by this env.
        enable_render: Passed through from register_env; unused (no camera).
        viser_debug: Passed through from register_env; unused.
    """

    def __init__(
        self,
        server_url: str = DEFAULT_URL,
        mjcf_path: str | None = None,
        mjcf_xml: str | None = None,
        sim_substeps: int = 10,
        sim_fps: float = 60.0,
        solver: str = "mujoco",
        robot_joint_names: list[str] | None = None,
        max_steps: int = 1500,
        request_timeout: float = 30.0,
        privileged: bool = False,
        enable_render: bool = False,
        viser_debug: bool = False,
    ) -> None:
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self._robot_joint_names = robot_joint_names
        self.max_steps = max_steps
        self._timeout = request_timeout
        self._sim_substeps = sim_substeps
        self._step_count: int = 0
        self._last_state: dict[str, Any] | None = None

        if mjcf_path is not None or mjcf_xml is not None:
            self._load_scene(mjcf_path, mjcf_xml, sim_substeps, sim_fps, solver)

    # ------------------------------------------------------------------ public

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step_count = 0
        resp = requests.post(
            f"{self.server_url}/reset",
            json={"seed": seed},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        self._last_state = resp.json()
        return self._state_to_obs(self._last_state), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        self._step_count += 1
        targets = action.tolist() if isinstance(action, np.ndarray) else list(action)
        resp = requests.post(
            f"{self.server_url}/step",
            json={"joint_targets": targets, "num_substeps": self._sim_substeps},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        self._last_state = resp.json()
        obs = self._state_to_obs(self._last_state)
        reward = float(self.compute_reward())
        terminated = bool(self.task_completed())
        truncated = self._step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def get_observation(self) -> dict[str, Any]:
        if self._last_state is None:
            resp = requests.get(f"{self.server_url}/state", timeout=self._timeout)
            resp.raise_for_status()
            self._last_state = resp.json()
        return self._state_to_obs(self._last_state)

    def compute_reward(self) -> SupportsFloat:
        """Default reward is 0. Override in task-specific subclasses."""
        return 0.0

    def task_completed(self) -> bool:
        """Default is never completed. Override in task-specific subclasses."""
        return False

    def set_joints(
        self,
        joint_q: np.ndarray,
        joint_qd: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Teleport robot joints to specified positions (zero velocity by default).

        Useful for setting the robot to a known configuration without running
        the physics solver.

        Args:
            joint_q: Target joint positions (DOF-indexed).
            joint_qd: Target joint velocities; zeros if omitted.

        Returns:
            Observation dict at the new state.
        """
        payload: dict[str, Any] = {"joint_q": joint_q.tolist()}
        if joint_qd is not None:
            payload["joint_qd"] = joint_qd.tolist()
        resp = requests.post(
            f"{self.server_url}/set_joints", json=payload, timeout=self._timeout
        )
        resp.raise_for_status()
        self._last_state = resp.json()
        return self._state_to_obs(self._last_state)

    def health(self) -> dict[str, Any]:
        """Return the server health dict."""
        resp = requests.get(f"{self.server_url}/health", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------- internals

    def _load_scene(
        self,
        mjcf_path: str | None,
        mjcf_xml: str | None,
        sim_substeps: int,
        sim_fps: float,
        solver: str,
    ) -> None:
        payload: dict[str, Any] = {
            "sim_substeps": sim_substeps,
            "sim_fps": sim_fps,
            "solver": solver,
        }
        if mjcf_path is not None:
            payload["mjcf_path"] = mjcf_path
        if mjcf_xml is not None:
            payload["mjcf_xml"] = mjcf_xml
        resp = requests.post(
            f"{self.server_url}/load", json=payload, timeout=120.0
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "Newton scene loaded: %d bodies, %d joints",
            data.get("body_count", 0),
            data.get("joint_count", 0),
        )

    def _state_to_obs(self, state: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw Newton state dict to the observation dict."""
        joint_names: list[str] = state.get("joint_names", [])
        joint_q = np.array(state.get("joint_q", []), dtype=np.float64)
        joint_qd = np.array(state.get("joint_qd", []), dtype=np.float64)
        body_names: list[str] = state.get("body_names", [])
        body_pos = np.array(state.get("body_positions", []), dtype=np.float64)
        body_quat = np.array(state.get("body_quaternions", []), dtype=np.float64)

        # Filter to robot joints when joint name list is given
        if self._robot_joint_names is not None and joint_names:
            name_to_idx = {n: i for i, n in enumerate(joint_names)}
            idx = [name_to_idx[n] for n in self._robot_joint_names if n in name_to_idx]
            robot_q = joint_q[idx] if idx else joint_q
            robot_qd = joint_qd[idx] if idx else joint_qd
        else:
            robot_q = joint_q
            robot_qd = joint_qd

        obs: dict[str, Any] = {
            "robot_joint_pos": robot_q,
            "robot_joint_vel": robot_qd,
            "joint_q": joint_q,
            "joint_qd": joint_qd,
            "joint_names": joint_names,
            "body_positions": body_pos,
            "body_quaternions": body_quat,
            "body_names": body_names,
            "sim_time": state.get("sim_time", 0.0),
        }

        # Convenience: named per-body keys (e.g. obs["body_panda_link7_pos"])
        for i, name in enumerate(body_names):
            if i < len(body_pos):
                obs[f"body_{name}_pos"] = body_pos[i]
                obs[f"body_{name}_quat"] = body_quat[i]

        return obs


__all__ = ["NewtonRemoteEnv"]
