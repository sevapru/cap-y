"""DemoGrasp inference server — universal dexterous grasping.

Wraps DemoGrasp (MIT, ICLR 2026) as a FastAPI service.
Accepts point cloud + RGB input, returns ranked 6-DOF grasp poses for
dexterous hands (Inspire RH56DFX natively supported).

Reference: https://beingbeyond.github.io/DemoGrasp/
           https://github.com/BeingBeyond/DemoGrasp

Usage (via launch_servers.py):
    python -m capx.serving.launch_demograsp_server --port 8119 --device cuda

API:
    GET  /health          — readiness check
    POST /grasp           — grasp poses from point cloud
    POST /grasp_from_rgb  — grasp poses from RGB-D image
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="DemoGrasp Server", version="1.0.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_MODEL: Any | None = None
_DEVICE: str = "cuda"
_CHECKPOINT_PATH: str | None = None

_GPU_SEMAPHORE = asyncio.Semaphore(1)


async def _run_on_gpu(fn, *args, **kwargs):
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _numpy_to_b64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode()


def _b64_to_numpy(s: str) -> np.ndarray:
    try:
        return np.load(io.BytesIO(base64.b64decode(s)))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid numpy payload: {exc}")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class GraspRequest(BaseModel):
    """Point cloud input for grasp generation."""

    pc_base64: str
    """Float32 (N, 3) point cloud in camera frame, base64-encoded numpy array."""

    rgb_base64: str | None = None
    """Optional uint8 (H, W, 3) RGB image, base64-encoded numpy array."""

    hand_type: str = "inspire"
    """Target hand embodiment. Supported: 'inspire', 'shadow', 'allegro'."""

    top_k: int = 10
    """Number of top grasp poses to return."""


class GraspResponse(BaseModel):
    """Ranked grasp poses for the target hand."""

    poses_base64: str
    """Float32 (K, 4, 4) SE(3) grasp poses, base64-encoded."""

    scores_base64: str
    """Float32 (K,) confidence scores, base64-encoded."""

    joint_targets_base64: str | None = None
    """Float32 (K, n_joints) finger joint angles for the target hand, if available."""


class RGBDRequest(BaseModel):
    """RGB-D image input for grasp generation."""

    rgb_base64: str
    """uint8 (H, W, 3) RGB image, base64-encoded."""

    depth_base64: str
    """Float32 (H, W) depth map in metres, base64-encoded."""

    cam_K_base64: str
    """Float64 (3, 3) camera intrinsics matrix, base64-encoded."""

    hand_type: str = "inspire"
    top_k: int = 10


# ---------------------------------------------------------------------------
# Inference logic
# ---------------------------------------------------------------------------


def _infer_grasps(pc: np.ndarray, rgb: np.ndarray | None, hand_type: str, top_k: int):
    """Run DemoGrasp inference. Returns (poses, scores, joint_targets)."""
    if _MODEL is None:
        raise RuntimeError("Model not loaded")

    import torch

    pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        if hasattr(_MODEL, "predict"):
            result = _MODEL.predict(pc_tensor, hand_type=hand_type)
        else:
            result = _MODEL(pc_tensor)

    if isinstance(result, dict):
        poses = result.get("poses", result.get("grasp_poses", np.zeros((0, 4, 4))))
        scores = result.get("scores", np.zeros(len(poses)))
        joints = result.get("joint_targets", None)
    elif isinstance(result, (list, tuple)) and len(result) >= 2:
        poses, scores = result[0], result[1]
        joints = result[2] if len(result) > 2 else None
    else:
        poses = np.zeros((0, 4, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        joints = None

    if isinstance(poses, __import__("torch").Tensor):
        poses = poses.cpu().numpy()
    if isinstance(scores, __import__("torch").Tensor):
        scores = scores.cpu().numpy()
    if joints is not None and hasattr(joints, "cpu"):
        joints = joints.cpu().numpy()

    # Sort by score descending and take top-k
    if len(scores) > 0:
        order = np.argsort(scores)[::-1][:top_k]
        poses = poses[order]
        scores = scores[order]
        if joints is not None:
            joints = joints[order]

    return poses, scores, joints


def _do_grasp(req: GraspRequest) -> GraspResponse:
    pc = _b64_to_numpy(req.pc_base64)
    rgb = _b64_to_numpy(req.rgb_base64) if req.rgb_base64 else None
    poses, scores, joints = _infer_grasps(pc, rgb, req.hand_type, req.top_k)
    return GraspResponse(
        poses_base64=_numpy_to_b64(poses.astype(np.float32)),
        scores_base64=_numpy_to_b64(scores.astype(np.float32)),
        joint_targets_base64=_numpy_to_b64(joints.astype(np.float32)) if joints is not None else None,
    )


def _do_grasp_from_rgbd(req: RGBDRequest) -> GraspResponse:
    rgb = _b64_to_numpy(req.rgb_base64)
    depth = _b64_to_numpy(req.depth_base64)
    cam_K = _b64_to_numpy(req.cam_K_base64)

    # Back-project depth → point cloud
    h, w = depth.shape
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    valid = z > 0
    pc = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)

    poses, scores, joints = _infer_grasps(pc, rgb, req.hand_type, req.top_k)
    return GraspResponse(
        poses_base64=_numpy_to_b64(poses.astype(np.float32)),
        scores_base64=_numpy_to_b64(scores.astype(np.float32)),
        joint_targets_base64=_numpy_to_b64(joints.astype(np.float32)) if joints is not None else None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ready" if _MODEL is not None else "initializing"}


@app.post("/grasp", response_model=GraspResponse)
async def grasp_endpoint(req: GraspRequest):
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        return await _run_on_gpu(_do_grasp, req)
    except Exception as exc:
        logger.error("DemoGrasp /grasp failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/grasp_from_rgb", response_model=GraspResponse)
async def grasp_from_rgb_endpoint(req: RGBDRequest):
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        return await _run_on_gpu(_do_grasp_from_rgbd, req)
    except Exception as exc:
        logger.error("DemoGrasp /grasp_from_rgb failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _load_model(checkpoint_path: str | None, device: str) -> Any:
    """Load DemoGrasp model from checkpoint."""
    demograsp_root = Path("/opt/demograsp")
    if not demograsp_root.exists():
        # Try workspace path for dev
        demograsp_root = Path(__file__).parents[2] / "capx" / "third_party" / "demograsp"

    if str(demograsp_root) not in sys.path:
        sys.path.insert(0, str(demograsp_root))

    try:
        import torch

        # DemoGrasp policy import — adapt as the repo's public API evolves
        try:
            from policy import DemoGraspPolicy  # type: ignore[import]

            model = DemoGraspPolicy()
        except ImportError:
            try:
                from demograsp.policy import DemoGraspPolicy  # type: ignore[import]

                model = DemoGraspPolicy()
            except ImportError:
                logger.warning(
                    "Could not import DemoGrasp policy class; running in stub mode. "
                    "Ensure /opt/demograsp is correctly installed."
                )
                return None

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state, strict=False)
            logger.info("Loaded DemoGrasp checkpoint: %s", checkpoint_path)
        else:
            logger.warning(
                "No checkpoint path provided or file not found (%s). "
                "Download pre-trained weights from: "
                "https://github.com/BeingBeyond/DemoGrasp",
                checkpoint_path,
            )

        model = model.to(device).eval()
        return model

    except Exception as exc:
        logger.error("Failed to load DemoGrasp model: %s", exc)
        return None


def main(
    device: str = "cuda",
    port: int = 8119,
    host: str = "127.0.0.1",
    checkpoint: str | None = None,
):
    """Launch DemoGrasp inference server.

    Args:
        device: PyTorch device string ("cuda" or "cpu").
        port: Port to bind the HTTP server to.
        host: Host address to bind.
        checkpoint: Path to DemoGrasp checkpoint file (.pt). If omitted,
            looks for DEMOGRASP_CHECKPOINT env var, then runs in stub mode.
    """
    global _MODEL, _DEVICE, _CHECKPOINT_PATH

    _DEVICE = device
    _CHECKPOINT_PATH = checkpoint or os.environ.get("DEMOGRASP_CHECKPOINT")

    logger.info("Loading DemoGrasp model on %s...", device)
    _MODEL = _load_model(_CHECKPOINT_PATH, device)

    if _MODEL is None:
        logger.warning(
            "DemoGrasp model not loaded — /grasp will return 503 until a checkpoint "
            "is provided. Server starting anyway for health checks."
        )
    else:
        logger.info("DemoGrasp ready. Starting server on %s:%d", host, port)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
