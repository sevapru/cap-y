import asyncio
import base64
import functools
import io
import logging
import os
from typing import Any

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAM3 Server", version="1.0.0")

# SAM 3 / SAM 3.1 HuggingFace repo identifiers
_HF_REPOS: dict[str, tuple[str, str]] = {
    "facebook/sam3":   ("sam3.pt",   "config.json"),
    "facebook/sam3.1": ("sam3_1.pt", "config.json"),
}

# Global state
_PROCESSOR: Any | None = None
_MODEL: Any | None = None
_DEVICE: str = "cuda"
_HF_REPO: str = "facebook/sam3"


@app.get("/health")
async def health():
    return {
        "status": "ready" if _MODEL is not None else "initializing",
        "model": _HF_REPO,
    }

# Semaphore to serialize GPU access (prevents OOM from concurrent inference)
_GPU_SEMAPHORE = asyncio.Semaphore(1)


async def _run_on_gpu(fn, *args, **kwargs):
    """Run a blocking GPU function without blocking the event loop."""
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))

# --- Helper Functions ---


def _to_numpy(tensor: Any) -> np.ndarray:
    """Convert tensor-like object to numpy array."""
    if hasattr(tensor, "detach"):
        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "cpu"):
        t = tensor.cpu()
        if hasattr(t, "dtype") and t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def decode_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def encode_mask(mask: np.ndarray) -> str:
    # Pack boolean mask to bytes (uint8) then base64
    return base64.b64encode(mask.astype(np.uint8).tobytes()).decode("utf-8")


def encode_array(arr: np.ndarray) -> str:
    """Encode a numpy array as base64 bytes."""
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("utf-8")


# --- Request/Response Models ---


class SegmentRequest(BaseModel):
    image_base64: str
    text_prompt: str


class PointPromptRequest(BaseModel):
    image_base64: str
    point_coords: list[float]  # [x, y] — JSON arrays, not tuples


class PointPromptResponse(BaseModel):
    scores: list[float]
    masks_base64: str
    masks_shape: list[int]  # [num_masks, H, W]
    masks_dtype: str


class MaskData(BaseModel):
    mask_base64: str
    shape: list[int]  # [H, W]
    box: list[float]  # [x1, y1, x2, y2]
    score: float
    label: str


class SegmentResponse(BaseModel):
    results: list[MaskData]


# --- Core Logic ---


def _do_segment(pil_image: Image.Image, text_prompt: str):
    """Blocking SAM3 text-prompt segmentation (runs on GPU thread)."""
    _device_type = "cuda" if "cuda" in _DEVICE else "cpu"
    dtype_context = torch.autocast(_device_type, dtype=torch.bfloat16)

    with dtype_context:
        inference_state = _PROCESSOR.set_image(pil_image)
        output = _PROCESSOR.set_text_prompt(state=inference_state, prompt=text_prompt)

    masks_tensor = output.get("masks")
    boxes_tensor = output.get("boxes")
    scores_tensor = output.get("scores")

    if masks_tensor is None or boxes_tensor is None:
        return SegmentResponse(results=[])

    masks_np = _to_numpy(masks_tensor)
    boxes_np = _to_numpy(boxes_tensor)
    scores_np = _to_numpy(scores_tensor)

    # Squeeze masks if needed: (N, 1, H, W) -> (N, H, W)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np.squeeze(1)

    results_data = []
    num_preds = len(scores_np)

    for i in range(num_preds):
        mask = masks_np[i] > 0  # Boolean mask
        box = boxes_np[i].tolist()  # [x1, y1, x2, y2]
        score = float(scores_np[i])

        results_data.append(
            MaskData(
                mask_base64=encode_mask(mask),
                shape=mask.shape,
                box=box,
                score=score,
                label=text_prompt,
            )
        )

    # Sort by score descending
    results_data.sort(key=lambda x: x.score, reverse=True)

    return SegmentResponse(results=results_data)


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    if _PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    pil_image = decode_image(req.image_base64)

    try:
        return await _run_on_gpu(_do_segment, pil_image, req.text_prompt)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


def _do_segment_point(pil_image: Image.Image, point_coords_tuple: tuple[float, float]):
    """Blocking SAM3 point-prompt segmentation (runs on GPU thread)."""
    _device_type = "cuda" if "cuda" in _DEVICE else "cpu"
    dtype_context = torch.autocast(_device_type, dtype=torch.bfloat16)

    with dtype_context:
        inference_state = _PROCESSOR.set_image(pil_image)
        point_coords = np.array([list(point_coords_tuple)], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int64)  # foreground point
        masks, scores, _ = _MODEL.predict_inst(
            inference_state,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    masks_np = np.asarray(masks)
    scores_np = np.asarray(scores)

    if masks_np.size == 0 or scores_np.size == 0:
        return PointPromptResponse(
            scores=[],
            masks_base64="",
            masks_shape=(0, 0, 0),
            masks_dtype="float32",
        )

    # Sort by score descending
    sort_idx = np.argsort(scores_np)[::-1]
    masks_np = masks_np[sort_idx]
    scores_np = scores_np[sort_idx]

    return PointPromptResponse(
        scores=scores_np.astype(float).tolist(),
        masks_base64=encode_array(masks_np),
        masks_shape=tuple(masks_np.shape),
        masks_dtype=str(masks_np.dtype),
    )


@app.post("/segment_point", response_model=PointPromptResponse)
async def segment_point(req: PointPromptRequest):
    if _PROCESSOR is None or _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    if getattr(_MODEL, "inst_interactive_predictor", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Instance interactivity not enabled on SAM3 model",
        )

    pil_image = decode_image(req.image_base64)

    try:
        return await _run_on_gpu(_do_segment_point, pil_image, req.point_coords)
    except Exception as e:
        logger.error(f"Point prompt inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Point prompt inference failed: {e}")


def _download_checkpoint(hf_repo: str, hf_token: str | None) -> str:
    """Download the SAM checkpoint from HuggingFace and return the local path.

    SAM 3.1 (``facebook/sam3.1``) is a gated model — you must accept the
    contact-information agreement on https://huggingface.co/facebook/sam3.1
    and then supply a token via ``--hf-token`` or the ``HF_TOKEN`` env var.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    token = hf_token or os.environ.get("HF_TOKEN")
    if hf_repo not in _HF_REPOS:
        raise ValueError(
            f"Unknown HF repo '{hf_repo}'. "
            f"Supported: {list(_HF_REPOS)}"
        )
    ckpt_name, cfg_name = _HF_REPOS[hf_repo]

    if hf_repo == "facebook/sam3.1" and token is None:
        logger.warning(
            "SAM 3.1 is a gated model. You must agree to the contact-info terms at "
            "https://huggingface.co/facebook/sam3.1 and supply a token via "
            "--hf-token or the HF_TOKEN environment variable. "
            "Attempting download without token — this will fail if you haven't agreed."
        )

    logger.info("Downloading %s from %s ...", ckpt_name, hf_repo)
    # Download config first (small, validates access early)
    hf_hub_download(repo_id=hf_repo, filename=cfg_name, token=token)
    checkpoint_path = hf_hub_download(repo_id=hf_repo, filename=ckpt_name, token=token)
    logger.info("Checkpoint cached at %s", checkpoint_path)
    return checkpoint_path


def main(
    device: str = "cuda",
    port: int = 8114,
    host: str = "127.0.0.1",
    hf_repo: str = "facebook/sam3",
    hf_token: str | None = None,
    checkpoint_path: str | None = None,
):
    """Launch the SAM3 / SAM 3.1 segmentation server.

    Args:
        device: CUDA device string, e.g. "cuda" or "cuda:1".
        port: HTTP port to bind.
        host: Host address to bind.
        hf_repo: HuggingFace repo to download the checkpoint from.
            Use "facebook/sam3" (default) for SAM 3 or "facebook/sam3.1"
            for SAM 3.1 (gated — requires --hf-token or HF_TOKEN env var).
        hf_token: HuggingFace access token for gated models (SAM 3.1).
            Can also be set via the HF_TOKEN environment variable.
        checkpoint_path: Path to an already-downloaded checkpoint file.
            Overrides --hf-repo / --hf-token if provided.
    """
    global _MODEL, _PROCESSOR, _DEVICE, _HF_REPO

    _DEVICE = device
    _HF_REPO = hf_repo

    # Setup torch settings for Ampere+ GPUs as recommended
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set the default CUDA device so all tensor allocations inside
        # build_sam3_image_model and Sam3Processor use the correct device.
        # Without this, internal tensors default to cuda:0 while model
        # weights are on the specified device, causing device mismatch errors.
        if "cuda" in device:
            device_idx = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_device(device_idx)

    # Resolve checkpoint: explicit path > HF download
    if checkpoint_path is None:
        checkpoint_path = _download_checkpoint(hf_repo, hf_token)

    logger.info("Loading SAM model from %s (repo: %s) ...", checkpoint_path, hf_repo)
    try:
        _MODEL = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            enable_inst_interactivity=True,
            device=device,
        )
    except Exception as e:
        logger.error("Error building SAM model: %s", e)
        raise

    _PROCESSOR = Sam3Processor(_MODEL, device=device, confidence_threshold=0.0)
    logger.info("SAM model (%s) loaded on %s. Starting server ...", hf_repo, device)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
