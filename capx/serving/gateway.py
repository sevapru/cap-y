"""cap-y serving gateway — single entry point for all perception/grasping/motion servers.

Routes requests to backend servers by functional group and name, loaded from
docker/gateway.yaml based on CAPX_PROFILE env var.

Usage:
    python -m capx.serving.gateway --port 8100

Routes:
    GET  /status                          — aggregate health of all profile backends
    ANY  /{group}/{backend}/{path:path}   — proxy to backend server
    GET  /docs                            — Swagger UI (this gateway's API)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load routing config from gateway.yaml
# ---------------------------------------------------------------------------

_GATEWAY_YAML = Path(__file__).parent.parent.parent / "docker" / "gateway.yaml"


_EFFECTIVE_PROFILE: str = "open"


def _load_profile() -> dict[str, dict[str, Any]]:
    """Load backend routing for the active CAPX_PROFILE from gateway.yaml."""
    global _EFFECTIVE_PROFILE
    requested = os.environ.get("CAPX_PROFILE", "open")
    if not _GATEWAY_YAML.exists():
        logger.warning(
            "gateway.yaml not found at %s; using hardcoded open profile", _GATEWAY_YAML
        )
        _EFFECTIVE_PROFILE = "open"
        return {
            "perception": {"sam3": {"port": 8114, "desc": "SAM3"}},
            "motion": {"pyroki": {"port": 8116, "desc": "PyRoKi IK"}},
        }
    with open(_GATEWAY_YAML) as f:
        cfg = yaml.safe_load(f)
    profiles = cfg.get("profiles", {})
    if requested not in profiles:
        logger.warning(
            "CAPX_PROFILE=%r not found in gateway.yaml (available: %s); falling back to 'open'",
            requested,
            list(profiles.keys()),
        )
        _EFFECTIVE_PROFILE = "open"
        return profiles.get("open", {})
    _EFFECTIVE_PROFILE = requested
    return profiles[requested]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="cap-y Gateway",
    description="Reverse proxy aggregating all perception/grasping/motion serving endpoints.",
    version="1.0.0",
)


def _get_backends() -> dict[str, tuple[str, int]]:
    """Return flat mapping of backend_name -> (group, port) for the active profile."""
    profile = _load_profile()
    backends: dict[str, tuple[str, int]] = {}
    for group, members in profile.items():
        for name, info in members.items():
            backends[name] = (group, info["port"])
    return backends


@app.get("/status", summary="Aggregate health of all profile backends")
async def status() -> dict[str, Any]:
    """Check health of every backend registered in the active profile.

    Returns a dict mapping backend name to its health status.
    """
    profile = _load_profile()
    results: dict[str, Any] = {"profile": _EFFECTIVE_PROFILE, "backends": {}}

    async with httpx.AsyncClient(timeout=2.0) as client:
        for group, members in profile.items():
            for name, info in members.items():
                port = info["port"]
                try:
                    r = await client.get(f"http://localhost:{port}/health")
                    health = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    results["backends"][name] = {
                        "group": group,
                        "port": port,
                        "status": health.get("status", "unknown"),
                        "ok": r.status_code == 200,
                        "desc": info.get("desc", ""),
                    }
                except Exception:
                    results["backends"][name] = {
                        "group": group,
                        "port": port,
                        "status": "unreachable",
                        "ok": False,
                        "desc": info.get("desc", ""),
                    }
    return results


@app.api_route(
    "/{group}/{backend}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    summary="Proxy request to a named backend server",
)
async def proxy(group: str, backend: str, path: str, request: Request) -> Response:
    """Proxy a request to the specified backend in the given group.

    Example: POST /grasping/demograsp/grasp → http://localhost:8119/grasp
    """
    profile = _load_profile()
    group_backends = profile.get(group)
    if group_backends is None:
        return JSONResponse(
            {"error": f"Unknown group: {group!r}. Available: {list(profile.keys())}"},
            status_code=404,
        )
    backend_info = group_backends.get(backend)
    if backend_info is None:
        return JSONResponse(
            {"error": f"Unknown backend {backend!r} in group {group!r}. Available: {list(group_backends.keys())}"},
            status_code=404,
        )

    port = backend_info["port"]
    url = f"http://localhost:{port}/{path}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.request(
            method=request.method,
            url=url,
            content=await request.body(),
            headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")},
            params=dict(request.query_params),
        )

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(port: int = 8100, host: str = "0.0.0.0") -> None:
    """Start the gateway server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
