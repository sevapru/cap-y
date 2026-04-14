#!/usr/bin/env python3
"""Preflight checks before LIBERO / CaP-X benchmark runs (one-button safety net).

Validates Python version, platform support, uv extra conflicts, git submodules,
LIBERO paths, optional packages implied by env YAML (API servers), LLM proxy
reachability, and disk space. Use --strict to also fail on WARN (e.g. missing LLM key or CPU-only PyTorch).

Usage:
    uv run python scripts/benchmark_preflight.py --suite libero --strict
    uv run python scripts/benchmark_preflight.py \\
        --config-path env_configs/libero/franka_libero_cap_agent0.yaml --strict
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import socket
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

PASS = 0
FAIL = 0
INFO = 0
WARN = 0


def report(status: str, module: str, detail: str) -> None:
    global PASS, FAIL, INFO, WARN
    if status == "PASS":
        PASS += 1
    elif status == "FAIL":
        FAIL += 1
    elif status == "INFO":
        INFO += 1
    elif status == "WARN":
        WARN += 1
    print(f"[{status}] {module} -- {detail}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_pyproject() -> dict[str, Any] | None:
    path = _repo_root() / "pyproject.toml"
    if not path.is_file():
        report("FAIL", "pyproject.toml", f"Not found at {path}")
        return None
    raw = path.read_text(encoding="utf-8")
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(raw)
    try:
        import tomli as tomllib  # type: ignore[no-redef, import-not-found]

        return tomllib.loads(raw)
    except ImportError:
        report(
            "WARN",
            "pyproject.toml",
            "Install `tomli` or use Python 3.11+ to parse conflicts and requires-python",
        )
        return None


def _parse_requires_python(spec: str) -> tuple[int, int] | tuple[int, int, int, int] | None:
    """Return (min_major, min_minor) or (min_major, min_minor, max_major, max_minor) upper exclusive."""
    # Examples: ">=3.10, <3.13"
    parts = [p.strip() for p in spec.split(",")]
    lo_maj, lo_min = 3, 10
    hi_maj, hi_min = 99, 99
    ok = False
    for p in parts:
        m = re.match(r">=\s*(\d+)\.(\d+)", p)
        if m:
            lo_maj, lo_min = int(m.group(1)), int(m.group(2))
            ok = True
            continue
        m = re.match(r"<\s*(\d+)\.(\d+)", p)
        if m:
            hi_maj, hi_min = int(m.group(1)), int(m.group(2))
            ok = True
    if not ok:
        return None
    return (lo_maj, lo_min, hi_maj, hi_min)


def check_python_version(data: dict[str, Any] | None) -> None:
    if data is None:
        report("INFO", "Python version", "Skipped (no TOML parse)")
        return
    req = data.get("project", {}).get("requires-python")
    if not req:
        report("INFO", "Python version", "requires-python not set in pyproject")
        return
    parsed = _parse_requires_python(req)
    if not parsed:
        report("WARN", "Python version", f"Could not parse {req!r}")
        return
    lo_maj, lo_min, hi_maj, hi_min = parsed
    vi = sys.version_info
    if (vi.major, vi.minor) < (lo_maj, lo_min):
        report("FAIL", "Python version", f"{vi.major}.{vi.minor} < required {lo_maj}.{lo_min} ({req})")
        return
    if (vi.major, vi.minor) >= (hi_maj, hi_min):
        report(
            "FAIL",
            "Python version",
            f"{vi.major}.{vi.minor} >= exclusive upper bound {hi_maj}.{hi_min} ({req})",
        )
        return
    report("PASS", "Python version", f"{vi.major}.{vi.minor} satisfies {req}")


def check_platform_uv_environments(data: dict[str, Any] | None) -> None:
    """tool.uv.environments only lists linux x86_64 and aarch64."""
    if sys.platform != "linux":
        report("FAIL", "Platform", f"CaP-X uv environments target Linux only; got {sys.platform!r}")
        return
    machine = os.uname().machine
    if machine not in ("x86_64", "aarch64"):
        report(
            "WARN",
            "Platform",
            f"Machine {machine!r} is not x86_64 or aarch64 — wheels/extras may be unsupported",
        )
    else:
        report("PASS", "Platform", f"linux {machine}")


def check_uv_overrides(data: dict[str, Any] | None) -> None:
    if data is None:
        return
    overrides = data.get("tool", {}).get("uv", {}).get("override-dependencies")
    if not overrides:
        return
    report(
        "INFO",
        "uv override-dependencies",
        "Pinned/transitive overrides active — if `uv sync` fails, compare versions below:",
    )
    for line in overrides[:20]:
        if isinstance(line, str):
            print(f"         - {line}")
    if len(overrides) > 20:
        print(f"         ... and {len(overrides) - 20} more (see pyproject.toml)")


def check_uv_conflicts(data: dict[str, Any] | None) -> None:
    if data is None:
        return
    conflicts = data.get("tool", {}).get("uv", {}).get("conflicts")
    if not conflicts:
        report("INFO", "uv conflicts", "None declared")
        return
    report(
        "INFO",
        "uv extras conflicts",
        "Cannot combine these optional dependency groups in one env (use separate venvs):",
    )
    for i, group in enumerate(conflicts, 1):
        if isinstance(group, list):
            labels = []
            for item in group:
                if isinstance(item, dict) and "extra" in item:
                    labels.append(item["extra"])
            if labels:
                print(f"         {i}. mutually exclusive: {', '.join(labels)}")
    report(
        "INFO",
        "LIBERO vs robosuite",
        "docs: use `.venv-libero` with `--extra libero` only — not `robosuite` in the same env",
    )


def check_submodule_paths(data: dict[str, Any] | None) -> None:
    root = _repo_root()
    required: list[tuple[str, Path]] = []

    libero_path = root / "capx/third_party/LIBERO-PRO"
    required.append(("LIBERO-PRO (benchmarks)", libero_path))

    if data:
        sources = data.get("tool", {}).get("uv", {}).get("sources", {})
        for name, spec in sources.items():
            if isinstance(spec, list):
                for entry in spec:
                    if isinstance(entry, dict) and "path" in entry:
                        p = root / entry["path"]
                        required.append((f"uv.sources[{name}] path", p))
            elif isinstance(spec, dict) and "path" in spec:
                p = root / spec["path"]
                required.append((f"uv.sources[{name}] path", p))

    seen: set[Path] = set()
    for label, p in required:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if not rp.is_dir():
            report("FAIL", label, f"Missing or not a directory: {rp}")
            continue
        try:
            nonempty = any(rp.iterdir())
        except OSError as e:
            report("WARN", label, f"Could not scan {rp}: {e}")
            continue
        if not nonempty:
            report("FAIL", label, f"Directory empty (init submodules?): {rp}")
        else:
            report("PASS", label, str(rp))


def check_libero_user_config(suite: str) -> None:
    if suite != "libero":
        return
    cfg_path = Path.home() / ".libero" / "config.yaml"
    if not cfg_path.is_file():
        report(
            "FAIL",
            "LIBERO config",
            f"Missing {cfg_path} — create per docs/libero-tasks.md (headless benchmark paths)",
        )
        return
    try:
        import yaml
    except ImportError:
        report("WARN", "LIBERO config", "PyYAML not installed; cannot validate paths inside config")
        return
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        report("FAIL", "LIBERO config", "config.yaml is not a mapping")
        return
    for key in ("assets", "bddl_files", "benchmark_root", "datasets", "init_states"):
        val = cfg.get(key)
        if not val:
            report("WARN", "LIBERO config", f"Missing or empty key {key!r}")
            continue
        p = Path(str(val).replace("$(pwd)", str(_repo_root()))).expanduser()
        if not p.exists():
            report("FAIL", "LIBERO config", f"{key} -> {p} does not exist")
        else:
            report("PASS", f"LIBERO {key}", str(p))


def check_imports_libero(suite: str) -> None:
    if suite != "libero":
        return
    try:
        from libero import benchmark  # noqa: F401

        d = benchmark.get_benchmark_dict()
        report("PASS", "libero.benchmark", f"{len(d)} suite(s) registered")
    except ImportError as e:
        report(
            "FAIL",
            "libero",
            f"Import failed: {e} — uv sync --extra libero (dedicated venv; conflicts with robosuite extra)",
        )


def check_mujoco_and_egl() -> None:
    try:
        import mujoco

        report("PASS", "MuJoCo", getattr(mujoco, "__version__", "import ok"))
    except ImportError as e:
        report("FAIL", "MuJoCo", str(e))
    egl = os.environ.get("MUJOCO_GL", "")
    if not egl:
        report(
            "WARN",
            "MUJOCO_GL",
            "Unset — headless EGL usually needs MUJOCO_GL=egl (launch.py sets a default when imported)",
        )
    elif egl.lower() != "egl":
        report("INFO", "MUJOCO_GL", f"{egl!r} (egl recommended for headless benchmarks)")
    else:
        report("PASS", "MUJOCO_GL", "egl")


def check_torch_cuda() -> None:
    try:
        import torch
    except ImportError as e:
        report("FAIL", "PyTorch", str(e))
        return
    if torch.cuda.is_available():
        report("PASS", "PyTorch CUDA", f"{torch.__version__}, {torch.cuda.get_device_name(0)}")
    else:
        report(
            "WARN",
            "PyTorch CUDA",
            "CUDA not available — SAM3 / ContactGraspNet / many sim paths will fail",
        )


def check_verl_extra_platform() -> None:
    if sys.platform == "linux" and os.uname().machine != "x86_64":
        report(
            "INFO",
            "verl / molmo extras",
            "pyproject optional-deps for verl, molmo, and flash-attn are x86_64-only — "
            "omit those extras on aarch64 (Jetson)",
        )


def check_decord_platform_limitation() -> None:
    if sys.platform == "linux" and os.uname().machine == "aarch64":
        spec = importlib.util.find_spec("decord")
        if spec is None:
            report(
                "INFO",
                "decord",
                "Not installed (expected on aarch64 — pyproject only lists decord for x86_64)",
            )
        else:
            report("PASS", "decord", "present on aarch64 (nonstandard)")
    else:
        spec = importlib.util.find_spec("decord")
        if spec is None:
            report("WARN", "decord", "Not installed — some dataloaders may fail on x86_64")
        else:
            report("PASS", "decord", "importable")


def check_llm_proxy(server_url: str, do_connect: bool) -> None:
    key_path = _repo_root() / ".openrouterkey"
    if key_path.is_file():
        report("PASS", "LLM credentials", f".openrouterkey present ({key_path})")
    elif os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        report("PASS", "LLM credentials", "API key in environment")
    else:
        report(
            "WARN",
            "LLM credentials",
            "No .openrouterkey or OPENROUTER_API_KEY/OPENAI_API_KEY — launch.py still runs if proxy is elsewhere",
        )

    if not do_connect:
        report("INFO", "LLM server TCP", "Skipped (--no-check-server)")
        return
    parsed = urllib.parse.urlparse(server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=3.0):
            report("PASS", "LLM server TCP", f"{host}:{port} accepts connections")
    except OSError as e:
        report(
            "FAIL",
            "LLM server TCP",
            f"{host}:{port} unreachable ({e}) — start openrouter_server or set --server-url",
        )


def check_disk_space(min_gb: float) -> None:
    out = _repo_root() / "outputs"
    path = out if out.is_dir() else _repo_root()
    try:
        usage = shutil.disk_usage(path)
    except OSError as e:
        report("WARN", "Disk space", str(e))
        return
    free_gb = usage.free / (1024**3)
    if free_gb < min_gb:
        report(
            "FAIL",
            "Disk space",
            f"{free_gb:.1f} GiB free at {path} (minimum --min-disk-gb {min_gb})",
        )
    else:
        report("PASS", "Disk space", f"{free_gb:.1f} GiB free at {path}")


def _module_from_api_target(target: str) -> str:
    # capx.serving.launch_pyroki_server.main -> capx.serving.launch_pyroki_server
    if "." not in target:
        return target
    parts = target.split(".")
    if parts[-1] == "main" and len(parts) >= 2:
        return ".".join(parts[:-1])
    return ".".join(parts[:-1]) if len(parts) > 1 else target


def check_yaml_config(path_str: str | None) -> None:
    if not path_str:
        return
    cfg_path = Path(path_str)
    if not cfg_path.is_file():
        report("FAIL", "config YAML", f"Not found: {cfg_path.resolve()}")
        return
    try:
        import yaml
    except ImportError:
        report("WARN", "config YAML", "PyYAML missing; skipping api_servers inspection")
        return
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        report("FAIL", "config YAML", "root must be a mapping")
        return
    report("PASS", "config YAML", str(cfg_path.resolve()))

    servers = cfg.get("api_servers")
    if not isinstance(servers, list):
        report("INFO", "api_servers", "None or not a list — no auto-started servers to verify")
        return

    for i, srv in enumerate(servers):
        if not isinstance(srv, dict):
            continue
        target = srv.get("_target_")
        port = srv.get("port")
        if not isinstance(target, str):
            continue
        mod = _module_from_api_target(target)
        spec = importlib.util.find_spec(mod)
        if spec is None:
            report("FAIL", f"api_servers[{i}]", f"Cannot resolve module {mod!r} for {target!r}")
        else:
            report("PASS", f"api_servers[{i}]", f"{mod} (port {port})")

    n_workers = cfg.get("num_workers")
    if isinstance(n_workers, int) and n_workers > 1:
        report(
            "INFO",
            "Parallel workers",
            f"num_workers={n_workers} — each process may init MuJoCo/CUDA; ensure enough VRAM",
        )
    if cfg.get("record_video") and not cfg.get("output_dir"):
        report("FAIL", "config", "record_video requires output_dir in YAML")
    elif cfg.get("record_video"):
        report("INFO", "record_video", "Enabled — needs writable output_dir and codec deps (ffmpeg)")


@dataclass
class BenchmarkPreflightArgs:
    """Preflight validation for benchmark-style runs."""

    suite: str = "libero"
    """Benchmark stack: libero (default) or generic (skip libero-only checks)."""

    config_path: str | None = None
    """Optional env YAML — validates file exists and api_server modules resolve."""

    server_url: str = "http://127.0.0.1:8110/chat/completions"
    """Used for TCP reachability check (host:port from URL)."""

    check_server: bool = True
    """Try to open a TCP connection to server_url before running benchmarks."""

    min_disk_gb: float = 5.0
    """Minimum free space (GiB) under outputs/ or repo root."""

    strict: bool = False
    """Exit with code 1 if any WARN (FAIL always exits 1)."""


def main(args: BenchmarkPreflightArgs) -> None:
    global PASS, FAIL, INFO, WARN
    PASS = FAIL = INFO = WARN = 0

    if args.suite not in ("libero", "generic"):
        print(f"Unknown suite {args.suite!r}; use libero or generic", file=sys.stderr)
        sys.exit(2)

    print("=" * 60)
    print("CaP-X benchmark preflight")
    print("=" * 60)
    print()

    data = _load_pyproject()
    check_python_version(data)
    check_platform_uv_environments(data)
    check_uv_overrides(data)
    check_uv_conflicts(data)
    check_submodule_paths(data)
    check_yaml_config(args.config_path)
    check_libero_user_config(args.suite)
    check_imports_libero(args.suite)
    check_mujoco_and_egl()
    check_torch_cuda()
    check_verl_extra_platform()
    check_decord_platform_limitation()
    check_llm_proxy(args.server_url, args.check_server)
    check_disk_space(args.min_disk_gb)

    print()
    print("-" * 60)
    print(f"PASS: {PASS}  FAIL: {FAIL}  WARN: {WARN}  INFO: {INFO}")
    print("-" * 60)
    if FAIL:
        print("RESULT: FAIL — fix items above before one-button benchmark runs")
        sys.exit(1)
    if args.strict and WARN:
        print("RESULT: FAIL (--strict: address WARN lines above)")
        sys.exit(1)
    print("RESULT: OK — no hard failures (review WARN/INFO)")
    sys.exit(0)


if __name__ == "__main__":
    main(tyro.cli(BenchmarkPreflightArgs))
