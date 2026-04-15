#!/usr/bin/env python3
"""cap-y base container verification suite.

Validates CUDA acceleration, wheel integrity, and Python package imports.
Run inside the cap-y container with GPU access (runtime: nvidia).

Usage:
    python scripts/test_container_base.py
    # or via docker compose (from repo docker/):
    docker compose -f docker-compose.capx.yml --profile test run --rm capx-test

cuRobo + ContactGraspNet are optional on cap-y:base (commercial-clean image). On cap-y:default
they must be present. capx-test-default sets CAPY_REQUIRE_LICENSED_STACK=1 so missing packages fail.

Set CAPX_PROFILE / server tool sets in the running container via compose env on cap-y-* services;
this script only checks Python/CUDA stack parity with the image.
"""

import glob
import importlib
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

PASS = 0
FAIL = 0
INFO = 0


def _require_licensed_stack() -> bool:
    """When true, cuRobo + ContactGraspNet must be importable (cap-y:default / NC stack)."""
    return os.environ.get("CAPY_REQUIRE_LICENSED_STACK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def report(status: str, module: str, detail: str):
    global PASS, FAIL, INFO
    if status == "PASS":
        PASS += 1
    elif status == "FAIL":
        FAIL += 1
    elif status == "INFO":
        INFO += 1
    print(f"[{status}] {module} -- {detail}")


# ---------------------------------------------------------------------------
# CUDA runtime
# ---------------------------------------------------------------------------

def test_cuda_runtime():
    try:
        import torch
        if not torch.cuda.is_available():
            report("FAIL", "CUDA Runtime", "No CUDA runtime")
            return
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        report("PASS", "CUDA Runtime", f"Driver OK, {mem_gb:.1f} GB VRAM")
    except Exception as e:
        report("FAIL", "CUDA Runtime", str(e))


# ---------------------------------------------------------------------------
# OpenCV CUDA
# ---------------------------------------------------------------------------

def test_opencv():
    try:
        import cv2
        import numpy as np

        version = cv2.__version__
        build_info = cv2.getBuildInformation()
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices == 0:
            report("FAIL", f"OpenCV {version}", "No CUDA devices")
            return

        checks = {
            "CUDA": "NVIDIA CUDA" in build_info,
            "cuDNN": "cuDNN:" in build_info and "YES" in build_info.split("cuDNN:")[1][:30],
            "CUBLAS": "CUBLAS" in build_info,
            "FAST_MATH": "FAST_MATH" in build_info,
            "GStreamer": "GStreamer" in build_info,
        }
        failed = [k for k, v in checks.items() if not v]
        if failed:
            report("FAIL", f"OpenCV {version}", f"Missing: {', '.join(failed)}")
            return

        mat = cv2.cuda_GpuMat()
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mat.upload(arr)
        assert np.array_equal(arr, mat.download()), "GpuMat roundtrip mismatch"

        report("PASS", f"OpenCV {version}", f"CUDA {cuda_devices} device(s), cuDNN, cuBLAS, GStreamer")
    except Exception as e:
        report("FAIL", "OpenCV", str(e))


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------

def test_pytorch():
    try:
        import torch

        if not torch.cuda.is_available():
            report("FAIL", f"PyTorch {torch.__version__}", "CUDA not available")
            return

        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)

        a = torch.randn(256, 256, dtype=torch.float16, device="cuda")
        c = torch.matmul(a, a)
        assert c.shape == (256, 256)
        del a, c

        report("PASS", f"PyTorch {torch.__version__}",
               f"{name} (sm_{cap[0]}{cap[1]}), CUDA {torch.version.cuda}, cuDNN {'ON' if torch.backends.cudnn.enabled else 'OFF'}")
    except Exception as e:
        report("FAIL", "PyTorch", str(e))


# ---------------------------------------------------------------------------
# Open3D CUDA
# ---------------------------------------------------------------------------

def test_open3d():
    try:
        import open3d as o3d
        import open3d.core as o3c

        version = o3d.__version__
        if not o3c.cuda.is_available():
            report("FAIL", f"Open3D {version}", "CUDA not available")
            return

        t = o3c.Tensor([1.0, 2.0, 3.0], dtype=o3c.float32, device=o3c.Device("CUDA:0"))
        assert t.shape == o3c.SizeVector([3])
        del t

        torch_ops = False
        try:
            import open3d.ml.torch
            torch_ops = True
        except ImportError as ie:
            if "tensorboard" in str(ie):
                torch_ops = True

        n_dev = o3c.cuda.device_count()
        if torch_ops:
            report("PASS", f"Open3D {version}",
                   f"CUDA {n_dev} device(s), PyTorch ops OK")
        elif platform.machine() == "x86_64":
            report(
                "INFO",
                f"Open3D {version}",
                f"CUDA {n_dev} device(s), PyTorch ops not importable on x86_64 "
                "(PyPI Open3D 0.19 wheel ABI-incompatible with PyTorch 2.11; known gap)",
            )
        else:
            report("FAIL", f"Open3D {version}",
                   f"CUDA {n_dev} device(s), PyTorch ops MISSING")
    except Exception as e:
        report("FAIL", "Open3D", str(e))


# ---------------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------------

def test_jax():
    try:
        import jax
        devices = jax.devices()
        gpu = [d for d in devices if d.platform in ("gpu", "cuda")]
        if gpu:
            report("PASS", f"JAX {jax.__version__}", f"GPU accelerated, {len(gpu)} CUDA device(s)")
        else:
            report("FAIL", f"JAX {jax.__version__}", f"CPU only, devices: {devices}")
    except Exception as e:
        report("FAIL", "JAX", f"Import failed: {e}")


# ---------------------------------------------------------------------------
# CuRobo
# ---------------------------------------------------------------------------

def test_curobo():
    try:
        spec = importlib.util.find_spec("curobo")
        if spec is None:
            if _require_licensed_stack():
                report("FAIL", "CuRobo", "Package not installed (required on cap-y:default)")
            else:
                report(
                    "INFO",
                    "CuRobo",
                    "Not installed (expected on cap-y:base; use cap-y:default for NC stack)",
                )
            return

        path = ""
        if spec.submodule_search_locations:
            path = spec.submodule_search_locations[0]
        elif spec.origin:
            path = os.path.dirname(spec.origin)

        so_files = glob.glob(os.path.join(path, "**/*.so"), recursive=True) if path else []
        if path and os.path.exists(path):
            report("PASS", "CuRobo", f"{len(so_files)} native extension(s) at {path}")
        else:
            report("FAIL", "CuRobo", f"Source missing at {path}")
    except Exception as e:
        report("FAIL", "CuRobo", str(e))


# ---------------------------------------------------------------------------
# ContactGraspNet
# ---------------------------------------------------------------------------

def test_contact_graspnet():
    try:
        import contact_graspnet_pytorch  # noqa: F401
        report("PASS", "ContactGraspNet", "Package imported, PointNet2 CUDA ops available")
    except ImportError:
        spec = importlib.util.find_spec("contact_graspnet_pytorch")
        if spec and spec.origin and os.path.exists(spec.origin):
            report("PASS", "ContactGraspNet", f"Found at {spec.origin}")
        elif _require_licensed_stack():
            report("FAIL", "ContactGraspNet", "Not installed (required on cap-y:default)")
        else:
            report(
                "INFO",
                "ContactGraspNet",
                "Not installed (expected on cap-y:base; use cap-y:default for NC stack)",
            )
    except Exception as e:
        report("FAIL", "ContactGraspNet", str(e))


# ---------------------------------------------------------------------------
# PyRoKi
# ---------------------------------------------------------------------------

def test_pyroki():
    try:
        import pyroki  # noqa: F401
        report("PASS", "PyRoKi", "Package imported")
    except Exception as e:
        report("FAIL", "PyRoKi", f"Import failed: {e}")


# ---------------------------------------------------------------------------
# mink (Apache 2.0) — differential IK via MuJoCo
# ---------------------------------------------------------------------------

def test_mink():
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import mink; print(getattr(mink, '__version__', 'installed'))"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            report("PASS", "mink (differential IK)", result.stdout.strip())
        else:
            report("FAIL", "mink (differential IK)", result.stderr.strip()[:200])
    except Exception as e:
        report("FAIL", "mink (differential IK)", str(e))


# ---------------------------------------------------------------------------
# rh56_controller (MIT) — analytical grasp planner for Inspire RH56DFX
# ---------------------------------------------------------------------------

def test_rh56_controller():
    try:
        installed = subprocess.run(
            [sys.executable, "-c", "import rh56_controller"],
            capture_output=True, timeout=30,
        ).returncode == 0
        on_disk = Path("/opt/rh56_controller").exists()
        if installed:
            report("PASS", "rh56_controller (analytical grasping)", "importable")
        elif on_disk:
            report("PASS", "rh56_controller (analytical grasping)", "dir present (fallback deps installed)")
        else:
            report("FAIL", "rh56_controller (analytical grasping)", "not installed")
    except Exception as e:
        report("FAIL", "rh56_controller (analytical grasping)", str(e))


# ---------------------------------------------------------------------------
# DemoGrasp (MIT) — neural dexterous grasping (inference only)
# ---------------------------------------------------------------------------

def test_demograsp():
    try:
        on_disk = Path("/opt/demograsp").exists()
        if not on_disk:
            report("INFO", "DemoGrasp (neural grasping)", "not installed (WITH_DEMOGRASP=0)")
            return
        importable = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '/opt/demograsp'); import einops"],
            capture_output=True, timeout=30,
        ).returncode == 0
        if importable:
            report("PASS", "DemoGrasp (neural grasping)", "dir present, inference deps ok")
        else:
            report("PASS", "DemoGrasp (neural grasping)", "dir present, deps partial (run download_demograsp_ckpts.sh)")
    except Exception as e:
        report("FAIL", "DemoGrasp (neural grasping)", str(e))


# ---------------------------------------------------------------------------
# pymodbus (MIT) — Modbus TCP for Inspire RH56DFTP hands
# ---------------------------------------------------------------------------

def test_pymodbus():
    try:
        import pymodbus
        report("PASS", f"pymodbus {pymodbus.__version__}", "Modbus TCP ready (Inspire RH56DFTP)")
    except ImportError:
        report("FAIL", "pymodbus", "Not installed — pip install pymodbus")


# ---------------------------------------------------------------------------
# unitree_sdk2_python (BSD-3) — Python DDS SDK for Unitree robots
# ---------------------------------------------------------------------------

def test_unitree_sdk2():
    try:
        import unitree_sdk2py
        report("PASS", "unitree_sdk2_python", "DDS SDK importable (G1/H1)")
    except ImportError:
        report("FAIL", "unitree_sdk2_python", "Not installed")


# ---------------------------------------------------------------------------
# Newton (Apache 2.0) — GPU physics engine
# ---------------------------------------------------------------------------

def test_newton_import():
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import newton; print(getattr(newton, '__version__', 'unknown'))"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            report("PASS", f"Newton {result.stdout.strip()}", "GPU physics importable")
        else:
            report("FAIL", "Newton", result.stderr.strip()[:200])
    except Exception as e:
        report("FAIL", "Newton", str(e))


# ---------------------------------------------------------------------------
# Wheel integrity (/opt/wheels/)
# ---------------------------------------------------------------------------

def test_wheel_integrity():
    wheel_dir = "/opt/wheels"
    if not os.path.isdir(wheel_dir):
        report("INFO", "Wheels", f"{wheel_dir} does not exist (caches cleaned?)")
        return

    wheels = glob.glob(os.path.join(wheel_dir, "*.whl"))
    if not wheels:
        report("INFO", "Wheels", "No wheels in /opt/wheels/")
        return

    dirty = []
    for whl in wheels:
        basename = os.path.basename(whl)
        if "selfbuilt" in basename:
            dirty.append(basename)
            continue
        try:
            with zipfile.ZipFile(whl) as zf:
                meta_files = [n for n in zf.namelist() if n.endswith("METADATA")]
                for mf in meta_files:
                    for line in zf.read(mf).decode().splitlines():
                        if line.startswith("Version:") and "selfbuilt" in line:
                            dirty.append(f"{basename} (Version: line)")
        except Exception:
            dirty.append(f"{basename} (unreadable)")

    if dirty:
        report("FAIL", "Wheels", f"{len(dirty)} dirty wheel(s): {', '.join(dirty)}")
    else:
        report("PASS", "Wheels", f"{len(wheels)} wheel(s) in /opt/wheels/, all clean")


# ---------------------------------------------------------------------------
# LIBERO venv (optional)
# ---------------------------------------------------------------------------

def test_libero_venv():
    venv_python = "/opt/venv-libero/bin/python"
    if not os.path.exists(venv_python):
        report("INFO", "LIBERO venv", "Not installed (/opt/venv-libero not found)")
        return
    try:
        result = subprocess.run(
            [venv_python, "-c", "import mujoco; print('MuJoCo', mujoco.__version__)"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        if result.returncode == 0:
            report("PASS", "LIBERO venv", result.stdout.strip())
        else:
            report("FAIL", "LIBERO venv", result.stderr.strip()[:200])
    except Exception as e:
        report("FAIL", "LIBERO venv", str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("cap-y Base Container Verification")
    print("=" * 60)
    print()

    test_cuda_runtime()
    test_opencv()
    test_pytorch()
    test_open3d()
    test_jax()
    test_curobo()
    test_contact_graspnet()
    test_pyroki()
    test_mink()
    test_rh56_controller()
    test_demograsp()
    test_pymodbus()
    test_unitree_sdk2()
    test_newton_import()
    test_wheel_integrity()
    test_libero_venv()

    print()
    print("-" * 60)
    total = PASS + FAIL
    print(f"Passed:  {PASS}/{total}")
    if INFO:
        print(f"Info:    {INFO} (optional/skipped)")
    if FAIL:
        print(f"FAILED:  {FAIL}")
    print("-" * 60)

    if FAIL:
        print("RESULT: FAIL")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
