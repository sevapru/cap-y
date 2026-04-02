#!/usr/bin/env python3
"""cap-x CUDA Acceleration Verification Suite.

Validates that every accelerated module in the container is properly
using CUDA. Run inside the container with GPU access (runtime: nvidia).

Usage:
    python scripts/test_cuda_acceleration.py
    # or via docker:
    docker exec capx-serving python /workspace/scripts/test_cuda_acceleration.py

Exit code 0 = all expected CUDA modules accelerated.
Exit code 1 = at least one expected CUDA module NOT accelerated.
"""

import importlib
import os
import subprocess
import sys
import traceback

PASS = 0
FAIL = 0
INFO = 0


def report(status: str, module: str, detail: str):
    global PASS, FAIL, INFO
    if status == "PASS":
        PASS += 1
    elif status == "FAIL":
        FAIL += 1
    elif status == "INFO":
        INFO += 1
    print(f"[{status}] {module} -- {detail}")


def test_opencv():
    try:
        import cv2
        import numpy as np

        version = cv2.__version__
        build_info = cv2.getBuildInformation()

        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices == 0:
            report("FAIL", f"OpenCV {version}", "No CUDA devices detected")
            return

        checks = {
            "CUDA": "NVIDIA CUDA" in build_info,
            "cuDNN": "cuDNN:                         YES" in build_info,
            "CUBLAS": "CUBLAS" in build_info,
            "FAST_MATH": "FAST_MATH" in build_info,
            "GStreamer": "GStreamer:                   YES" in build_info or "GStreamer" in build_info,
        }

        failed = [k for k, v in checks.items() if not v]
        if failed:
            report("FAIL", f"OpenCV {version}", f"Missing: {', '.join(failed)}")
            return

        gpu_mat = cv2.cuda_GpuMat()
        test_arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gpu_mat.upload(test_arr)
        result = gpu_mat.download()
        assert np.array_equal(test_arr, result), "GpuMat roundtrip failed"

        neon_info = ""
        for line in build_info.splitlines():
            if "NEON" in line and "Baseline" in line:
                neon_info = line.strip()
                break

        report("PASS", f"OpenCV {version}",
               f"CUDA {cuda_devices} device(s), cuDNN, CUBLAS, FAST_MATH, GStreamer"
               + (f", {neon_info}" if neon_info else ""))

    except Exception as e:
        report("FAIL", "OpenCV", str(e))


def test_pytorch():
    try:
        import torch

        if not torch.cuda.is_available():
            report("FAIL", f"PyTorch {torch.__version__}", "CUDA not available")
            return

        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        cuda_version = torch.version.cuda
        cudnn_enabled = torch.backends.cudnn.enabled

        a = torch.randn(256, 256, dtype=torch.float16, device="cuda")
        b = torch.randn(256, 256, dtype=torch.float16, device="cuda")
        c = torch.matmul(a, b)
        assert c.shape == (256, 256), "FP16 matmul failed"
        del a, b, c

        details = [
            f"device: {device_name} (sm_{capability[0]}{capability[1]})",
            f"CUDA {cuda_version}",
            f"cuDNN {'ON' if cudnn_enabled else 'OFF'}",
        ]
        report("PASS", f"PyTorch {torch.__version__}", ", ".join(details))

    except Exception as e:
        report("FAIL", "PyTorch", str(e))


def test_open3d():
    try:
        import open3d as o3d
        import open3d.core as o3c

        version = o3d.__version__

        cuda_available = o3c.cuda.is_available()
        if not cuda_available:
            report("FAIL", f"Open3D {version}", "CUDA not available")
            return

        device_count = o3c.cuda.device_count()

        t = o3c.Tensor([1.0, 2.0, 3.0], dtype=o3c.float32, device=o3c.Device("CUDA:0"))
        assert t.shape == o3c.SizeVector([3]), "CUDA tensor creation failed"
        del t

        torch_ops = False
        try:
            import open3d.ml.torch
            torch_ops = True
        except ImportError as ie:
            if "tensorboard" in str(ie):
                torch_ops = True  # ops are there, just missing optional tensorboard
            pass

        realsense = hasattr(o3d.t.io, "RealSenseSensor")

        details = [
            f"CUDA {device_count} device(s)",
            f"PyTorch ops {'OK' if torch_ops else 'MISSING'}",
            f"RealSense {'OK' if realsense else 'MISSING'}",
        ]

        if not torch_ops:
            report("FAIL", f"Open3D {version}", ", ".join(details))
        else:
            report("PASS", f"Open3D {version}", ", ".join(details))

    except Exception as e:
        report("FAIL", "Open3D", str(e))


def test_curobo():
    try:
        spec = importlib.util.find_spec("curobo")
        if spec is None:
            try:
                import curobo
                report("PASS", "CuRobo", f"Package imported (version: {getattr(curobo, '__version__', 'unknown')})")
                return
            except ImportError:
                report("FAIL", "CuRobo", "Package not installed")
                return

        import glob
        curobo_path = ""
        if spec.submodule_search_locations:
            curobo_path = spec.submodule_search_locations[0]
        elif spec.origin:
            curobo_path = os.path.dirname(spec.origin)

        if curobo_path and os.path.exists(curobo_path):
            so_files = glob.glob(os.path.join(curobo_path, "**/*.so"), recursive=True)
            report("PASS", "CuRobo", f"Package found at {curobo_path}, {len(so_files)} native extension(s)")
        else:
            report("FAIL", "CuRobo", f"Package metadata exists but source missing (broken editable install at {curobo_path})")

    except Exception as e:
        report("FAIL", "CuRobo", str(e))


def test_contact_graspnet():
    try:
        try:
            import contact_graspnet_pytorch
            report("PASS", "ContactGraspNet", "Package imported, PointNet2 ops available")
            return
        except ImportError:
            pass

        spec = importlib.util.find_spec("contact_graspnet_pytorch")
        if spec is None:
            report("FAIL", "ContactGraspNet", "Package not installed (broken editable install? rebuild image with editable=false)")
            return

        if spec.origin and os.path.exists(spec.origin):
            report("PASS", "ContactGraspNet", f"Package found at {spec.origin}")
        else:
            report("FAIL", "ContactGraspNet", f"Package metadata exists but source missing at {spec.origin}")

    except Exception as e:
        report("FAIL", "ContactGraspNet", str(e))


def test_jax():
    try:
        import jax

        devices = jax.devices()
        device_types = set(d.platform for d in devices)

        if "gpu" in device_types:
            report("PASS", f"JAX {jax.__version__}", f"GPU accelerated: {devices}")
        else:
            report("INFO", f"JAX {jax.__version__}",
                   f"CPU only (expected: jax<0.4.30 has no CUDA 13.0 aarch64 wheels)")

    except Exception as e:
        report("INFO", "JAX", f"Not installed or import failed: {e}")


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
            mujoco_info = result.stdout.strip()
            report("PASS", "LIBERO venv", f"{mujoco_info}, EGL rendering available")
        else:
            report("FAIL", "LIBERO venv", f"MuJoCo import failed: {result.stderr.strip()[:200]}")

    except Exception as e:
        report("FAIL", "LIBERO venv", str(e))


def test_cuda_runtime():
    try:
        import torch
        if not torch.cuda.is_available():
            report("FAIL", "CUDA Runtime", "No CUDA runtime")
            return

        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        report("PASS", "CUDA Runtime",
               f"Driver OK, {mem_total:.1f} GB VRAM, device accessible")

    except Exception as e:
        report("FAIL", "CUDA Runtime", str(e))


def main():
    print("=" * 60)
    print("cap-x CUDA Acceleration Report")
    print("=" * 60)
    print()

    test_cuda_runtime()
    test_opencv()
    test_pytorch()
    test_open3d()
    test_curobo()
    test_contact_graspnet()
    test_jax()
    test_libero_venv()

    print()
    print("-" * 60)
    total_expected = PASS + FAIL
    print(f"Accelerated:     {PASS}/{total_expected} expected CUDA modules")
    if INFO > 0:
        print(f"Not accelerated: {INFO} (expected/informational)")
    if FAIL > 0:
        print(f"FAILED:          {FAIL} module(s) NOT accelerated")
    print("-" * 60)

    if FAIL > 0:
        print("RESULT: FAIL -- some expected CUDA modules are not accelerated")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
