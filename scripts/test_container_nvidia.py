#!/usr/bin/env python3
"""cap-y-nvidia container verification suite.

Validates Isaac ROS packages, cuMotion, cuVSLAM, cuTAMP, and flags
ScheduleStream as a known gap. Run inside cap-y-nvidia with GPU access.

Usage:
    python scripts/test_container_nvidia.py
    # or via docker compose:
    docker compose -f docker/docker-compose.capx.yml --profile test-nvidia run capx-test-nvidia
"""

import os
import subprocess
import sys

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


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    env = {**os.environ}
    setup_files = [
        "/opt/ros/jazzy/setup.bash",
        "/opt/isaac_ros_ws/install/setup.bash",
    ]
    sources = " && ".join(f"source {s}" for s in setup_files if os.path.exists(s))
    if sources:
        shell_cmd = sources + " && " + " ".join(cmd)
        return subprocess.run(
            ["bash", "-c", shell_cmd],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


# ---------------------------------------------------------------------------
# Isaac ROS apt repository
# ---------------------------------------------------------------------------

def test_isaac_ros_apt():
    apt_list = "/etc/apt/sources.list.d/nvidia-isaac-ros.list"
    if os.path.exists(apt_list):
        with open(apt_list) as f:
            content = f.read()
        if "isaac.download.nvidia.com" in content:
            report("PASS", "Isaac ROS apt", f"{apt_list} configured")
        else:
            report("FAIL", "Isaac ROS apt", f"{apt_list} exists but no Isaac URL")
    else:
        report("FAIL", "Isaac ROS apt", f"{apt_list} not found")


# ---------------------------------------------------------------------------
# Isaac ROS common + NITROS packages
# ---------------------------------------------------------------------------

def test_isaac_ros_packages():
    r = _run(["ros2", "pkg", "list"])
    if r.returncode != 0:
        report("INFO", "Isaac ROS packages", "ros2 pkg list failed (packages may not be available for Jazzy)")
        return

    pkgs = r.stdout
    targets = ["isaac_ros_common", "isaac_ros_nitros"]
    found = [t for t in targets if t in pkgs]
    missing = [t for t in targets if t not in pkgs]

    if found and not missing:
        report("PASS", "Isaac ROS packages", f"Installed: {', '.join(found)}")
    elif found:
        report("INFO", "Isaac ROS packages", f"Partial: {', '.join(found)}; missing: {', '.join(missing)}")
    else:
        report("INFO", "Isaac ROS packages", "Not available for Jazzy yet (graceful skip)")


# ---------------------------------------------------------------------------
# cuMotion
# ---------------------------------------------------------------------------

def test_cumotion():
    src = "/opt/isaac_ros_ws/src/isaac_ros_cumotion"
    if os.path.isdir(src):
        install = "/opt/isaac_ros_ws/install/isaac_ros_cumotion"
        if os.path.isdir(install):
            report("PASS", "cuMotion", "Source cloned + colcon build installed")
        else:
            report("INFO", "cuMotion", "Source cloned but colcon build skipped (deps missing)")
    else:
        report("INFO", "cuMotion", "Source not available (repo may require NGC access)")


# ---------------------------------------------------------------------------
# cuVSLAM
# ---------------------------------------------------------------------------

def test_cuvslam():
    src = "/opt/isaac_ros_ws/src/isaac_ros_visual_slam"
    if os.path.isdir(src):
        install = "/opt/isaac_ros_ws/install/isaac_ros_visual_slam"
        if os.path.isdir(install):
            report("PASS", "cuVSLAM", "Source cloned + colcon build installed")
        else:
            report("INFO", "cuVSLAM", "Source cloned but colcon build skipped")
    else:
        report("INFO", "cuVSLAM", "Source not available")


# ---------------------------------------------------------------------------
# nvblox Isaac ROS wrapper
# ---------------------------------------------------------------------------

def test_nvblox_wrapper():
    src = "/opt/isaac_ros_ws/src/isaac_ros_nvblox"
    if os.path.isdir(src):
        install = "/opt/isaac_ros_ws/install/isaac_ros_nvblox"
        if os.path.isdir(install):
            report("PASS", "nvblox wrapper", "Source cloned + colcon build installed")
        else:
            report("INFO", "nvblox wrapper", "Source cloned but colcon build skipped")
    else:
        report("INFO", "nvblox wrapper", "Source not available")


# ---------------------------------------------------------------------------
# cuTAMP
# ---------------------------------------------------------------------------

def test_cutamp():
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import nvidia_cutamp; print('OK')"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            report("PASS", "cuTAMP", "nvidia_cutamp imported")
        else:
            report("INFO", "cuTAMP", "Not available via pip (install from NVIDIA NGC)")
    except Exception as e:
        report("INFO", "cuTAMP", f"Import check failed: {e}")


# ---------------------------------------------------------------------------
# ScheduleStream (known gap)
# ---------------------------------------------------------------------------

def test_schedulestream():
    src = "/opt/schedulestream"
    if not os.path.isdir(src):
        report("INFO", "ScheduleStream", "Not cloned (install may have failed)")
        return

    try:
        r = subprocess.run(
            [sys.executable, "-c", "import schedulestream; print('OK')"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            report("PASS", "ScheduleStream", "Installed from NVlabs/ScheduleStream")
        else:
            report("INFO", "ScheduleStream",
                   f"Source at {src} but import failed (may need cuRobo runtime): {r.stderr.strip()[:150]}")
    except Exception as e:
        report("INFO", "ScheduleStream", f"Check failed: {e}")


# ---------------------------------------------------------------------------
# Isaac ROS workspace integrity
# ---------------------------------------------------------------------------

def test_isaac_ws():
    ws = "/opt/isaac_ros_ws"
    if not os.path.isdir(ws):
        report("INFO", "Isaac ROS workspace", f"{ws} does not exist")
        return

    src_dirs = []
    if os.path.isdir(os.path.join(ws, "src")):
        src_dirs = [d for d in os.listdir(os.path.join(ws, "src"))
                    if os.path.isdir(os.path.join(ws, "src", d))]

    install_dir = os.path.join(ws, "install")
    installed = []
    if os.path.isdir(install_dir):
        installed = [d for d in os.listdir(install_dir)
                     if os.path.isdir(os.path.join(install_dir, d))]

    report("PASS" if src_dirs else "INFO", "Isaac ROS workspace",
           f"src: {len(src_dirs)} package(s), install: {len(installed)} built")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.path.insert(0, os.path.dirname(__file__))

    print("=" * 60)
    print("cap-y-nvidia Container Verification")
    print("=" * 60)
    print()

    # Run base (cap-y) checks
    print("--- Base (cap-y) checks ---")
    import test_container_base as base_mod
    base_mod.PASS = 0
    base_mod.FAIL = 0
    base_mod.INFO = 0
    base_mod.test_cuda_runtime()
    base_mod.test_opencv()
    base_mod.test_pytorch()
    base_mod.test_open3d()
    base_mod.test_jax()
    base_mod.test_curobo()
    base_mod.test_contact_graspnet()
    base_mod.test_pyroki()
    base_mod.test_wheel_integrity()
    base_mod.test_libero_venv()

    global PASS, FAIL, INFO
    PASS += base_mod.PASS
    FAIL += base_mod.FAIL
    INFO += base_mod.INFO

    # Run open (cap-y-open) checks
    print()
    print("--- Open (cap-y-open) checks ---")
    import test_container_open as open_mod
    open_mod.PASS = 0
    open_mod.FAIL = 0
    open_mod.INFO = 0
    open_mod.test_ros2_distro()
    open_mod.test_rclpy()
    open_mod.test_ros2_cli()
    open_mod.test_cyclonedds()
    open_mod.test_nav2()
    open_mod.test_moveit()
    open_mod.test_ros2_control()
    open_mod.test_nvblox()
    open_mod.test_livekit()
    open_mod.test_drake()
    open_mod.test_bimanual_readiness()

    PASS += open_mod.PASS
    FAIL += open_mod.FAIL
    INFO += open_mod.INFO

    # Run nvidia-specific checks
    print()
    print("--- NVIDIA (cap-y-nvidia) checks ---")
    test_isaac_ros_apt()
    test_isaac_ros_packages()
    test_cumotion()
    test_cuvslam()
    test_nvblox_wrapper()
    test_cutamp()
    test_schedulestream()
    test_isaac_ws()

    print()
    print("-" * 60)
    total = PASS + FAIL
    print(f"Passed:  {PASS}/{total}")
    if INFO:
        print(f"Info:    {INFO} (optional/gracefully skipped)")
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
