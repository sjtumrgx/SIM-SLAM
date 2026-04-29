import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "launch"))

from controller_launch_utils import format_preflight_error, probe_python_runtime


def test_probe_python_runtime_accepts_standard_library_module():
    probe = probe_python_runtime(sys.executable, ("json",))

    assert probe["modules"]["json"]["ok"], probe
    assert probe["executable"]
    assert probe["version"]


def test_preflight_error_explains_rclpy_abi_mismatch():
    probe = {
        "executable": "/data1/anaconda3/envs/env_isaaclab/bin/python3",
        "version": "3.11.15",
        "prefix": "/data1/anaconda3/envs/env_isaaclab",
    }
    failures = {
        "rclpy": {
            "ok": False,
            "spec": "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/__init__.py",
            "error": "ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'",
        }
    }

    message = format_preflight_error(probe, failures)

    assert "ABI mismatch" in message
    assert "CPython 3.10" in message
    assert "Python 3.11" in message
    assert "uv venv" in message
    assert "--torch-backend cu128" in message
    assert "python_executable" in message
