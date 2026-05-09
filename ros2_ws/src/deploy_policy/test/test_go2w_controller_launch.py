import importlib.util
import sys
from pathlib import Path

from launch.actions import DeclareLaunchArgument, OpaqueFunction


LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch" / "go2w_controller.launch.py"


def _load_launch_module():
    launch_dir = LAUNCH_FILE.parent
    sys.path.insert(0, str(launch_dir))
    spec = importlib.util.spec_from_file_location("go2w_controller_launch", LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.get_package_share_directory = lambda _package: str(LAUNCH_FILE.resolve().parents[1])
    return module


def _text_of(value):
    if hasattr(value, "text"):
        return value.text
    if isinstance(value, tuple):
        return "".join(_text_of(item) for item in value)
    if isinstance(value, list):
        return "".join(_text_of(item) for item in value)
    return str(value)


def _declared_arguments(ld):
    declared = {}
    for action in ld.entities:
        if isinstance(action, DeclareLaunchArgument):
            declared[action.name] = _text_of(getattr(action, "default_value", None))
    return declared


def test_go2w_controller_launch_exposes_safe_motion_limits():
    args = _declared_arguments(_load_launch_module().generate_launch_description())

    assert args["max_cmd_vel_x"] == "0.20"
    assert args["max_cmd_vel_y"] == "0.10"
    assert args["max_cmd_vel_yaw"] == "0.30"
    assert args["max_leg_delta"] == "0.35"
    assert args["max_wheel_velocity"] == "5.0"
    assert args["hold_without_cmd_vel"] == "true"
    assert args["cmd_vel_timeout_sec"] == "0.75"


def test_go2w_controller_launch_uses_preflight_wrapper():
    assert any(
        isinstance(action, OpaqueFunction)
        for action in _load_launch_module().generate_launch_description().entities
    )
