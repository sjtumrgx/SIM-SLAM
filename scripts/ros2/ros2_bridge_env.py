#!/usr/bin/env python3
"""Preflight helpers for Isaac Sim's ROS 2 Bridge environment.

Isaac Sim's ROS 2 Bridge loads native RMW libraries when the extension starts.
For pip/conda Isaac Sim installs this usually means either:

1. start the process from a shell that has sourced an external ROS 2 install, or
2. start the process with Isaac's bundled bridge libraries already present in
   ``LD_LIBRARY_PATH``.

Mutating ``LD_LIBRARY_PATH`` after the Python process has started is not a
reliable fix for native library lookup, so these helpers fail early with the
exact relaunch commands instead of letting Isaac Sim spend ~90 seconds starting
and then fail during extension startup.
"""

from __future__ import annotations

import os
from pathlib import Path

SUPPORTED_INTERNAL_RMWS = ("rmw_fastrtps_cpp", "rmw_cyclonedds_cpp")


class Ros2BridgeEnvironmentError(RuntimeError):
    """Raised when Isaac ROS 2 Bridge cannot load with the current env."""


def _path_entries(value: str | None) -> list[Path]:
    return [Path(entry).resolve() for entry in (value or "").split(os.pathsep) if entry]


def _path_contains(path_value: str | None, target: Path) -> bool:
    target = target.resolve()
    return any(entry == target for entry in _path_entries(path_value))


def find_internal_ros2_bridge_lib() -> Path | None:
    """Return Isaac Sim's bundled Humble bridge lib dir when discoverable."""

    try:
        import isaacsim  # type: ignore
    except Exception:
        return None

    package_dir = Path(isaacsim.__file__).resolve().parent
    candidates = [
        package_dir / "exts" / "isaacsim.ros2.bridge" / "humble" / "lib",
        package_dir.parent / "isaacsim" / "exts" / "isaacsim.ros2.bridge" / "humble" / "lib",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def ros2_bridge_environment_ok() -> bool:
    """Return True when the current process env has a plausible ROS 2 setup."""

    # External ROS 2 path: sourcing /opt/ros/humble/setup.* populates this.
    # RMW_IMPLEMENTATION may be unset and still resolve through the ROS 2
    # default, so AMENT_PREFIX_PATH is the least intrusive acceptance signal.
    if os.environ.get("AMENT_PREFIX_PATH"):
        return True

    # Internal Isaac bridge path: Isaac's own startup error requires these env
    # vars before the Python process starts.
    internal_lib = find_internal_ros2_bridge_lib()
    if internal_lib is None:
        return False
    return (
        os.environ.get("ROS_DISTRO") == "humble"
        and os.environ.get("RMW_IMPLEMENTATION") in SUPPORTED_INTERNAL_RMWS
        and _path_contains(os.environ.get("LD_LIBRARY_PATH"), internal_lib)
    )


def format_ros2_bridge_environment_help(command_hint: str | None = None) -> str:
    """Build an actionable relaunch message for Isaac ROS 2 Bridge failures."""

    internal_lib = find_internal_ros2_bridge_lib()
    command = command_hint or "python scripts/ros2/isaac_fast_lio2_go2w_scene.py ..."
    lines = [
        "Isaac ROS 2 Bridge environment is not initialized.",
        "",
        "The current process has neither a sourced external ROS 2 environment "
        "(`AMENT_PREFIX_PATH`) nor Isaac's bundled Humble bridge libraries in "
        "`LD_LIBRARY_PATH`.",
        "",
        "Use one of these before starting Isaac Sim:",
        "",
        "Option A — Isaac Sim bundled Humble bridge (matches the Isaac error log):",
        "  export ROS_DISTRO=humble",
        "  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
    ]
    if internal_lib is not None:
        lines.append(f"  export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:{internal_lib}\"")
    else:
        lines.append(
            "  export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:<path-to-isaacsim>/exts/"
            "isaacsim.ros2.bridge/humble/lib\""
        )
    lines.extend(
        [
            f"  {command}",
            "",
            "Option B — external ROS 2 Humble install:",
            "  source /opt/ros/humble/setup.zsh",
            "  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp  # optional but explicit",
            f"  {command}",
            "",
            "Do not try to fix this after SimulationApp has started; relaunch "
            "the Python process with the environment above.",
        ]
    )
    return "\n".join(lines)


def ensure_ros2_bridge_environment(command_hint: str | None = None) -> None:
    """Raise a clear error if Isaac ROS 2 Bridge env is not ready."""

    if ros2_bridge_environment_ok():
        return
    raise Ros2BridgeEnvironmentError(format_ros2_bridge_environment_help(command_hint))
