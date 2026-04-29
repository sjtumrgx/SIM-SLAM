#!/usr/bin/env python3
"""Check installed Isaac Sim exposes ROS2/RTX LiDAR nodes needed by the plan.

Run from the Isaac shell, e.g. `isaaclab.sh -p scripts/ros2/check_isaac_ros2_node_schema.py`.
"""
from __future__ import annotations

import json
import sys

REQUIRED_NODE_TYPE_CANDIDATES = {
    "ros2_context": [
        "isaacsim.ros2.bridge.ROS2Context",
        "omni.isaac.ros2_bridge.ROS2Context",
    ],
    "publish_clock": [
        "isaacsim.ros2.bridge.ROS2PublishClock",
        "omni.isaac.ros2_bridge.ROS2PublishClock",
    ],
    "publish_joint_state": [
        "isaacsim.ros2.bridge.ROS2PublishJointState",
        "omni.isaac.ros2_bridge.ROS2PublishJointState",
    ],
    "subscribe_joint_state": [
        "isaacsim.ros2.bridge.ROS2SubscribeJointState",
        "omni.isaac.ros2_bridge.ROS2SubscribeJointState",
    ],
    "publish_imu": [
        "isaacsim.ros2.bridge.ROS2PublishImu",
        "omni.isaac.ros2_bridge.ROS2PublishImu",
    ],
    "publish_transform_tree": [
        "isaacsim.ros2.bridge.ROS2PublishTransformTree",
        "omni.isaac.ros2_bridge.ROS2PublishTransformTree",
    ],
}


def _get_registered_node_types(og):
    for attr in ("get_registered_node_types", "get_node_type_names"):
        fn = getattr(og, attr, None)
        if fn:
            try:
                return set(fn())
            except Exception:
                pass
    # Older Isaac builds do not expose an easy public list in every context.
    return set()


def main() -> int:
    try:
        from isaacsim import SimulationApp
    except Exception as exc:
        print("This checker must run inside an Isaac Sim Python environment.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.utils import extensions
        import omni.graph.core as og

        enabled = []
        for name in ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge", "isaacsim.ros2.nodes"):
            try:
                extensions.enable_extension(name)
                enabled.append(name)
            except Exception:
                pass

        registered = _get_registered_node_types(og)
        writer_ok = False
        writer_error = None
        try:
            import omni.replicator.core as rep

            writer_ok = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud") is not None
        except Exception as exc:
            writer_error = str(exc)
        results = {}
        ok = True
        for requirement, candidates in REQUIRED_NODE_TYPE_CANDIDATES.items():
            available = [candidate for candidate in candidates if not registered or candidate in registered]
            # If the registry is unavailable, mark as unknown rather than pass.
            status = "present" if registered and available else ("unknown" if not registered else "missing")
            if status == "missing":
                ok = False
            results[requirement] = {
                "status": status,
                "candidates": candidates,
                "matched": available,
            }
        payload = {
            "ok": ok and bool(registered) and writer_ok,
            "enabled_extensions": enabled,
            "registry_available": bool(registered),
            "rtx_lidar_pointcloud_writer_available": writer_ok,
            "rtx_lidar_pointcloud_writer_error": writer_error,
            "requirements": results,
            "note": "If registry_available=false, manually verify node type names in the installed Isaac version before implementing runner wiring.",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["ok"] else 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
