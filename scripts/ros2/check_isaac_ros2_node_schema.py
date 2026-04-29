#!/usr/bin/env python3
"""Check installed Isaac Sim exposes ROS2/RTX LiDAR nodes needed by the plan.

Run from the Isaac shell, e.g. `isaaclab.sh -p scripts/ros2/check_isaac_ros2_node_schema.py`.
"""
from __future__ import annotations

import argparse
import json
import sys

from ros2_bridge_env import Ros2BridgeEnvironmentError, ensure_ros2_bridge_environment

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
    "read_imu": [
        "isaacsim.sensors.physics.IsaacReadIMU",
        "omni.isaac.sensor.IsaacReadIMU",
    ],
    "publish_transform_tree": [
        "isaacsim.ros2.bridge.ROS2PublishTransformTree",
        "omni.isaac.ros2_bridge.ROS2PublishTransformTree",
    ],
}

REQUIRED_NODE_ATTRIBUTES = {
    "publish_imu": {
        "node_type": "isaacsim.ros2.bridge.ROS2PublishImu",
        "attributes": [
            "inputs:angularVelocity",
            "inputs:context",
            "inputs:execIn",
            "inputs:frameId",
            "inputs:linearAcceleration",
            "inputs:orientation",
            "inputs:timeStamp",
            "inputs:topicName",
        ],
        "note": "Isaac Sim 5.1 publishes IMU from explicit vector/quaternion inputs; do not wire inputs:targetPrim here.",
    },
    "read_imu": {
        "node_type": "isaacsim.sensors.physics.IsaacReadIMU",
        "attributes": [
            "inputs:execIn",
            "inputs:imuPrim",
            "inputs:readGravity",
            "outputs:angVel",
            "outputs:execOut",
            "outputs:linAcc",
            "outputs:orientation",
            "outputs:sensorTime",
        ],
        "note": "Read an IsaacImuSensor prim and connect its outputs to ROS2PublishImu.",
    },
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


def _probe_required_attributes(og):
    keys = og.Controller.Keys
    graph_path = "/IsaacFastLioSchemaCheck"
    results = {}
    node_names = {
        requirement: f"Probe_{requirement}"
        for requirement in REQUIRED_NODE_ATTRIBUTES
    }
    try:
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    (node_names[requirement], spec["node_type"])
                    for requirement, spec in REQUIRED_NODE_ATTRIBUTES.items()
                ]
            },
        )
    except Exception as exc:
        return {
            requirement: {
                "status": "probe_failed",
                "node_type": spec["node_type"],
                "error": str(exc),
                "note": spec["note"],
            }
            for requirement, spec in REQUIRED_NODE_ATTRIBUTES.items()
        }

    for requirement, spec in REQUIRED_NODE_ATTRIBUTES.items():
        node_name = node_names[requirement]
        node_path = f"{graph_path}/{node_name}"
        try:
            node = og.Controller.node(node_path)
            missing = [attr for attr in spec["attributes"] if not node.get_attribute_exists(attr)]
            results[requirement] = {
                "status": "present" if not missing else "missing_attributes",
                "node_type": spec["node_type"],
                "missing_attributes": missing,
                "has_legacy_targetPrim": node.get_attribute_exists("inputs:targetPrim"),
                "note": spec["note"],
            }
        except Exception as exc:
            results[requirement] = {
                "status": "probe_failed",
                "node_type": spec["node_type"],
                "error": str(exc),
                "note": spec["note"],
            }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-ros2-env-check",
        action="store_true",
        help="Skip the early Isaac ROS 2 Bridge environment preflight",
    )
    args = parser.parse_args()
    if not args.skip_ros2_env_check:
        ensure_ros2_bridge_environment("python scripts/ros2/check_isaac_ros2_node_schema.py")

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
        for name in (
            "isaacsim.ros2.bridge",
            "omni.isaac.ros2_bridge",
            "isaacsim.sensors.physics",
            "omni.isaac.sensor",
        ):
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
        attribute_results = _probe_required_attributes(og)
        attributes_ok = all(result["status"] == "present" for result in attribute_results.values())
        payload = {
            "ok": ok and writer_ok and attributes_ok,
            "enabled_extensions": enabled,
            "registry_available": bool(registered),
            "rtx_lidar_pointcloud_writer_available": writer_ok,
            "rtx_lidar_pointcloud_writer_error": writer_error,
            "requirements": results,
            "attribute_requirements": attribute_results,
            "note": "If registry_available=false, manually verify node type names in the installed Isaac version before implementing runner wiring.",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0 if payload["ok"] else 1
    finally:
        app.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Ros2BridgeEnvironmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
