#!/usr/bin/env python3
"""Isaac Sim runner for Go2W ROS2 + RTX LiDAR FAST-LIO2 experiments.

This script is intentionally conservative: it follows the existing
`scripts/test_core/sim.py` ActionGraph style, adds IMU and RTX LiDAR publishing,
and keeps all names aligned with `deploy_policy` defaults.

Run from an Isaac shell, not a plain ROS shell.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument("--scene", default="assets/Map/robocon2026.usd", help="USD scene path")
    parser.add_argument("--robot", default="assets/Go2W/go2w_ros2.usd", help="Go2W USD path")
    parser.add_argument("--robot-prim", default="/World/Go2W", help="Robot prim path")
    parser.add_argument("--scene-prim", default="/World/Robocon2026Map", help="Scene prim path")
    parser.add_argument("--lidar-prim", default="/World/Go2W/base/lidar", help="RTX LiDAR prim path")
    parser.add_argument("--imu-prim", default="/World/Go2W/imu", help="IMU prim path for ROS2PublishImu")
    parser.add_argument("--points-topic", default="points_raw", help="Raw RTX LiDAR PointCloud2 topic")
    parser.add_argument("--lidar-frame", default="lidar_link", help="LiDAR frame id")
    parser.add_argument("--base-frame", default="base_link", help="ROS TF frame name for the robot root")
    parser.add_argument("--imu-frame", default="imu_link", help="ROS TF frame name for the IMU prim")
    parser.add_argument("--tf-parent-prim", default="/World", help="Parent prim for Isaac ROS2PublishTransformTree")
    parser.add_argument("--scan-rate", type=float, default=10.0, help="Configured RTX LiDAR scan rate")
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _abs_usd(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _repo_root() / candidate
    return str(candidate)


def _enable_ros2_bridge(extensions) -> None:
    last_error = None
    for ext_name in ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"):
        try:
            extensions.enable_extension(ext_name)
            return
        except Exception as exc:  # pragma: no cover - Isaac version dependent
            last_error = exc
    raise RuntimeError(f"Unable to enable Isaac ROS2 bridge extension: {last_error}")


def _create_rtx_lidar(lidar_prim: str, scan_rate: float) -> str:
    import omni.kit.commands
    from pxr import Sdf
    import omni.usd

    if abs(scan_rate - round(scan_rate)) > 1e-6 or scan_rate <= 0:
        raise ValueError(
            f"RTX LiDAR scanRateBaseHz is a positive integer attribute; got scan_rate={scan_rate}."
        )
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(lidar_prim).IsValid():
        prim = stage.GetPrimAtPath(lidar_prim)
    else:
        parent = str(Path(lidar_prim).parent).replace(".", "/")
        name = Path(lidar_prim).name
        try:
            success, prim = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path=name,
                parent=parent,
                config="Example_Rotary",
            )
            if not success:
                raise RuntimeError("IsaacSensorCreateRtxLidar returned success=False")
        except Exception as exc:
            raise RuntimeError(
                "Could not create RTX LiDAR. Verify installed Isaac version exposes "
                "IsaacSensorCreateRtxLidar and Example_Rotary config."
            ) from exc
    # Required for metadata/timestamp exposure in recent Isaac builds.
    try:
        attr = prim.CreateAttribute("omni:sensor:Core:auxOutputType", Sdf.ValueTypeNames.Token)
        attr.Set("FULL")
    except Exception:
        pass
    try:
        scan_attr = prim.GetAttribute("omni:sensor:Core:scanRateBaseHz")
        if not scan_attr:
            scan_attr = prim.CreateAttribute("omni:sensor:Core:scanRateBaseHz", Sdf.ValueTypeNames.UInt)
        scan_attr.Set(int(round(scan_rate)))
    except Exception as exc:
        raise RuntimeError(
            "Unable to set RTX LiDAR scanRateBaseHz. Use a sensor config whose "
            "scanRateBaseHz matches the adapter/FAST-LIO scan_rate."
        ) from exc
    return str(prim.GetPath())


def _attach_rtx_lidar_pointcloud_writer(lidar_prim_path: str, topic_name: str, frame_id: str) -> object:
    """Attach the official Replicator RTX LiDAR ROS2 PointCloud2 writer.

    NVIDIA's Isaac Sim RTX LiDAR ROS2 tutorial requires a render product for
    each RTX sensor before PointCloud2 can be published. Keeping the writer and
    render product alive avoids relying on an unconnected OmniGraph helper node.
    """
    import omni.replicator.core as rep

    render_product = rep.create.render_product(lidar_prim_path, [1, 1], name="IsaacFastLioRtxLidar")
    writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
    writer.initialize(topicName=topic_name, frameId=frame_id)
    writer.attach([render_product])
    return {"render_product": render_product, "writer": writer}


def _set_name_override(stage, prim_path: str, frame_name: str) -> None:
    """Set Isaac ROS nameOverride so TF/joint publishers use canonical frame IDs."""
    from pxr import Sdf

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    attr = prim.GetAttribute("isaac:nameOverride")
    if not attr:
        attr = prim.CreateAttribute("isaac:nameOverride", Sdf.ValueTypeNames.String)
    attr.Set(frame_name)


def _build_action_graph(args) -> None:
    import omni.graph.core as og
    import usdrt.Sdf

    graph_path = "/ActionGraph"
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            "create_nodes": [
                ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("PublishImu", "isaacsim.ros2.bridge.ROS2PublishImu"),
                ("PublishTf", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            "connect": [
                ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishImu.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishTf.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishImu.inputs:context"),
                ("Context.outputs:context", "PublishTf.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishImu.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTf.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            "set_values": [
                ("ArticulationController.inputs:robotPath", args.robot_prim),
                ("PublishJointState.inputs:topicName", "joint_states"),
                ("SubscribeJointState.inputs:topicName", "joint_command"),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(args.robot_prim)]),
                ("PublishImu.inputs:topicName", "imu"),
                ("PublishImu.inputs:frameId", args.imu_frame),
                ("PublishImu.inputs:targetPrim", usdrt.Sdf.Path(args.imu_prim)),
                ("PublishTf.inputs:parentPrim", usdrt.Sdf.Path(args.tf_parent_prim)),
                (
                    "PublishTf.inputs:targetPrims",
                    [usdrt.Sdf.Path(args.robot_prim), usdrt.Sdf.Path(args.imu_prim), usdrt.Sdf.Path(args.lidar_prim)],
                ),
            ],
        },
    )


def main() -> int:
    args = parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": args.headless})
    try:
        import omni.graph.core as og
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.utils import extensions, prims, rotations, stage
        from pxr import Gf

        _enable_ros2_bridge(extensions)
        simulation_context = SimulationContext(stage_units_in_meters=1.0, physics_dt=1 / 120.0, rendering_dt=1 / 60.0)
        stage.add_reference_to_stage(_abs_usd(args.scene), args.scene_prim)
        prims.create_prim(
            args.robot_prim,
            "Xform",
            position=np.array([0.0, -0.64, 0.35]),
            orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
            usd_path=_abs_usd(args.robot),
        )
        usd_stage = omni.usd.get_context().get_stage()
        lidar_prim_path = _create_rtx_lidar(args.lidar_prim, args.scan_rate)
        _set_name_override(usd_stage, args.robot_prim, args.base_frame)
        _set_name_override(usd_stage, args.imu_prim, args.imu_frame)
        _set_name_override(usd_stage, args.lidar_prim, args.lidar_frame)
        rtx_lidar_handles = _attach_rtx_lidar_pointcloud_writer(lidar_prim_path, args.points_topic, args.lidar_frame)
        print(f"[INFO] RTX LiDAR PointCloud2 writer attached: {rtx_lidar_handles}")
        _build_action_graph(args)
        simulation_context.initialize_physics()
        simulation_context.play()
        while simulation_app.is_running():
            simulation_context.step(render=not args.headless)
            og.Controller.set(og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True)
        simulation_context.stop()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
