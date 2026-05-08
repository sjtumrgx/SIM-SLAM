#!/usr/bin/env python3
"""Isaac Sim runner for Go2W ROS2 + RTX LiDAR FAST-LIO2 experiments.

This script is intentionally conservative: it follows the existing
`scripts/test_core/sim.py` ActionGraph style, adds IMU and RTX LiDAR publishing,
and keeps all names aligned with `deploy_policy` defaults.

Run from an Isaac shell, not a plain ROS shell.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ros2_bridge_env import Ros2BridgeEnvironmentError, ensure_ros2_bridge_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument(
        "--skip-ros2-env-check",
        action="store_true",
        help="Skip the early Isaac ROS 2 Bridge environment preflight",
    )
    parser.add_argument("--scene", default="assets/Map/robocon2026.usd", help="USD scene path")
    parser.add_argument("--robot", default="assets/Go2W/go2w_ros2.usd", help="Go2W USD path")
    parser.add_argument("--robot-prim", default="/World/Go2W", help="Robot prim path")
    parser.add_argument("--scene-prim", default="/World/Robocon2026Map", help="Scene prim path")
    parser.add_argument("--lidar-prim", default="/World/Go2W/base/lidar", help="RTX LiDAR prim path")
    parser.add_argument(
        "--lidar-translation",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.20),
        metavar=("X", "Y", "Z"),
        help="Local LiDAR mount translation relative to --lidar-prim parent",
    )
    parser.add_argument(
        "--imu-prim",
        default="/World/Go2W/base/trunk/imu_link/Imu_Sensor",
        help="IsaacImuSensor prim read by IsaacReadIMU before publishing /imu",
    )
    parser.add_argument(
        "--articulation-prim",
        default="/World/Go2W/base",
        help="Go2W articulation root prim used by joint-state publishers and the articulation controller",
    )
    parser.add_argument("--points-topic", default="points_raw", help="Raw RTX LiDAR PointCloud2 topic")
    parser.add_argument(
        "--partial-scan-pointcloud",
        action="store_true",
        help=(
            "Publish per-render partial RTX LiDAR point-cloud slices. "
            "Default is full-scan buffering, which is required for FAST-LIO2."
        ),
    )
    parser.add_argument("--imu-topic", default="imu", help="ROS Imu topic published from the Isaac IMU sensor")
    parser.add_argument("--lidar-frame", default="lidar_link", help="LiDAR frame id")
    parser.add_argument("--base-frame", default="base_link", help="ROS TF frame name for the robot root")
    parser.add_argument("--imu-frame", default="imu_link", help="ROS TF frame name for the IMU prim")
    parser.add_argument("--tf-parent-prim", default="/World", help="Parent prim for Isaac ROS2PublishTransformTree")
    parser.add_argument("--scan-rate", type=float, default=10.0, help="Configured RTX LiDAR scan rate")
    parser.add_argument(
        "--no-viewer-setup",
        action="store_true",
        help="Do not add the default viewer light/camera framing helpers",
    )
    parser.add_argument(
        "--viewer-light-prim",
        default="/World/ViewerLight",
        help="Dome light prim created when the referenced USDs do not provide lights",
    )
    parser.add_argument(
        "--viewer-light-intensity",
        type=float,
        default=2000.0,
        help="Intensity for the fallback viewer dome light",
    )
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        default=(6.0, -8.0, 4.0),
        metavar=("X", "Y", "Z"),
        help="Initial GUI viewport camera position",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.7),
        metavar=("X", "Y", "Z"),
        help="Initial GUI viewport camera target",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many simulation steps; 0 means run until Isaac Sim exits",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _abs_usd(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _repo_root() / candidate
    return str(candidate)


@contextmanager
def _runner_stage(name: str):
    """Print coarse progress markers so Isaac shutdown logs show the failing step."""

    print(f"[RUNNER] BEGIN {name}", flush=True)
    try:
        yield
    except Ros2BridgeEnvironmentError:
        # Let the top-level handler print the actionable bridge setup message
        # without duplicating it with an internal traceback.
        raise
    except Exception as exc:
        print(f"[RUNNER] ERROR {name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    else:
        print(f"[RUNNER] END {name}", flush=True)


def _enable_ros2_bridge(extensions) -> None:
    last_error = None
    for ext_name in ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"):
        try:
            extensions.enable_extension(ext_name)
            return
        except Exception as exc:  # pragma: no cover - Isaac version dependent
            last_error = exc
    raise RuntimeError(f"Unable to enable Isaac ROS2 bridge extension: {last_error}")


def _enable_physics_sensors(extensions) -> None:
    last_error = None
    for ext_name in ("isaacsim.sensors.physics", "omni.isaac.sensor"):
        try:
            extensions.enable_extension(ext_name)
            return
        except Exception as exc:  # pragma: no cover - Isaac version dependent
            last_error = exc
    raise RuntimeError(f"Unable to enable Isaac physics sensor extension: {last_error}")


def _create_rtx_lidar(lidar_prim: str, scan_rate: float, translation: tuple[float, float, float]) -> str:
    import omni.kit.commands
    from pxr import Gf, Sdf, UsdGeom
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
                translation=Gf.Vec3d(*translation),
            )
            if not success:
                raise RuntimeError("IsaacSensorCreateRtxLidar returned success=False")
        except Exception as exc:
            raise RuntimeError(
                "Could not create RTX LiDAR. Verify installed Isaac version exposes "
                "IsaacSensorCreateRtxLidar and Example_Rotary config."
            ) from exc
    xformable = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(*translation))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))

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


def _find_first_imu_sensor(stage, robot_prim: str) -> str | None:
    robot_prefix = robot_prim.rstrip("/")
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path != robot_prefix and not prim_path.startswith(robot_prefix + "/"):
            continue
        if prim.GetTypeName() == "IsaacImuSensor":
            return prim_path
    return None


def _ensure_imu_sensor(stage, requested_imu_prim: str, robot_prim: str, articulation_prim: str) -> str:
    """Return a valid IsaacImuSensor prim path, creating one only if needed.

    Isaac Sim 5.1's ``ROS2PublishImu`` node no longer accepts a ``targetPrim``
    input. The IMU sensor prim must instead be read by ``IsaacReadIMU`` and the
    resulting orientation/velocity/acceleration values wired into
    ``ROS2PublishImu``.
    """
    import omni.kit.commands

    requested = stage.GetPrimAtPath(requested_imu_prim)
    if requested.IsValid():
        if requested.GetTypeName() != "IsaacImuSensor":
            raise RuntimeError(
                f"--imu-prim must point to an IsaacImuSensor prim; "
                f"{requested_imu_prim} is {requested.GetTypeName()!r}."
            )
        return requested_imu_prim

    discovered = _find_first_imu_sensor(stage, robot_prim)
    if discovered:
        print(
            f"[INFO] Requested IMU prim {requested_imu_prim} was not found; "
            f"using existing IsaacImuSensor {discovered}.",
            flush=True,
        )
        return discovered

    parent = stage.GetPrimAtPath(articulation_prim)
    if not parent.IsValid():
        raise RuntimeError(
            f"Cannot create fallback IMU sensor: articulation prim {articulation_prim} does not exist."
        )

    success, prim = omni.kit.commands.execute(
        "IsaacSensorCreateImuSensor",
        path="/Imu_Sensor",
        parent=articulation_prim,
        sensor_period=0.0,
        linear_acceleration_filter_size=1,
        angular_velocity_filter_size=1,
        orientation_filter_size=1,
    )
    if not success or prim is None:
        raise RuntimeError(
            "Could not create fallback Isaac IMU sensor. Verify the articulation "
            f"prim {articulation_prim} is a rigid body/articulation root."
        )
    imu_prim = str(prim.GetPath())
    print(f"[INFO] Created fallback IsaacImuSensor at {imu_prim}.", flush=True)
    return imu_prim


def _disable_referenced_ros_graphs(stage, robot_prim: str) -> list[str]:
    """Disable ROS graphs embedded in go2w_ros2.usd to avoid duplicate topics.

    The Go2W USD already contains ROS IMU and joint-state graphs. The runner
    owns a canonical runtime graph so it can pin Isaac-version-specific schema
    choices and frame names in one place.
    """
    graph_root = robot_prim.rstrip("/") + "/Graph"
    disabled = []
    for graph_name in ("ROS_IMU", "ROS_Joint_States"):
        graph_path = f"{graph_root}/{graph_name}"
        prim = stage.GetPrimAtPath(graph_path)
        if prim.IsValid():
            prim.SetActive(False)
            disabled.append(graph_path)
    if disabled:
        print(
            "[INFO] Disabled referenced Go2W ROS graphs and using runtime /ActionGraph: "
            + ", ".join(disabled),
            flush=True,
        )
    return disabled


def _rtx_lidar_pointcloud_writer_name(full_scan: bool) -> str:
    """Return the Isaac ROS2 RTX LiDAR writer name for partial or full scans."""

    suffix = "PublishPointCloudBuffer" if full_scan else "PublishPointCloud"
    return "RtxLidar" + "ROS2" + suffix


def _attach_rtx_lidar_pointcloud_writer(
    lidar_prim_path: str,
    topic_name: str,
    frame_id: str,
    *,
    full_scan: bool,
) -> object:
    """Attach the official Replicator RTX LiDAR ROS2 PointCloud2 writer.

    NVIDIA's Isaac Sim RTX LiDAR ROS2 helper uses the buffer writer for full
    scans. FAST-LIO2 expects each PointCloud2 to be a complete scan; publishing
    per-render partial slices makes the adapter schema-valid but breaks the
    downstream mapping semantics.
    """
    import omni.replicator.core as rep

    render_product = rep.create.render_product(
        lidar_prim_path,
        (128, 128),
        name="IsaacFastLioRtxLidar",
        render_vars=["GenericModelOutput", "RtxSensorMetadata"],
    )
    writer = rep.writers.get(_rtx_lidar_pointcloud_writer_name(full_scan))
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


def _stage_has_light(stage) -> bool:
    for prim in stage.Traverse():
        if prim.IsActive() and prim.GetTypeName().endswith("Light"):
            return True
    return False


def _ensure_viewer_light(stage, light_prim: str, intensity: float) -> bool:
    """Create a fallback light when loaded USD assets do not contain one.

    The Robocon map asset is shared with Isaac Lab tasks that add lights from
    their scene config. This standalone ROS2 runner loads the raw USDs directly,
    so it must provide its own light or RaytracedLighting can render a black
    viewport even when geometry is loaded correctly.
    """

    if _stage_has_light(stage):
        return False

    from pxr import UsdLux

    light = UsdLux.DomeLight.Define(stage, light_prim)
    light.CreateIntensityAttr(float(intensity))
    print(
        f"[INFO] Added fallback viewer DomeLight at {light_prim} "
        f"(intensity={float(intensity):g}).",
        flush=True,
    )
    return True


def _set_initial_viewer_camera(
    camera_eye: tuple[float, float, float],
    camera_target: tuple[float, float, float],
) -> None:
    """Aim the active GUI viewport at the loaded map/robot.

    This is best-effort: in headless runs there may be no active viewport, so
    the helper logs a warning and returns without affecting simulation/ROS.
    """

    from isaacsim.core.utils import viewports

    viewports.set_camera_view(np.array(camera_eye, dtype=float), np.array(camera_target, dtype=float))


def _requires_rendering_step() -> bool:
    """RTX LiDAR data is produced by the render pipeline, even in headless mode."""

    return True


def _imu_read_gravity_enabled() -> bool:
    """FAST-LIO2 initialization needs raw accelerometer measurements including gravity."""

    return True


def _imu_use_latest_data_enabled() -> bool:
    """Read the latest physics IMU sample so gravity is available on every tick."""

    return True


def _build_action_graph(args, imu_prim_path: str) -> None:
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
                ("ReadImu", "isaacsim.sensors.physics.IsaacReadIMU"),
                ("PublishImu", "isaacsim.ros2.bridge.ROS2PublishImu"),
                ("PublishTf", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            "connect": [
                ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "ReadImu.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishTf.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishImu.inputs:context"),
                ("Context.outputs:context", "PublishTf.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTf.inputs:timeStamp"),
                ("ReadImu.outputs:execOut", "PublishImu.inputs:execIn"),
                ("ReadImu.outputs:sensorTime", "PublishImu.inputs:timeStamp"),
                ("ReadImu.outputs:angVel", "PublishImu.inputs:angularVelocity"),
                ("ReadImu.outputs:linAcc", "PublishImu.inputs:linearAcceleration"),
                ("ReadImu.outputs:orientation", "PublishImu.inputs:orientation"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            "set_values": [
                ("ArticulationController.inputs:robotPath", args.articulation_prim),
                ("PublishJointState.inputs:topicName", "joint_states"),
                ("SubscribeJointState.inputs:topicName", "joint_command"),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(args.articulation_prim)]),
                ("ReadImu.inputs:imuPrim", [usdrt.Sdf.Path(imu_prim_path)]),
                ("ReadImu.inputs:readGravity", _imu_read_gravity_enabled()),
                ("ReadImu.inputs:useLatestData", _imu_use_latest_data_enabled()),
                ("PublishImu.inputs:topicName", args.imu_topic),
                ("PublishImu.inputs:frameId", args.imu_frame),
                ("PublishTf.inputs:parentPrim", usdrt.Sdf.Path(args.tf_parent_prim)),
                (
                    "PublishTf.inputs:targetPrims",
                    [
                        usdrt.Sdf.Path(args.articulation_prim),
                        usdrt.Sdf.Path(imu_prim_path),
                        usdrt.Sdf.Path(args.lidar_prim),
                    ],
                ),
            ],
        },
    )


def main() -> int:
    args = parse_args()
    if not args.skip_ros2_env_check:
        with _runner_stage("ros2_bridge_environment_preflight"):
            ensure_ros2_bridge_environment(
                "python scripts/ros2/isaac_fast_lio2_go2w_scene.py "
                f"--scene {args.scene} --robot {args.robot} --scan-rate {args.scan_rate}"
            )
    with _runner_stage("import_simulation_app"):
        from isaacsim import SimulationApp

    with _runner_stage("start_simulation_app"):
        simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": args.headless})
    try:
        with _runner_stage("import_isaac_modules"):
            import omni.graph.core as og
            import omni.usd
            from isaacsim.core.api import SimulationContext
            from isaacsim.core.utils import extensions, prims, rotations, stage
            from pxr import Gf

        with _runner_stage("enable_ros2_bridge_extension"):
            _enable_ros2_bridge(extensions)

        with _runner_stage("enable_physics_sensor_extension"):
            _enable_physics_sensors(extensions)

        with _runner_stage("create_simulation_context"):
            simulation_context = SimulationContext(
                stage_units_in_meters=1.0,
                physics_dt=1 / 120.0,
                rendering_dt=1 / 60.0,
            )

        with _runner_stage("load_scene_reference"):
            stage.add_reference_to_stage(_abs_usd(args.scene), args.scene_prim)

        with _runner_stage("load_robot_reference"):
            prims.create_prim(
                args.robot_prim,
                "Xform",
                position=np.array([0.0, -0.64, 0.35]),
                orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
                usd_path=_abs_usd(args.robot),
            )

        with _runner_stage("create_lidar_and_frame_aliases"):
            usd_stage = omni.usd.get_context().get_stage()
            _disable_referenced_ros_graphs(usd_stage, args.robot_prim)
            lidar_prim_path = _create_rtx_lidar(args.lidar_prim, args.scan_rate, tuple(args.lidar_translation))
            imu_prim_path = _ensure_imu_sensor(usd_stage, args.imu_prim, args.robot_prim, args.articulation_prim)
            _set_name_override(usd_stage, args.robot_prim, args.base_frame)
            _set_name_override(usd_stage, args.articulation_prim, args.base_frame)
            _set_name_override(usd_stage, imu_prim_path, args.imu_frame)
            _set_name_override(usd_stage, args.lidar_prim, args.lidar_frame)

        if not args.no_viewer_setup:
            with _runner_stage("configure_viewer_light_and_camera"):
                _ensure_viewer_light(usd_stage, args.viewer_light_prim, args.viewer_light_intensity)
                if not args.headless:
                    _set_initial_viewer_camera(tuple(args.camera_eye), tuple(args.camera_target))

        with _runner_stage("attach_rtx_lidar_pointcloud_writer"):
            rtx_lidar_handles = _attach_rtx_lidar_pointcloud_writer(
                lidar_prim_path,
                args.points_topic,
                args.lidar_frame,
                full_scan=not args.partial_scan_pointcloud,
            )
            print(f"[INFO] RTX LiDAR PointCloud2 writer attached: {rtx_lidar_handles}", flush=True)

        with _runner_stage("build_ros2_action_graph"):
            _build_action_graph(args, imu_prim_path)

        with _runner_stage("initialize_physics"):
            simulation_context.initialize_physics()

        with _runner_stage("play_simulation"):
            simulation_context.play()
            # Let physics sensors produce their first valid sample before the
            # ROS ActionGraph starts publishing. Without this warm-up the IMU
            # stream can get stuck at zero linear acceleration in Isaac 5.1,
            # which prevents FAST-LIO2 from initializing gravity correctly.
            simulation_context.step(render=_requires_rendering_step())

        print("[RUNNER] ENTER simulation_loop", flush=True)
        steps = 0
        while simulation_app.is_running():
            simulation_context.step(render=_requires_rendering_step())
            og.Controller.set(og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True)
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        print(f"[RUNNER] EXIT simulation_loop: steps={steps}", flush=True)
        with _runner_stage("stop_simulation_context"):
            simulation_context.stop()
        return 0
    finally:
        with _runner_stage("close_simulation_app"):
            simulation_app.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Ros2BridgeEnvironmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
