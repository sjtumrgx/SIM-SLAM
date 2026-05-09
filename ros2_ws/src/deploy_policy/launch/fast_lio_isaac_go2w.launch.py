import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, os.path.dirname(__file__))
from controller_launch_utils import (  # noqa: E402
    declare_python_executable_argument,
    python_node_with_preflight,
)


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare("deploy_policy"), "config", "fast_lio", "isaac_go2w.yaml"
    ])
    default_rviz = PathJoinSubstitution([
        FindPackageShare("deploy_policy"), "rviz", "fast_lio_isaac_go2w.rviz"
    ])
    use_sim_time = LaunchConfiguration("use_sim_time")
    config_file = LaunchConfiguration("config_file")
    rviz = LaunchConfiguration("rviz")
    rviz_cfg = LaunchConfiguration("rviz_cfg")
    publish_static_tf = LaunchConfiguration("publish_static_tf")
    publish_sensor_static_tf = LaunchConfiguration("publish_sensor_static_tf")
    enable_adapter = LaunchConfiguration("enable_adapter")
    input_topic = LaunchConfiguration("input_topic")
    output_topic = LaunchConfiguration("output_topic")
    timestamp_unit = LaunchConfiguration("timestamp_unit")
    lidar_type = LaunchConfiguration("lidar_type")
    scan_rate_hz = LaunchConfiguration("scan_rate_hz")
    scan_rate_int = PythonExpression(["int(float('", scan_rate_hz, "'))"])
    scan_line = LaunchConfiguration("scan_line")
    frame_id = LaunchConfiguration("frame_id")
    derive_time_if_missing = LaunchConfiguration("derive_time_if_missing")
    derive_ring_if_missing = LaunchConfiguration("derive_ring_if_missing")
    derive_intensity_if_missing = LaunchConfiguration("derive_intensity_if_missing")
    filter_invalid_xyz = LaunchConfiguration("filter_invalid_xyz")
    max_abs_coordinate = LaunchConfiguration("max_abs_coordinate")

    adapter_node = python_node_with_preflight(
        package="deploy_policy",
        executable="isaac_pointcloud_time_adapter.py",
        name="isaac_pointcloud_time_adapter",
        output="screen",
        required_modules=("rclpy",),
        parameters=[{
            "input_topic": input_topic,
            "output_topic": output_topic,
            "timestamp_unit": ParameterValue(timestamp_unit, value_type=int),
            "lidar_type": ParameterValue(lidar_type, value_type=int),
            "scan_rate_hz": ParameterValue(scan_rate_hz, value_type=float),
            "derive_time_if_missing": ParameterValue(derive_time_if_missing, value_type=bool),
            "derive_ring_if_missing": ParameterValue(derive_ring_if_missing, value_type=bool),
            "derive_intensity_if_missing": ParameterValue(derive_intensity_if_missing, value_type=bool),
            "filter_invalid_xyz": ParameterValue(filter_invalid_xyz, value_type=bool),
            "max_abs_coordinate": ParameterValue(max_abs_coordinate, value_type=float),
            "scan_line": ParameterValue(scan_line, value_type=int),
            "frame_id": frame_id,
            "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("config_file", default_value=default_config),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("rviz_cfg", default_value=default_rviz),
        DeclareLaunchArgument(
            "publish_static_tf",
            default_value="true",
            description="Publish FAST-LIO fork frame aliases: camera_init/map/odom/body/base_link.",
        ),
        DeclareLaunchArgument(
            "publish_sensor_static_tf",
            default_value="false",
            description=(
                "Also publish base_link->imu_link/lidar_link static aliases. "
                "Keep false for the Isaac runner because its TransformTree publishes the robot/sensor tree."
            ),
        ),
        DeclareLaunchArgument(
            "enable_adapter",
            default_value="true",
            description="Start the Isaac /points_raw -> FAST-LIO /points_fast_lio adapter in this Route A launch.",
        ),
        DeclareLaunchArgument("input_topic", default_value="/points_raw"),
        DeclareLaunchArgument("output_topic", default_value="/points_fast_lio"),
        DeclareLaunchArgument("timestamp_unit", default_value="0"),
        DeclareLaunchArgument("lidar_type", default_value="2"),
        DeclareLaunchArgument("scan_rate_hz", default_value="10"),
        DeclareLaunchArgument("derive_time_if_missing", default_value="true"),
        DeclareLaunchArgument("derive_ring_if_missing", default_value="true"),
        DeclareLaunchArgument("derive_intensity_if_missing", default_value="true"),
        DeclareLaunchArgument("filter_invalid_xyz", default_value="true"),
        DeclareLaunchArgument("max_abs_coordinate", default_value="200.0"),
        DeclareLaunchArgument("scan_line", default_value="32"),
        DeclareLaunchArgument("frame_id", default_value="lidar_link"),
        declare_python_executable_argument(),
        GroupAction(
            actions=[adapter_node],
            condition=IfCondition(enable_adapter),
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="camera_init_to_map_tf",
            arguments=["--frame-id", "camera_init", "--child-frame-id", "map"],
            condition=IfCondition(publish_static_tf),
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_odom_tf",
            arguments=["--frame-id", "map", "--child-frame-id", "odom"],
            condition=IfCondition(publish_static_tf),
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="body_to_base_link_tf",
            arguments=["--frame-id", "body", "--child-frame-id", "base_link"],
            condition=IfCondition(publish_static_tf),
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_link_to_imu_tf",
            arguments=["--frame-id", "base_link", "--child-frame-id", "imu_link"],
            condition=IfCondition(PythonExpression(["'", publish_static_tf, "' == 'true' and '", publish_sensor_static_tf, "' == 'true'"])),
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_link_to_lidar_tf",
            arguments=["--x", "0.0", "--y", "0.0", "--z", "0.20", "--frame-id", "base_link", "--child-frame-id", "lidar_link"],
            condition=IfCondition(PythonExpression(["'", publish_static_tf, "' == 'true' and '", publish_sensor_static_tf, "' == 'true'"])),
            output="screen",
        ),
        Node(
            package="fast_lio",
            executable="fastlio_mapping",
            name="fastlio_mapping",
            parameters=[
                config_file,
                {
                    "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                    "common.lid_topic": output_topic,
                    "preprocess.lidar_type": ParameterValue(lidar_type, value_type=int),
                    "preprocess.timestamp_unit": ParameterValue(timestamp_unit, value_type=int),
                    "preprocess.scan_rate": ParameterValue(scan_rate_int, value_type=int),
                    "preprocess.scan_line": ParameterValue(scan_line, value_type=int),
                },
            ],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_cfg],
            condition=IfCondition(rviz),
            output="screen",
        ),
    ])
