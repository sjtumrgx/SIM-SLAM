from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from controller_launch_utils import (  # noqa: E402
    declare_python_executable_argument,
    python_node_with_preflight,
)


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("input_topic", default_value="/points_raw"),
        DeclareLaunchArgument("output_topic", default_value="/points_fast_lio"),
        DeclareLaunchArgument("timestamp_unit", default_value="0"),
        DeclareLaunchArgument("lidar_type", default_value="2"),
        DeclareLaunchArgument("scan_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("derive_time_if_missing", default_value="false"),
        DeclareLaunchArgument("derive_ring_if_missing", default_value="false"),
        DeclareLaunchArgument("scan_line", default_value="32"),
        DeclareLaunchArgument("frame_id", default_value="lidar_link"),
        declare_python_executable_argument(),
        python_node_with_preflight(
            package="deploy_policy",
            executable="isaac_pointcloud_time_adapter.py",
            name="isaac_pointcloud_time_adapter",
            output="screen",
            required_modules=("rclpy",),
            parameters=[{
                "input_topic": LaunchConfiguration("input_topic"),
                "output_topic": LaunchConfiguration("output_topic"),
                "timestamp_unit": LaunchConfiguration("timestamp_unit"),
                "lidar_type": LaunchConfiguration("lidar_type"),
                "scan_rate_hz": LaunchConfiguration("scan_rate_hz"),
                "derive_time_if_missing": LaunchConfiguration("derive_time_if_missing"),
                "derive_ring_if_missing": LaunchConfiguration("derive_ring_if_missing"),
                "scan_line": LaunchConfiguration("scan_line"),
                "frame_id": LaunchConfiguration("frame_id"),
                "use_sim_time": True,
            }],
        ),
    ])
