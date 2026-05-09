from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from controller_launch_utils import (  # noqa: E402
    declare_python_executable_argument,
    python_node_with_preflight,
)


def generate_launch_description():
    policy_path = os.path.join(
        get_package_share_directory('deploy_policy'),
        'policy/go2w/rough/exported/policy.pt'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "policy_path",
            default_value=policy_path,
            description="path to the policy file"),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="True",
            description="Use simulation (Omniverse Isaac Sim) clock if true"),
        DeclareLaunchArgument(
            "max_cmd_vel_x",
            default_value="0.20",
            description="Debug-safe absolute limit for cmd_vel linear.x before policy inference."),
        DeclareLaunchArgument(
            "max_cmd_vel_y",
            default_value="0.10",
            description="Debug-safe absolute limit for cmd_vel linear.y before policy inference."),
        DeclareLaunchArgument(
            "max_cmd_vel_yaw",
            default_value="0.30",
            description="Debug-safe absolute limit for cmd_vel angular.z before policy inference."),
        DeclareLaunchArgument(
            "max_leg_delta",
            default_value="0.35",
            description="Maximum leg joint position delta from default pose produced by one policy command."),
        DeclareLaunchArgument(
            "max_wheel_velocity",
            default_value="5.0",
            description="Maximum absolute wheel joint velocity command."),
        DeclareLaunchArgument(
            "hold_without_cmd_vel",
            default_value="true",
            description="Hold default pose and zero wheel velocity until a recent cmd_vel is received."),
        DeclareLaunchArgument(
            "cmd_vel_timeout_sec",
            default_value="0.75",
            description="Seconds after the last cmd_vel before returning to hold/default command."),
        declare_python_executable_argument(),
        python_node_with_preflight(
            package='deploy_policy',
            executable='go2w_controller.py',
            name='go2w_controller',
            output="screen",
            required_modules=("rclpy", "torch"),
            parameters=[{
                'policy_path': LaunchConfiguration('policy_path'),
                "use_sim_time": ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                "max_cmd_vel_x": ParameterValue(LaunchConfiguration('max_cmd_vel_x'), value_type=float),
                "max_cmd_vel_y": ParameterValue(LaunchConfiguration('max_cmd_vel_y'), value_type=float),
                "max_cmd_vel_yaw": ParameterValue(LaunchConfiguration('max_cmd_vel_yaw'), value_type=float),
                "max_leg_delta": ParameterValue(LaunchConfiguration('max_leg_delta'), value_type=float),
                "max_wheel_velocity": ParameterValue(LaunchConfiguration('max_wheel_velocity'), value_type=float),
                "hold_without_cmd_vel": ParameterValue(LaunchConfiguration('hold_without_cmd_vel'), value_type=bool),
                "cmd_vel_timeout_sec": ParameterValue(LaunchConfiguration('cmd_vel_timeout_sec'), value_type=float),
            }],
        ),
    ])
