from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
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
        'policy/armdog/rough/exported/policy.pt'
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
        declare_python_executable_argument(),
        python_node_with_preflight(
            package='deploy_policy',
            executable='armdog_controller.py',
            name='armdog_controller',
            output="screen",
            required_modules=("rclpy", "torch"),
            parameters=[{
                'policy_path': LaunchConfiguration('policy_path'),
                "use_sim_time": LaunchConfiguration('use_sim_time'),
            }],
        ),
    ])
