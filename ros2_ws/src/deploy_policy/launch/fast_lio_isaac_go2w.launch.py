from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("config_file", default_value=default_config),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("rviz_cfg", default_value=default_rviz),
        DeclareLaunchArgument(
            "publish_static_tf",
            default_value="true",
            description="Publish FAST-LIO fork frame aliases: camera_init/map/odom/body/base_link/sensor frames.",
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
            condition=IfCondition(publish_static_tf),
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_link_to_lidar_tf",
            arguments=["--x", "0.0", "--y", "0.0", "--z", "0.20", "--frame-id", "base_link", "--child-frame-id", "lidar_link"],
            condition=IfCondition(publish_static_tf),
            output="screen",
        ),
        Node(
            package="fast_lio",
            executable="fastlio_mapping",
            name="fastlio_mapping",
            parameters=[config_file, {"use_sim_time": use_sim_time}],
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
