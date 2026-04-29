cmake_minimum_required(VERSION 3.8)
project(deploy_policy)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(message_filters REQUIRED)

install(DIRECTORY
  config
  launch
  policy
  rviz
  scripts
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)

# Install Python executables
install(PROGRAMS
  scripts/armdog_controller.py
  scripts/check_pointcloud_timing.py
  scripts/go2_controller.py
  scripts/go2w_controller.py
  scripts/isaac_pointcloud_time_adapter.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  scripts/pointcloud_timing_core.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
