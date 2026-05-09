cmake_minimum_required(VERSION 3.8)
project(deploy_policy)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(message_filters REQUIRED)
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
endif()

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

if(BUILD_TESTING)
  ament_add_pytest_test(test_pointcloud_timing_core test/test_pointcloud_timing_core.py)
  ament_add_pytest_test(test_docs_fast_lio_go2w_debug test/test_docs_fast_lio_go2w_debug.py)
  ament_add_pytest_test(test_isaac_pointcloud_time_adapter test/test_isaac_pointcloud_time_adapter.py)
  ament_add_pytest_test(test_controller_launch_utils test/test_controller_launch_utils.py)
  ament_add_pytest_test(test_go2w_controller_sync test/test_go2w_controller_sync.py)
  ament_add_pytest_test(test_go2w_controller_launch test/test_go2w_controller_launch.py)
  ament_add_pytest_test(test_fast_lio_isaac_go2w_launch test/test_fast_lio_isaac_go2w_launch.py)
  ament_add_pytest_test(test_fast_lio_isaac_go2w_rviz test/test_fast_lio_isaac_go2w_rviz.py)
  ament_add_pytest_test(test_isaac_fast_lio2_runner_config test/test_isaac_fast_lio2_runner_config.py)
endif()

ament_package()
