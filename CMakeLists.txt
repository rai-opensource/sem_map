# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
cmake_minimum_required(VERSION 3.8)
project(sem_map)

set(CMAKE_CXX_STANDARD 17)
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find all packages
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS imgcodecs imgproc)
find_package(Sophus REQUIRED)

set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 3.10 COMPONENTS Interpreter Development REQUIRED)

# Common dependencies for all Python-embedding targets
set(PYTHON_DEPS ament_index_cpp rclcpp rclcpp_components)

# Main sem_map component (now using pybind11)
add_library(sem_map_component SHARED src/sem_map_node.cpp src/sem_map_bindings.cpp)
target_include_directories(sem_map_component PUBLIC include ${OpenCV_INCLUDE_DIRS} ${SOPHUS_INCLUDE_DIRS} PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(sem_map_component pybind11::embed Python3::Python ${OpenCV_LIBS})
ament_target_dependencies(sem_map_component rclcpp sensor_msgs geometry_msgs nav_msgs diagnostic_msgs std_msgs std_srvs grid_map_msgs rclcpp_components tf2 tf2_ros tf2_sensor_msgs tf2_geometry_msgs ament_index_cpp)
target_compile_options(sem_map_component PRIVATE "-fvisibility=hidden")

rclcpp_components_register_nodes(
  sem_map_component
  PLUGIN "sem_map::SemMapNode"
  EXECUTABLE sem_map_node
)
install(TARGETS sem_map_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch/
        DESTINATION share/${PROJECT_NAME}/launch)

install(DIRECTORY config/
        DESTINATION share/${PROJECT_NAME}/config)

install(DIRECTORY include/
        DESTINATION include)

# Install Python scripts package for runtime imports
install(DIRECTORY scripts/
        DESTINATION share/${PROJECT_NAME}/scripts)

# Install Python executable scripts
install(PROGRAMS scripts/sem_map/sem_map_ros.py
        DESTINATION lib/${PROJECT_NAME})

ament_package()
