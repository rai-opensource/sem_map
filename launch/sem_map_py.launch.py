#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Launch file for sem_map_ros node.

Launches only the unified semantic mapping node with configuration from sem_map.yaml.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for unified semantic mapping node."""

    # Package directories
    pkg_share = FindPackageShare('sem_map')

    # Declare arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'sem_map.yaml']),
        description='Path to configuration YAML file'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug, info, warn, error, fatal)'
    )

    # Get launch configurations
    config_file = LaunchConfiguration('config_file')
    log_level = LaunchConfiguration('log_level')

    # Unified semantic mapping node
    sem_map_node = Node(
        package='sem_map',
        executable='sem_map_ros.py',
        name='sem_map_node',
        output='screen',
        parameters=[config_file, {'use_sim_time': True}],
        arguments=['--ros-args', '--log-level', log_level],
        emulate_tty=True,
    )

    return LaunchDescription([
        config_file_arg,
        log_level_arg,
        sem_map_node,
    ])
