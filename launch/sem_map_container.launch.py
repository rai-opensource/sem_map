# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os
import pathlib


def launch_setup(context, *args, **kwargs):
    actions = []

    container_name = LaunchConfiguration('container_name').perform(context)
    use_sim_time_lc = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot').perform(context)

    pkg_share = get_package_share_directory('sem_map')
    sem_map_cfg = os.path.join(pkg_share, 'config', 'sem_map.yaml')

    # Build parameter list: base config + optional robot override
    param_files = [sem_map_cfg]
    if robot_name:
        robot_cfg = os.path.join(pkg_share, 'config', 'robots', f'{robot_name}.yaml')
        if not os.path.isfile(robot_cfg):
            available = [p.stem for p in pathlib.Path(pkg_share, 'config', 'robots').glob('*.yaml')]
            raise RuntimeError(
                f"Robot config not found: {robot_cfg}\n"
                f"Available robots: {', '.join(sorted(available))}"
            )
        param_files.append(robot_cfg)

    sem_map_node = ComposableNode(
        package='sem_map',
        plugin='sem_map::SemMapNode',
        name='sem_map_node',
        namespace='sem_map',
        parameters=[*param_files, {'use_sim_time': use_sim_time_lc}],
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    load_nodes = LoadComposableNodes(
        composable_node_descriptions=[sem_map_node],
        target_container=f"/{container_name}" if not container_name.startswith('/') else container_name,
    )

    actions.append(load_nodes)
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'container_name',
            default_value='zed/zed_container',
            description='Name of an already-running component container to load sem_map into.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time from /clock.',
            choices=['true', 'false'],
        ),
        DeclareLaunchArgument(
            'robot',
            default_value='',
            description='Robot config name (e.g., grandtour_real). '
                        'Loads config/robots/{robot}.yaml as override.'
        ),
        OpaqueFunction(function=launch_setup),
    ])
