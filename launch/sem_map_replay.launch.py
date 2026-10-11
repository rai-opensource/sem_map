# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import LoadComposableNodes, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os
import sysconfig
import pathlib


def launch_setup(context, *args, **kwargs):
    actions = []

    # Launch configurations
    container_name = LaunchConfiguration('container_name').perform(context)
    use_sim_time_lc = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot').perform(context)

    # Resolve config path from this package (sem_map)
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

    # Container environment with Python preload
    container_env = dict(os.environ)

    # Preload libpython to make Python symbols globally visible
    # This fixes "undefined symbol: PyTuple_Type" errors in Python extension modules
    # Dynamically detect the Python library path from the current interpreter
    libdir = sysconfig.get_config_var('LIBDIR')
    ldlibrary = sysconfig.get_config_var('LDLIBRARY')
    python_lib_path = str(pathlib.Path(libdir) / ldlibrary)

    # Try the versioned .so first
    if not pathlib.Path(python_lib_path).exists():
        # Fall back to .so.1.0 version
        python_lib_path = f"{python_lib_path}.1.0"

    if 'LD_PRELOAD' in container_env:
        container_env['LD_PRELOAD'] = f"{python_lib_path}:{container_env['LD_PRELOAD']}"
    else:
        container_env['LD_PRELOAD'] = python_lib_path

    # Container hosting the sem_map component
    container = ComposableNodeContainer(
        name=container_name,
        namespace='',
        package='rclcpp_components',
        executable='component_container_isolated',
        output='screen',
        arguments=['--use_multi_threaded_executor', '--ros-args', '--log-level', 'info'],
        env=container_env,
        composable_node_descriptions=[],
    )
    actions.append(container)

    # sem_map component only
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
        target_container=f"/{container_name}"
    )
    actions.append(load_nodes)

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'container_name',
            default_value='sem_map_container',
            description='Composable node container name.'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time from /clock.',
            choices=['true', 'false']
        ),
        DeclareLaunchArgument(
            'robot',
            default_value='',
            description='Robot config name (e.g., grandtour_real). '
                        'Loads config/robots/{robot}.yaml as override.'
        ),
        OpaqueFunction(function=launch_setup)
    ])
