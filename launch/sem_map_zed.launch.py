# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from launch import LaunchDescription
import os
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, TextSubstitution, Command
from launch.substitutions import IfElseSubstitution
from launch.substitutions import AndSubstitution
from launch.substitutions import EqualsSubstitution
from launch.substitutions import NotEqualsSubstitution
from launch_ros.actions import LoadComposableNodes, ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
import sysconfig
import pathlib


def launch_setup(context, *args, **kwargs):
    actions = []

    # Launch configurations
    container_name = LaunchConfiguration('container_name').perform(context)
    camera_model = LaunchConfiguration('camera_model').perform(context)
    camera_name_lc = LaunchConfiguration('camera_name')
    # Keep these as LaunchConfiguration substitutions (no .perform) to preserve types
    use_sim_time_lc = LaunchConfiguration('use_sim_time')
    svo_path_lc = LaunchConfiguration('svo_path')
    publish_urdf_lc = LaunchConfiguration('publish_urdf')
    publish_svo_clock_lc = LaunchConfiguration('publish_svo_clock')
    publish_tf_lc = LaunchConfiguration('publish_tf')
    publish_map_tf_lc = LaunchConfiguration('publish_map_tf')

    # Compute use_sim_time (for sem_map only): true iff publishing SVO clock and not live, else fallback to launch arg
    computed_use_sim_time = IfElseSubstitution(
        AndSubstitution(
            EqualsSubstitution(publish_svo_clock_lc, 'true'),
            NotEqualsSubstitution(svo_path_lc, 'live'),
        ),
        TextSubstitution(text='true'),
        use_sim_time_lc,
    )

    robot_name = LaunchConfiguration('robot').perform(context)

    # Resolve config paths from this package (sem_map)
    pkg_share = get_package_share_directory('sem_map')
    common_cfg = f"{pkg_share}/config/zed/common_stereo.yaml"
    camera_cfg = f"{pkg_share}/config/zed/{camera_model}.yaml"
    sem_map_cfg = f"{pkg_share}/config/sem_map.yaml"

    # Build parameter list: base config + optional robot override
    sem_map_param_files = [sem_map_cfg]
    if robot_name:
        robot_cfg = os.path.join(pkg_share, 'config', 'robots', f'{robot_name}.yaml')
        if not os.path.isfile(robot_cfg):
            available = [p.stem for p in pathlib.Path(pkg_share, 'config', 'robots').glob('*.yaml')]
            raise RuntimeError(
                f"Robot config not found: {robot_cfg}\n"
                f"Available robots: {', '.join(sorted(available))}"
            )
        sem_map_param_files.append(robot_cfg)

    # Container to host both ZED and sem_map components
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

    # Optional Robot State Publisher (URDF)
    rsp_node = Node(
        condition=IfCondition(publish_urdf_lc),
        package='robot_state_publisher',
        namespace=camera_name_lc,
        executable='robot_state_publisher',
        name=[camera_name_lc, TextSubstitution(text='_state_publisher')],
        output='both',
        parameters=[{
            'use_sim_time': publish_svo_clock_lc,
            'robot_description': Command([
                'xacro ', LaunchConfiguration('xacro_path'), ' ',
                'camera_name:=', camera_name_lc, ' ',
                'camera_model:=', TextSubstitution(text=camera_model)
            ])
        }]
    )
    actions.append(rsp_node)

    # ZED camera component (minimal parameters via our local configs)
    zed_node = ComposableNode(
        package='zed_components',
        plugin='stereolabs::ZedCamera',
        name='zed_node',
        namespace=camera_name_lc,
        parameters=[common_cfg, camera_cfg, {
            'use_sim_time': use_sim_time_lc,
            'svo.svo_path': svo_path_lc,
            'svo.publish_svo_clock': publish_svo_clock_lc,
            'general.camera_name': camera_name_lc,
            'general.camera_model': TextSubstitution(text=camera_model),
            'pos_tracking.publish_tf': publish_tf_lc,
            'pos_tracking.publish_map_tf': publish_map_tf_lc,
        }],
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    # sem_map component
    sem_map_node = ComposableNode(
        package='sem_map',
        plugin='sem_map::SemMapNode',
        name='sem_map_node',
        namespace='sem_map',
        parameters=[*sem_map_param_files, {'use_sim_time': computed_use_sim_time}],
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    # Load both components into our container
    load_nodes = LoadComposableNodes(
        composable_node_descriptions=[zed_node, sem_map_node],
        target_container=f"/{container_name}"
    )
    actions.append(load_nodes)

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_name',
            default_value=TextSubstitution(text='zed'),
            description='The name of the camera. Used as the namespace for the ZED node.'
        ),
        DeclareLaunchArgument(
            'container_name',
            default_value='sem_map_container',
            description='Composable node container name.'
        ),
        DeclareLaunchArgument(
            'camera_model',
            default_value='zedxm',
            description='Camera model config to use from sem_map/config/zed (e.g., zedx, zedxm)'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='If set to `true` the node will wait for messages on the `/clock` topic to start and will use this information as the timestamp reference',
            choices=['true', 'false']
        ),
        DeclareLaunchArgument(
            'svo_path',
            default_value=TextSubstitution(text='live'),
            description='Path to an input SVO file or "live" to use the camera.'
        ),
        DeclareLaunchArgument(
            'publish_svo_clock',
            default_value='true',
            description='When playing SVO, publish its time on /clock so nodes can use sim time.',
            choices=['true', 'false']
        ),
        DeclareLaunchArgument(
            'publish_urdf',
            default_value='false',
            description='Enable URDF processing and start Robot State Publisher to propagate static TF.',
            choices=['true', 'false']
        ),
        DeclareLaunchArgument(
            'xacro_path',
            default_value=TextSubstitution(text=(get_package_share_directory('zed_wrapper') + '/urdf/zed_descr.urdf.xacro')),
            description='Path to the camera URDF file as a xacro file.'
        ),
        DeclareLaunchArgument(
            'publish_tf',
            default_value='true',
            description='Enable publication of the `odom -> camera_link` TF.',
            choices=['true', 'false']
        ),
        DeclareLaunchArgument(
            'publish_map_tf',
            default_value='true',
            description='Enable publication of the `map -> odom` TF.',
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
