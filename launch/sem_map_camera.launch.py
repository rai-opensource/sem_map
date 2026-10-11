# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Run SemMap against a camera that is already publishing.

This launches SemMap on its own. It does not start a camera driver, so it works
with any camera that publishes an image and a matching CameraInfo: a USB webcam,
a RealSense, an industrial camera, or a ZED you started yourself.

Start your camera driver first, check the topic names with `ros2 topic list`,
then point SemMap at them:

    ros2 launch sem_map sem_map_camera.launch.py \\
        image_topic:=/camera/color/image_raw \\
        camera_info_topic:=/camera/color/camera_info \\
        image_width:=1280 image_height:=720

For a compressed stream, add `image_is_compressed:=true` and give the
`/compressed` topic.

By default this publishes the image outputs only (segmentation, traversability,
confidence). To also build the 2.5D map, pass a pose and a point cloud:

    ros2 launch sem_map sem_map_camera.launch.py \\
        image_topic:=/camera/color/image_raw \\
        camera_info_topic:=/camera/color/camera_info \\
        pose_topic:=/odom/pose pointcloud_topic:=/lidar/points \\
        world_frame:=odom base_frame:=base_link pose_frame:=base_link

Settings you use often are better kept in a config file: copy
config/robots/grandtour_real.yaml, edit it, and pass `robot:=<name>`. Anything
you also pass on the command line wins over the file.
"""
import os
import pathlib
import sysconfig

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode

# Launch arguments that map straight onto node parameters. Empty means "leave
# whatever the config files set".
_STRING_OVERRIDES = (
    'image_topic',
    'camera_info_topic',
    'pose_topic',
    'pointcloud_topic',
    'world_frame',
    'base_frame',
    'pose_frame',
    'seg_layer',
)
_INT_OVERRIDES = ('image_width', 'image_height')
_BOOL_OVERRIDES = ('image_is_compressed',)


def launch_setup(context, *args, **kwargs):
    container_name = LaunchConfiguration('container_name').perform(context)
    robot_name = LaunchConfiguration('robot').perform(context)
    use_sim_time_lc = LaunchConfiguration('use_sim_time')

    pkg_share = get_package_share_directory('sem_map')
    param_files = [os.path.join(pkg_share, 'config', 'sem_map.yaml')]

    if robot_name:
        robot_cfg = os.path.join(pkg_share, 'config', 'robots', f'{robot_name}.yaml')
        if not os.path.isfile(robot_cfg):
            available = [p.stem for p in pathlib.Path(pkg_share, 'config', 'robots').glob('*.yaml')]
            raise RuntimeError(
                f"Robot config not found: {robot_cfg}\n"
                f"Available robots: {', '.join(sorted(available))}"
            )
        param_files.append(robot_cfg)

    # Command-line overrides are applied after the config files, so they win.
    overrides = {}
    for name in _STRING_OVERRIDES:
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = value
    for name in _INT_OVERRIDES:
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = int(value)
    for name in _BOOL_OVERRIDES:
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = value.lower() == 'true'

    # The 2.5D map needs a point cloud, and fusing into it needs a pose. Match
    # the map settings to what was actually supplied, otherwise the node waits
    # for data that never arrives. A robot config is left alone: if you wrote
    # one, you chose those settings deliberately.
    if 'pointcloud_topic' in overrides:
        overrides['enable_height_map'] = True
        overrides['enable_bev_fusion'] = 'pose_topic' in overrides
    elif not robot_name:
        overrides['enable_height_map'] = False
        overrides['enable_bev_fusion'] = False

    if not overrides.get('image_topic') and not robot_name:
        raise RuntimeError(
            "No camera given. Pass image_topic:= and camera_info_topic:=, or "
            "robot:=<name> to load config/robots/<name>.yaml. "
            "Run `ros2 topic list` to see what your camera driver publishes."
        )

    # The node embeds a Python interpreter, and Python extension modules need
    # libpython's symbols to be globally visible. Without this preload they fail
    # with "undefined symbol: PyTuple_Type".
    container_env = dict(os.environ)
    libdir = sysconfig.get_config_var('LIBDIR')
    ldlibrary = sysconfig.get_config_var('LDLIBRARY')
    python_lib_path = str(pathlib.Path(libdir) / ldlibrary)
    if not pathlib.Path(python_lib_path).exists():
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

    sem_map_node = ComposableNode(
        package='sem_map',
        plugin='sem_map::SemMapNode',
        name='sem_map_node',
        namespace='sem_map',
        parameters=[*param_files, {'use_sim_time': use_sim_time_lc}, overrides],
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    load_nodes = LoadComposableNodes(
        composable_node_descriptions=[sem_map_node],
        target_container=f"/{container_name}",
    )

    return [container, load_nodes]


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'image_topic', default_value='',
            description='Camera image topic, e.g. /camera/color/image_raw.'),
        DeclareLaunchArgument(
            'camera_info_topic', default_value='',
            description='CameraInfo topic that matches image_topic.'),
        DeclareLaunchArgument(
            'image_is_compressed', default_value='',
            description='Set true if image_topic is a CompressedImage.',
            choices=['', 'true', 'false']),
        DeclareLaunchArgument(
            'image_width', default_value='',
            description='Image width in pixels. Must match the camera output.'),
        DeclareLaunchArgument(
            'image_height', default_value='',
            description='Image height in pixels. Must match the camera output.'),
        DeclareLaunchArgument(
            'pose_topic', default_value='',
            description='Robot pose topic. Only needed for the 2.5D map.'),
        DeclareLaunchArgument(
            'pointcloud_topic', default_value='',
            description='LiDAR or depth cloud topic. Only needed for the 2.5D map.'),
        DeclareLaunchArgument(
            'world_frame', default_value='',
            description='Fixed, gravity-aligned frame the map is published in.'),
        DeclareLaunchArgument(
            'base_frame', default_value='',
            description='Robot body frame that point clouds are converted into.'),
        DeclareLaunchArgument(
            'pose_frame', default_value='',
            description='Frame that pose_topic reports.'),
        DeclareLaunchArgument(
            'seg_layer', default_value='',
            description='Model to use, e.g. dinov3s_sem_seg_snow_v1.'),
        DeclareLaunchArgument(
            'robot', default_value='',
            description='Load config/robots/<robot>.yaml. Command-line arguments override it.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use /clock instead of the wall clock. Leave false for a live camera.',
            choices=['true', 'false']),
        DeclareLaunchArgument(
            'container_name', default_value='sem_map_container',
            description='Name of the component container.'),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=launch_setup)])
