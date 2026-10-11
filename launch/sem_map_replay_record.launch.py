# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Replay a bag through sem_map and record the results to a new bag.

Sequence:
  1. Start the bag recorder (waits for topics, nothing recorded yet).
  2. Start the sem_map container (via sem_map_replay.launch.py).
  3. Start `ros2 bag play` PAUSED with --clock (sim time is available immediately,
     but no data flows while sem_map initializes its GPU workers).
  4. A waiter process blocks until sem_map publishes its first /sem_map/stats
     message (this only happens after the full pipeline, workers included, is up),
     then calls /rosbag2_player/resume.
  5. When playback finishes, wait a grace period for in-flight frames, then shut
     everything down. The recorder finalizes the output bag on SIGINT.

Usage:
  ros2 launch sem_map sem_map_replay_record.launch.py \
      bag:=data/rosbags/.../input.mcap robot:=grandtour_real
Optional:
  output:=/path/to/output_bag   (default: ./recordings/<stem>_semmap_<ts>)
  topics_yaml:=/path/to/topics.yaml  (default: config/ros/record_sem_map_topics.yaml)
  rate:=1.0                     (playback rate)
"""
import os
import datetime
import pathlib

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

# Grace period after playback ends before shutting down, so frames still in the
# GPU pipeline get published and recorded.
SHUTDOWN_GRACE_SEC = 5.0


def _share_file(*rel: str) -> pathlib.Path:
    """A file shipped with the package.

    Prefers the installed share copy; falls back to the source tree when the
    launch file is run directly from a checkout.
    """
    installed = pathlib.Path(get_package_share_directory('sem_map')).joinpath(*rel)
    if installed.is_file():
        return installed
    # --symlink-install / direct source run: <repo>/launch/<this file>
    return pathlib.Path(os.path.realpath(__file__)).parents[1].joinpath(*rel)


def _default_topics_yaml() -> pathlib.Path:
    return _share_file('config', 'ros', 'record_sem_map_topics.yaml')


def launch_setup(context, *args, **kwargs):
    bag_path = LaunchConfiguration('bag').perform(context)
    robot = LaunchConfiguration('robot').perform(context)
    output = LaunchConfiguration('output').perform(context)
    topics_yaml = LaunchConfiguration('topics_yaml').perform(context)
    rate = LaunchConfiguration('rate').perform(context)

    if not bag_path:
        raise RuntimeError("Argument 'bag' is required (input bag to replay)")
    bag_path = os.path.abspath(bag_path)
    if not os.path.exists(bag_path):
        raise RuntimeError(f"Input bag not found: {bag_path}")

    if not topics_yaml:
        topics_yaml = str(_default_topics_yaml())
    if not os.path.isfile(topics_yaml):
        raise RuntimeError(f"Topics YAML not found: {topics_yaml} (pass topics_yaml:=...)")
    with open(topics_yaml) as f:
        topics = (yaml.safe_load(f) or {}).get('topics') or []
    if not topics:
        raise RuntimeError(f"No topics loaded from {topics_yaml}")

    if not output:
        stem = pathlib.Path(bag_path).stem
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output = str(pathlib.Path.cwd() / 'recordings' / f'{stem}_semmap_{stamp}')
    os.makedirs(os.path.dirname(output), exist_ok=True)

    actions = []

    # 1. Recorder first: subscribes and waits, so nothing is missed once data flows.
    #    --use-sim-time stamps receive times in the bag clock domain.
    record_proc = ExecuteProcess(
        cmd=['ros2', 'bag', 'record',
             '--output', output,
             '--storage', 'mcap',
             '--storage-config-file', str(_share_file('config', 'ros', 'mcap_zstd.yaml')),
             '--use-sim-time',
             *topics],
        name='bag_record',
        output='screen',
    )
    actions.append(record_proc)

    # 2. sem_map container (reuses the existing replay launch).
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('sem_map'), 'launch', 'sem_map_replay.launch.py')),
        launch_arguments={'robot': robot, 'use_sim_time': 'true'}.items(),
    ))

    # 3. Player, started PAUSED: /clock is published (frozen) but no data flows
    #    until sem_map is ready.
    play_proc = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '--clock', '-p', '-s', 'mcap',
             '--rate', rate, bag_path],
        name='bag_play',
        output='screen',
    )
    actions.append(play_proc)

    # 4. Waiter: the first /sem_map/stats message is only published after the
    #    node's constructor (including GPU worker startup) has completed.
    # Explicit message type: without it, `topic echo` fails with "Could not
    # determine the type" when it starts before the publisher exists. The
    # retry loop covers transient discovery errors on top of that.
    waiter = ExecuteProcess(
        cmd=['bash', '-c',
             'echo "[waiter] waiting for sem_map to come up..." && '
             'until ros2 topic echo --once /sem_map/stats diagnostic_msgs/msg/DiagnosticArray '
             '> /dev/null 2>&1; do '
             'echo "[waiter] sem_map not up yet, retrying..."; sleep 2; done && '
             'echo "[waiter] sem_map ready, resuming bag playback" && '
             'until ros2 service call /rosbag2_player/resume rosbag2_interfaces/srv/Resume '
             '> /dev/null 2>&1; do sleep 1; done && '
             'echo "[waiter] playback resumed"'],
        name='semmap_waiter',
        output='screen',
    )
    actions.append(waiter)

    # 5. Playback finished -> grace period -> shut everything down.
    #    Shutdown SIGINTs the recorder, which finalizes the output bag.
    actions.append(RegisterEventHandler(OnProcessExit(
        target_action=play_proc,
        on_exit=[
            LogInfo(msg=f'Playback finished; stopping in {SHUTDOWN_GRACE_SEC}s '
                        f'(output bag: {output})'),
            TimerAction(
                period=SHUTDOWN_GRACE_SEC,
                actions=[EmitEvent(event=Shutdown(reason='replay finished'))],
            ),
        ],
    )))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('bag', description='Input bag (.mcap) to replay.'),
        DeclareLaunchArgument('robot', default_value='grandtour_real',
                              description='Robot config name (config/robots/{robot}.yaml).'),
        DeclareLaunchArgument('output', default_value='',
                              description='Output bag path. Default: ./recordings/'
                                          '<input stem>_semmap_<timestamp>.'),
        DeclareLaunchArgument('topics_yaml', default_value='',
                              description='YAML with topics to record. Default: the '
                                          'record_sem_map_topics.yaml installed with the package.'),
        DeclareLaunchArgument('rate', default_value='1.0',
                              description='Playback rate.'),
        OpaqueFunction(function=launch_setup),
    ])
