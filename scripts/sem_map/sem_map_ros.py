#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
ROS 2 node for semantic mapping pipeline.

Subscribes to:
- Pose topic (geometry_msgs/PoseStamped)
- Pointcloud topic (sensor_msgs/PointCloud2)
- RGB image topic (sensor_msgs/CompressedImage)
- Camera info topic (sensor_msgs/CameraInfo) - stored once

Publishes:
- Elevation map via grid_map_msgs
"""

import sys
import os
from pathlib import Path

# Add sem_map package to Python path for imports
# When installed via ROS 2, scripts are in share/sem_map/scripts
script_dir = Path(__file__).parent.resolve()
if script_dir.name == 'sem_map':  # Running from lib/sem_map/
    # Add the scripts directory to path
    install_prefix = script_dir.parent.parent  # Go up to install/sem_map
    scripts_path = install_prefix / 'share' / 'sem_map' / 'scripts'
    if scripts_path.exists():
        sys.path.insert(0, str(scripts_path))

# import rclpy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
import numpy as np

# ROS message types
from sensor_msgs.msg import PointCloud2, Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Header

# ROS utilities
from sensor_msgs_py import point_cloud2
import tf2_ros
from tf2_ros import TransformException, TransformBroadcaster

# Semantic map components
from sem_map.parameter import Parameter

import yaml

# For pose interpolation
from scipy.spatial.transform import Rotation

# Semantic pipeline disabled for now
HAS_SEMANTIC_PIPELINE = False

# Try to import CuPy (falls back to NumPy if unavailable)
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    HAS_CUPY = False


class SemMapNode(Node):
    """ROS 2 node for unified semantic elevation mapping."""

    def __init__(self):
        super().__init__('sem_map_node')

        # Declare parameters from config
        self._declare_parameters()

        # Load parameters
        self.param = self._load_parameters()
        self.param.update()
        self.semantic_bev_layer_name = self.param.get_value('semantic_bev_layer_name', 'trail')

        # Initialize components
        self.xp = cp if HAS_CUPY else np

        # Initialize TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize TF2 broadcaster for map_center
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create callback groups for MultiThreadedExecutor
        from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
        # Reentrant for subscribers (lightweight buffer operations can run in parallel)
        self.subscriber_callback_group = ReentrantCallbackGroup()
        # Mutually exclusive for processing (heavy operations should not overlap)
        self.pointcloud_processing_callback_group = MutuallyExclusiveCallbackGroup()
        self.rgb_processing_callback_group = MutuallyExclusiveCallbackGroup()

        # Initialize unified semantic map (combines SemanticPipeline + ElevationMapping)
        self.sem_map = None
        try:
            self.get_logger().info('Initializing unified SemMap (semantic + elevation mapping)...')
            from sem_map.sem_map import SemMap
            self.sem_map = SemMap(self.param)
            self.get_logger().info('✅ SemMap initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize SemMap: {e}')
            self.sem_map = None

        # State tracking
        self.last_published_pointcloud_stamp_ns = None  # Track last published pointcloud timestamp in nanoseconds

        # Camera info storage (stored once)
        self.camera_info = None
        self.camera_info_received = False

        # Map center storage (updated with robot position)
        self.map_center = None  # Will store the current map center position

        # Message buffers (FIFO with max size)
        from collections import deque
        pose_buffer_size = self.get_parameter('pose_buffer_size').value
        pointcloud_buffer_size = self.get_parameter('pointcloud_buffer_size').value
        rgb_buffer_size = self.get_parameter('rgb_buffer_size').value

        # Log buffer sizes
        self.get_logger().info(f"Buffer sizes → pose: {pose_buffer_size}, pointcloud: {pointcloud_buffer_size}, rgb: {rgb_buffer_size}")

        self.pose_buffer = deque(maxlen=pose_buffer_size)
        self.pointcloud_buffer = deque(maxlen=pointcloud_buffer_size)
        self.rgb_buffer = deque(maxlen=rgb_buffer_size)

        # Cached transforms for efficiency (static transforms)
        self.cached_pointcloud_to_base_transform = None
        self.cached_pointcloud_transform_frame_pair = None
        self.cached_pose_to_base_transform = None
        self.cached_pose_transform_frame_pair = None
        self.cached_base_to_camera_transform = None
        self.cached_camera_transform_frame_pair = None

        # Frame IDs
        self.world_frame = self.param.get_value('world_frame', 'map')
        self.base_frame = self.param.get_value('base_frame', 'base_link')
        self.pose_frame = self.param.get_value('pose_frame', 'zed_camera_center')

        # Setup QoS profiles
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Individual subscribers with buffer-only callbacks
        self.get_logger().info('Setting up individual subscribers with buffers...')
        self.get_logger().info('  Using BEST_EFFORT QoS for all topics (pose, pointcloud, image)')

        # Get topics
        pose_topic = self.param.get_value('pose_topic', '/pose')
        pc_topic = self.param.get_value('pointcloud_topic', '/pointcloud')
        image_topic = self.param.get_value('image_topic', '/zed/zed_node/left/image_rect_color/compressed')
        camera_info_topic = self.param.get_value('camera_info_topic', '/zed/zed_node/left/camera_info')

        # Debug: Log the actual topics being subscribed to
        self.get_logger().info(f'📡 Subscribing to topics:')
        self.get_logger().info(f'  Pose: {pose_topic}')
        self.get_logger().info(f'  Pointcloud: {pc_topic}')
        self.get_logger().info(f'  Image: {image_topic}')
        self.get_logger().info(f'  Camera Info: {camera_info_topic}')

        # Create individual subscribers with buffer-only callbacks
        self.pose_sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.pose_callback,
            sensor_qos,
            callback_group=self.subscriber_callback_group
        )
        self.pc_sub = self.create_subscription(
            PointCloud2,
            pc_topic,
            self.pointcloud_callback,
            sensor_qos,
            callback_group=self.subscriber_callback_group
        )
        self.image_sub = self.create_subscription(
            CompressedImage,
            image_topic,
            self.rgb_callback,
            sensor_qos,
            callback_group=self.subscriber_callback_group
        )

        # Camera info subscriber (one-time storage)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            sensor_qos
        )

        # Publishers
        self.get_logger().info('Setting up publishers...')
        self.get_logger().info('  Using BEST_EFFORT QoS for all publishers (sensor data)')

        # Publisher for elevation map (if available via sem_map)
        if self.sem_map is not None and self.sem_map.elevation_map is not None:
            self.grid_map_pub = self.create_publisher(
                GridMap,
                '/sem_map/grid_map',
                sensor_qos
            )

        # Publisher for transformed pointcloud (for debugging/visualization)
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/sem_map/pointcloud',
            sensor_qos
        )

        # Publisher for map center pose (for debugging/visualization)
        self.map_center_pub = self.create_publisher(
            PoseStamped,
            '/sem_map/map_center',
            sensor_qos
        )

        # Publisher for traversability image
        self.traversability_pub = self.create_publisher(
            Image,
            '/sem_map/traversability',
            sensor_qos
        )

        # Publisher for semantic layer from elevation map
        self.semantic_layer_map_pub = self.create_publisher(
            Image,
            '/sem_map/trav_map',
            sensor_qos
        )

        # Processing timers to check buffers and process data
        pointcloud_processing_rate = self.get_parameter('pointcloud_processing_rate_hz').value
        rgb_processing_rate = self.get_parameter('rgb_processing_rate_hz').value

        self.get_logger().info(f'Pointcloud processing rate: {pointcloud_processing_rate}Hz')
        self.get_logger().info(f'RGB processing rate: {rgb_processing_rate}Hz')

        # Timer for pointcloud processing
        self.pointcloud_timer = self.create_timer(
            1.0 / pointcloud_processing_rate,
            self.pointcloud_processing_timer_callback,
            callback_group=self.pointcloud_processing_callback_group
        )

        # Timer for RGB processing
        self.rgb_timer = self.create_timer(
            1.0 / rgb_processing_rate,
            self.rgb_processing_timer_callback,
            callback_group=self.rgb_processing_callback_group
        )

        self.get_logger().info('Semantic pipeline node initialized!')
        self.get_logger().info(f'  World frame: {self.world_frame}')
        self.get_logger().info(f'  Base frame: {self.base_frame}')
        self.get_logger().info(f'  Pose topic: {pose_topic}')
        self.get_logger().info(f'  Pointcloud topic: {pc_topic}')
        self.get_logger().info(f'  Image topic: {image_topic}')
        self.get_logger().info(f'  Camera info topic: {camera_info_topic}')
        self.get_logger().info(f'  Transformed pointcloud topic: /sem_map/pointcloud')
        self.get_logger().info(f'  Map center pose topic: /sem_map/map_center')
        self.get_logger().info(f'  Traversability image topic: /sem_map/traversability')
        self.get_logger().info(f'  Map center TF: {self.world_frame} -> map_center')
        self.get_logger().info(f'  Grid map topic: /sem_map/grid_map')
        self.get_logger().info(f'  Pointcloud processing rate: {pointcloud_processing_rate}Hz')
        self.get_logger().info(f'  RGB processing rate: {rgb_processing_rate}Hz')
        self.get_logger().info(f'  Using backend: {"CuPy (GPU)" if HAS_CUPY else "NumPy (CPU)"}')
        self.get_logger().info(f'  Semantic features: {"Enabled" if self.sem_map is not None else "Disabled"}')

    def _declare_parameters(self):
        """Declare all ROS parameters."""
        # Topics
        self.declare_parameter('pose_topic', '/zed/zed_node/pose')
        self.declare_parameter('pointcloud_topic', '/rslidar_points')
        self.declare_parameter('image_topic', '/zed/zed_node/left/image_rect_color/compressed')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/left/camera_info')

        # Frames
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('base_frame', 'zed_camera_link')
        self.declare_parameter('pose_frame', 'zed_camera_center')

        # Semantic pipeline
        self.declare_parameter('backbone_name', 'dinov3s')
        self.declare_parameter('upsampler_type', 'jafar_partial')
        self.declare_parameter('upsample_partial_factor', 4)
        self.declare_parameter('seg_layer', 'dinov3s_sem_seg_trail_v1')
        self.declare_parameter('enable_pca', False)
        self.declare_parameter('enable_semantic_pipeline', True)
        self.declare_parameter('traversability_temperature', 1.0)
        self.declare_parameter('semantic_bev_layer_name', 'trail')

        # Elevation mapping
        self.declare_parameter('enable_height_map', True)
        self.declare_parameter('enable_bev_fusion', True)

        # Map parameters
        self.declare_parameter('resolution', 0.10)
        self.declare_parameter('map_size', 20.0)
        self.declare_parameter('backend', 'cupy')
        self.declare_parameter('precision', 'float16')
        self.declare_parameter('map_move_threshold_m', 1.0)

        # TF parameters
        self.declare_parameter('tf_lookup_timeout_sec', 0.25)

        # Buffer parameters
        self.declare_parameter('pose_buffer_size', 1000)
        self.declare_parameter('pointcloud_buffer_size', 100)
        self.declare_parameter('rgb_buffer_size', 100)
        self.declare_parameter('pointcloud_processing_rate_hz', 10.0)
        self.declare_parameter('rgb_processing_rate_hz', 10.0)

    def _load_parameters(self):
        """Load parameters from ROS and populate Parameter object."""
        param = Parameter()

        # Get all parameter names and update from ROS
        for name in param.get_names():
            try:
                ros_value = self.get_parameter(name).value
                param.set_value(name, ros_value)
            except rclpy.exceptions.ParameterNotDeclaredException:
                # Parameter not declared in ROS, use default
                pass

        # Additional parameters not in Parameter class
        param.set_value('pose_topic', self.get_parameter('pose_topic').value)
        param.set_value('pointcloud_topic', self.get_parameter('pointcloud_topic').value)
        param.set_value('image_topic', self.get_parameter('image_topic').value)
        param.set_value('camera_info_topic', self.get_parameter('camera_info_topic').value)
        param.set_value('tf_lookup_timeout_sec', self.get_parameter('tf_lookup_timeout_sec').value)
        param.set_value('world_frame', self.get_parameter('world_frame').value)
        param.set_value('base_frame', self.get_parameter('base_frame').value)
        param.set_value('pose_frame', self.get_parameter('pose_frame').value)

        # Ordered startup summary
        self.get_logger().info('───── Configuration Summary ─────')
        # Topics
        self.get_logger().info('Topics:')
        self.get_logger().info(f'  pose_topic: {param.get_value("pose_topic")}')
        self.get_logger().info(f'  pointcloud_topic: {param.get_value("pointcloud_topic")}')
        self.get_logger().info(f'  image_topic: {param.get_value("image_topic")}')
        self.get_logger().info(f'  camera_info_topic: {param.get_value("camera_info_topic")}')
        # Frames
        self.get_logger().info('Frames:')
        self.get_logger().info(f'  world_frame: {param.get_value("world_frame")}')
        self.get_logger().info(f'  base_frame: {param.get_value("base_frame")}')
        self.get_logger().info(f'  pose_frame: {param.get_value("pose_frame")}')
        # Buffers & Rates
        pose_bs = self.get_parameter('pose_buffer_size').value
        pc_bs = self.get_parameter('pointcloud_buffer_size').value
        rgb_bs = self.get_parameter('rgb_buffer_size').value
        pc_rate = self.get_parameter('pointcloud_processing_rate_hz').value
        rgb_rate = self.get_parameter('rgb_processing_rate_hz').value
        self.get_logger().info('Buffers & Rates:')
        self.get_logger().info(f'  pose_buffer_size: {pose_bs}  pointcloud_buffer_size: {pc_bs}  rgb_buffer_size: {rgb_bs}')
        self.get_logger().info(f'  pointcloud_processing_rate_hz: {pc_rate}  rgb_processing_rate_hz: {rgb_rate}')
        # Map
        self.get_logger().info('Map:')
        self.get_logger().info(f'  map_size: {param.get_value("map_size")}  resolution: {param.get_value("resolution")}')
        self.get_logger().info(f'  backend: {param.get_value("backend")}  precision: {param.get_value("precision")}')
        # Semantic pipeline
        self.get_logger().info('Semantic:')
        self.get_logger().info(f'  backbone: {param.get_value("backbone_name")}  upsampler_type: {param.get_value("upsampler_type")}  seg_layer: {param.get_value("seg_layer")}')
        self.get_logger().info(f'  enable_pca: {param.get_value("enable_pca")}  enable_semantic_pipeline: {param.get_value("enable_semantic_pipeline")}')
        self.get_logger().info(f'  traversability_temperature: {param.get_value("traversability_temperature")}')
        self.get_logger().info(f'  semantic_bev_layer_name: {param.get_value("semantic_bev_layer_name")}')
        # Elevation mapping
        self.get_logger().info('Elevation:')
        self.get_logger().info(f'  enable_height_map: {param.get_value("enable_height_map")}  enable_bev_fusion: {param.get_value("enable_bev_fusion")}')
        # TF
        self.get_logger().info('TF:')
        self.get_logger().info(f'  tf_lookup_timeout_sec: {self.get_parameter("tf_lookup_timeout_sec").value}')
        self.get_logger().info('─────────────────────────────────')

        return param

    def camera_info_callback(self, camera_info_msg: CameraInfo):
        """Store camera info once when first received."""
        if not self.camera_info_received:
            self.camera_info = camera_info_msg
            self.camera_info_received = True
            self.get_logger().info(f'✅ Camera info received and stored from {camera_info_msg.header.frame_id}')
            self.get_logger().info(f'   Image size: {camera_info_msg.width}x{camera_info_msg.height}')
            self.get_logger().info(f'   Camera matrix: {camera_info_msg.k}')

            # Pass camera info to unified SemMap
            if self.sem_map is not None:
                # Convert K array to numpy 3x3 matrix
                K = np.array(camera_info_msg.k).reshape(3, 3)
                D = np.array(camera_info_msg.d)

                self.sem_map.set_camera_info(
                    K,
                    D,
                    camera_info_msg.distortion_model,
                    camera_info_msg.width,
                    camera_info_msg.height
                )
                self.get_logger().info('✅ Camera info passed to SemMap for traversability fusion')

    def pose_callback(self, pose_msg: PoseStamped):
        """Store pose in buffer (lightweight callback)."""
        # Always try to validate and cache transform, then store pose
        if self._validate_and_cache_pose_transform(pose_msg):
            self.pose_buffer.append(pose_msg)
            self.get_logger().debug(f'Stored pose in buffer: {len(self.pose_buffer)} poses')
        else:
            self.get_logger().warn('Failed to validate pose transform, skipping pose')

    def pointcloud_callback(self, pointcloud_msg: PointCloud2):
        """Store pointcloud in buffer (lightweight callback)."""
        # Always try to validate and cache transform, then store pointcloud
        if self._validate_and_cache_pointcloud_transform(pointcloud_msg):
            self.pointcloud_buffer.append(pointcloud_msg)
            self.get_logger().debug(f'Stored pointcloud in buffer: {len(self.pointcloud_buffer)} pointclouds')
        else:
            self.get_logger().warn('Failed to validate pointcloud transform, skipping pointcloud')

    def rgb_callback(self, image_msg: CompressedImage):
        """Store RGB image in buffer (lightweight callback)."""
        # Always try to validate and cache transform, then store RGB image
        if self._validate_and_cache_camera_transform(image_msg):
            self.rgb_buffer.append(image_msg)
            self.get_logger().debug(f'Stored RGB image in buffer: {len(self.rgb_buffer)} images')
        else:
            self.get_logger().warn('Failed to validate camera transform, skipping RGB image')

    def get_pose_with_timestamp(self, target_timestamp):
        """Find pose with closest timestamp to target, with interpolation if needed."""
        if not self.pose_buffer:
            return None

        target_ns = target_timestamp.sec * 1_000_000_000 + target_timestamp.nanosec

        # If only one pose, return it
        if len(self.pose_buffer) == 1:
            pose = self.pose_buffer[0]
            pose_ns = pose.header.stamp.sec * 1_000_000_000 + pose.header.stamp.nanosec
            delta_sec = abs(pose_ns - target_ns) / 1e9
            if delta_sec > 0.1:
                self.get_logger().warn(f'Large timestamp delta: {delta_sec*1000:.1f}ms (single pose)')
            return pose

        # Find two poses to interpolate between
        pose_before = None
        pose_after = None
        time_before = None
        time_after = None

        # Search for poses before and after target timestamp
        for pose in self.pose_buffer:
            pose_ns = pose.header.stamp.sec * 1_000_000_000 + pose.header.stamp.nanosec

            if pose_ns <= target_ns:
                if pose_before is None or pose_ns > time_before:
                    pose_before = pose
                    time_before = pose_ns
            else:  # pose_ns > target_ns
                if pose_after is None or pose_ns < time_after:
                    pose_after = pose
                    time_after = pose_ns

        # Determine best approach
        if pose_before is None and pose_after is None:
            return None
        elif pose_before is None:
            # Only pose after target - use it
            delta_sec = (time_after - target_ns) / 1e9
            if delta_sec > 0.1:
                self.get_logger().warn(f'Large timestamp delta: {delta_sec*1000:.1f}ms (using future pose)')
            return pose_after
        elif pose_after is None:
            # Only pose before target - use it
            delta_sec = (target_ns - time_before) / 1e9
            if delta_sec > 0.1:
                self.get_logger().warn(f'Large timestamp delta: {delta_sec*1000:.1f}ms (using past pose)')
            return pose_before
        else:
            # Both poses available - interpolate
            return self._interpolate_poses(pose_before, pose_after, target_ns, time_before, time_after)

    def _interpolate_poses(self, pose_before, pose_after, target_ns, time_before, time_after):
        """Interpolate between two poses at target timestamp."""
        # Calculate interpolation factor (0 = pose_before, 1 = pose_after)
        if time_after == time_before:
            # Same timestamp, return either pose
            return pose_before

        alpha = (target_ns - time_before) / (time_after - time_before)
        alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]

        # Extract positions
        pos_before = np.array([
            pose_before.pose.position.x,
            pose_before.pose.position.y,
            pose_before.pose.position.z
        ])
        pos_after = np.array([
            pose_after.pose.position.x,
            pose_after.pose.position.y,
            pose_after.pose.position.z
        ])

        # Linear interpolation for translation
        pos_interp = pos_before + alpha * (pos_after - pos_before)

        # Extract quaternions
        q_before = np.array([
            pose_before.pose.orientation.w,
            pose_before.pose.orientation.x,
            pose_before.pose.orientation.y,
            pose_before.pose.orientation.z
        ])
        q_after = np.array([
            pose_after.pose.orientation.w,
            pose_after.pose.orientation.x,
            pose_after.pose.orientation.y,
            pose_after.pose.orientation.z
        ])

        # Validate quaternions before interpolation
        norm_before = np.linalg.norm(q_before)
        norm_after = np.linalg.norm(q_after)

        if norm_before < 1e-6 or norm_after < 1e-6:
            self.get_logger().warn(f'Invalid quaternion detected: norm_before={norm_before:.6f}, norm_after={norm_after:.6f}')
            # Use the pose with valid quaternion, or default to identity
            if norm_before > 1e-6:
                q_interp = q_before
            elif norm_after > 1e-6:
                q_interp = q_after
            else:
                # Both invalid, use identity quaternion
                q_interp = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Normalize quaternions
            q_before = q_before / norm_before
            q_after = q_after / norm_after

            try:
                # Spherical linear interpolation (SLERP) for rotation
                rot_before = Rotation.from_quat(q_before[[1, 2, 3, 0]])  # scipy expects [x,y,z,w]
                rot_after = Rotation.from_quat(q_after[[1, 2, 3, 0]])

                # Interpolate rotation using SLERP
                rot_interp = rot_before * (rot_before.inv() * rot_after) ** alpha
                q_interp_scipy = rot_interp.as_quat()  # Returns [x,y,z,w]
                q_interp = np.array([q_interp_scipy[3], q_interp_scipy[0], q_interp_scipy[1], q_interp_scipy[2]])  # Convert back to [w,x,y,z]

            except Exception as e:
                self.get_logger().warn(f'Quaternion interpolation failed: {e}, using linear interpolation')
                # Fallback to linear interpolation
                q_interp = q_before + alpha * (q_after - q_before)
                # Normalize result
                q_interp = q_interp / np.linalg.norm(q_interp)

        # Create interpolated pose message
        from geometry_msgs.msg import PoseStamped
        from builtin_interfaces.msg import Time
        interp_pose = PoseStamped()
        interp_pose.header = pose_before.header  # Use header from pose_before

        # Convert target_ns back to Time message
        target_sec = target_ns // 1_000_000_000
        target_nanosec = target_ns % 1_000_000_000
        interp_pose.header.stamp = Time(sec=int(target_sec), nanosec=int(target_nanosec))

        # Set interpolated position
        interp_pose.pose.position.x = float(pos_interp[0])
        interp_pose.pose.position.y = float(pos_interp[1])
        interp_pose.pose.position.z = float(pos_interp[2])

        # Set interpolated orientation (q_interp is already in [w,x,y,z] format)
        interp_pose.pose.orientation.w = float(q_interp[0])
        interp_pose.pose.orientation.x = float(q_interp[1])
        interp_pose.pose.orientation.y = float(q_interp[2])
        interp_pose.pose.orientation.z = float(q_interp[3])

        # Log interpolation info
        delta_before = (target_ns - time_before) / 1e9
        delta_after = (time_after - target_ns) / 1e9
        self.get_logger().debug(
            f'Interpolated pose: alpha={alpha:.3f}, '
            f'delta_before={delta_before*1000:.1f}ms, delta_after={delta_after*1000:.1f}ms'
        )

        return interp_pose

    def process_pointcloud_from_buffer(self):
        """Process pointcloud from buffer with closest timestamp pose."""

        if not self.pointcloud_buffer:
            return  # No pointcloud to process


        # Get latest pointcloud and find closest pose by timestamp
        pointcloud_msg = self.pointcloud_buffer.popleft()  # Remove from buffer
        pose_msg = self.get_pose_with_timestamp(pointcloud_msg.header.stamp)

        if pose_msg is None:
            self.get_logger().warn('No pose available for pointcloud processing')
            return

        # Calculate and log timestamp delta
        pc_ns = pointcloud_msg.header.stamp.sec * 1_000_000_000 + pointcloud_msg.header.stamp.nanosec
        pose_ns = pose_msg.header.stamp.sec * 1_000_000_000 + pose_msg.header.stamp.nanosec
        delta_ms = abs(pc_ns - pose_ns) / 1e6

        self.get_logger().debug(
            f'Processing pointcloud with closest pose: pc={pointcloud_msg.header.stamp.sec}.{pointcloud_msg.header.stamp.nanosec}, '
            f'pose={pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec}, delta={delta_ms:.1f}ms'
        )

        # Use existing pointcloud processing logic
        self._process_pointcloud(pose_msg, pointcloud_msg)

    def process_rgb_from_buffer(self):
        """Process RGB image from buffer with closest timestamp pose."""
        if not self.rgb_buffer:
            return  # No RGB image to process

        # Get latest RGB image and find closest pose by timestamp
        image_msg = self.rgb_buffer.popleft()  # Remove from buffer
        pose_msg = self.get_pose_with_timestamp(image_msg.header.stamp)

        if pose_msg is None:
            self.get_logger().warn('No pose available for RGB processing')
            return

        # Calculate and log timestamp delta
        rgb_ns = image_msg.header.stamp.sec * 1_000_000_000 + image_msg.header.stamp.nanosec
        pose_ns = pose_msg.header.stamp.sec * 1_000_000_000 + pose_msg.header.stamp.nanosec
        delta_ms = abs(rgb_ns - pose_ns) / 1e6

        self.get_logger().debug(
            f'Processing RGB with closest pose: rgb={image_msg.header.stamp.sec}.{image_msg.header.stamp.nanosec}, '
            f'pose={pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec}, delta={delta_ms:.1f}ms'
        )

        # Use existing RGB processing logic
        self._process_image(pose_msg, image_msg)

    def pointcloud_processing_timer_callback(self):
        """Timer callback to process pointclouds from buffer."""
        if self.pointcloud_buffer:
            self.process_pointcloud_from_buffer()

    def rgb_processing_timer_callback(self):
        """Timer callback to process RGB images from buffer."""
        if self.rgb_buffer:
            self.process_rgb_from_buffer()

    def _validate_and_cache_pose_transform(self, pose_msg: PoseStamped):
        """Validate pose frame and cache pose_T_base transform if needed."""
        # Get pose frame from message header
        pose_header_frame = pose_msg.header.frame_id

        # Validate: pose header frame should match world frame
        if pose_header_frame != self.world_frame:
            self.get_logger().error(
                f'❌ Pose header frame mismatch! Expected {self.world_frame}, got {pose_header_frame}. '
                f'Pose topic header frame should match world_frame.'
            )
            return False

        # If pose_frame != base_frame, cache pose_T_base static transform
        if self.pose_frame != self.base_frame:
            frame_pair = f"{self.pose_frame}->{self.base_frame}"

            # Check if we need to update the cached transform
            if (self.cached_pose_to_base_transform is None or
                self.cached_pose_transform_frame_pair != frame_pair):
                try:
                    self.get_logger().info(f'Caching pose_T_base transform: {frame_pair}')
                    transform_pose_to_base = self.tf_buffer.lookup_transform(
                        self.pose_frame,
                        self.base_frame,
                        rclpy.time.Time(),  # Latest available
                        timeout=Duration(seconds=self.get_parameter('tf_lookup_timeout_sec').value)
                    )

                    # Extract and cache transform
                    t_p2b = transform_pose_to_base.transform.translation
                    q_p2b = transform_pose_to_base.transform.rotation
                    R_pose_to_base = self._quaternion_to_rotation_matrix(q_p2b.w, q_p2b.x, q_p2b.y, q_p2b.z)
                    t_pose_to_base = np.array([t_p2b.x, t_p2b.y, t_p2b.z])

                    self.cached_pose_to_base_transform = (R_pose_to_base, t_pose_to_base)
                    self.cached_pose_transform_frame_pair = frame_pair
                    self.get_logger().info(f'✅ Cached pose_T_base transform for {frame_pair}')
                    self.get_logger().info(f'   R_pose_to_base: {R_pose_to_base}')
                    self.get_logger().info(f'   t_pose_to_base: {t_pose_to_base}')

                except TransformException as ex:
                    self.get_logger().warn(f'⚠️  Could not cache pose_T_base transform {frame_pair}: {ex}')
                    return False
            else:
                # Transform already cached, return early
                return True
        else:
            # pose_frame == base_frame, no transform needed
            self.cached_pose_to_base_transform = None
            self.cached_pose_transform_frame_pair = None

        return True

    @staticmethod
    def _np_dtype_for_pointfield(datatype: int, is_bigendian: bool) -> str:
        """Map sensor_msgs/PointField datatype to numpy dtype string with endianness."""
        endian = ">" if is_bigendian else "<"
        # Datatype IDs follow sensor_msgs.msg.PointField constants
        if datatype == 1:   # INT8
            return endian + "i1"
        if datatype == 2:   # UINT8
            return endian + "u1"
        if datatype == 3:   # INT16
            return endian + "i2"
        if datatype == 4:   # UINT16
            return endian + "u2"
        if datatype == 5:   # INT32
            return endian + "i4"
        if datatype == 6:   # UINT32
            return endian + "u4"
        if datatype == 7:   # FLOAT32
            return endian + "f4"
        if datatype == 8:   # FLOAT64
            return endian + "f8"
        raise ValueError(f"Unsupported PointField datatype: {datatype}")

    def _parse_pointcloud_xyz(self, msg: PointCloud2) -> np.ndarray:
        """Efficiently parse PointCloud2 to (N,3) float32 using a structured dtype view.

        Works with mixed-field clouds by creating a dtype with explicit offsets for x,y,z.
        """
        if msg.height == 0 or msg.width == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Build mapping from name to field for quick lookup
        name_to_field = {f.name: f for f in msg.fields}
        for key in ("x", "y", "z"):
            if key not in name_to_field:
                raise KeyError(f"PointCloud2 missing field '{key}'")

        fmt_x = self._np_dtype_for_pointfield(name_to_field["x"].datatype, msg.is_bigendian)
        fmt_y = self._np_dtype_for_pointfield(name_to_field["y"].datatype, msg.is_bigendian)
        fmt_z = self._np_dtype_for_pointfield(name_to_field["z"].datatype, msg.is_bigendian)

        sdt = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': [fmt_x, fmt_y, fmt_z],
            'offsets': [int(name_to_field['x'].offset), int(name_to_field['y'].offset), int(name_to_field['z'].offset)],
            'itemsize': int(msg.point_step),
        })

        count = int(msg.width) * int(msg.height)
        recs = np.frombuffer(msg.data, dtype=sdt, count=count)

        # Extract as float32 without copies when possible
        x = np.asarray(recs['x']).astype(np.float32, copy=False)
        y = np.asarray(recs['y']).astype(np.float32, copy=False)
        z = np.asarray(recs['z']).astype(np.float32, copy=False)
        pts = np.stack((x, y, z), axis=1)

        # Filter invalid values
        mask = np.isfinite(pts).all(axis=1)
        if not mask.all():
            pts = pts[mask]
        return pts

    def _validate_and_cache_pointcloud_transform(self, pointcloud_msg: PointCloud2):
        """Validate pointcloud frame and cache pointcloud_T_base transform if needed."""
        # Get pointcloud frame from message header
        pc_frame = pointcloud_msg.header.frame_id

        # If pointcloud_frame != base_frame, cache pointcloud_T_base static transform
        if pc_frame != self.base_frame:
            frame_pair = f"{pc_frame}->{self.base_frame}"

            # Check if we need to update the cached transform
            if (self.cached_pointcloud_to_base_transform is None or
                self.cached_pointcloud_transform_frame_pair != frame_pair):
                try:
                    self.get_logger().info(f'Caching pointcloud_T_base transform: {frame_pair}')
                    transform_sensor_to_base = self.tf_buffer.lookup_transform(
                        pc_frame,
                        self.base_frame,
                        rclpy.time.Time(),  # Latest available
                        timeout=Duration(seconds=self.get_parameter('tf_lookup_timeout_sec').value)
                    )

                    # Extract and cache transform
                    t_s2b = transform_sensor_to_base.transform.translation
                    q_s2b = transform_sensor_to_base.transform.rotation
                    R_sensor_to_base = self._quaternion_to_rotation_matrix(q_s2b.w, q_s2b.x, q_s2b.y, q_s2b.z)
                    t_sensor_to_base = np.array([t_s2b.x, t_s2b.y, t_s2b.z])

                    self.cached_pointcloud_to_base_transform = (R_sensor_to_base, t_sensor_to_base)
                    self.cached_pointcloud_transform_frame_pair = frame_pair
                    self.get_logger().info(f'✅ Cached pointcloud_T_base transform for {frame_pair}')
                    self.get_logger().info(f'   R_sensor_to_base: {R_sensor_to_base}')
                    self.get_logger().info(f'   t_sensor_to_base: {t_sensor_to_base}')

                except TransformException as ex:
                    self.get_logger().warn(f'⚠️  Could not cache pointcloud_T_base transform {frame_pair}: {ex}')
                    return False
            else:
                # Transform already cached, return early
                return True
        else:
            # pointcloud_frame == base_frame, no transform needed
            self.cached_pointcloud_to_base_transform = None
            self.cached_pointcloud_transform_frame_pair = None

        return True

    def _validate_and_cache_camera_transform(self, image_msg: CompressedImage):
        """Validate camera frame and cache base_T_camera transform if needed."""
        # Get camera frame from message header
        camera_frame = image_msg.header.frame_id

        # If camera_frame != base_frame, cache base_T_camera static transform
        if camera_frame != self.base_frame:
            frame_pair = f"{self.base_frame}->{camera_frame}"

            # Check if we need to update the cached transform
            if (self.cached_base_to_camera_transform is None or
                self.cached_camera_transform_frame_pair != frame_pair):
                try:
                    self.get_logger().info(f'Caching base_T_camera transform: {frame_pair}')
                    transform_base_to_camera = self.tf_buffer.lookup_transform(
                        self.base_frame,
                        camera_frame,
                        rclpy.time.Time(),  # Latest available
                        timeout=Duration(seconds=self.get_parameter('tf_lookup_timeout_sec').value)
                    )

                    # Extract and cache transform
                    t_b2c = transform_base_to_camera.transform.translation
                    q_b2c = transform_base_to_camera.transform.rotation
                    R_base_to_camera = self._quaternion_to_rotation_matrix(q_b2c.w, q_b2c.x, q_b2c.y, q_b2c.z)
                    t_base_to_camera = np.array([t_b2c.x, t_b2c.y, t_b2c.z])

                    self.cached_base_to_camera_transform = (R_base_to_camera, t_base_to_camera)
                    self.cached_camera_transform_frame_pair = frame_pair
                    self.get_logger().info(f'✅ Cached base_T_camera transform for {frame_pair}')
                    self.get_logger().info(f'   R_base_to_camera: {R_base_to_camera}')
                    self.get_logger().info(f'   t_base_to_camera: {t_base_to_camera}')

                except TransformException as ex:
                    self.get_logger().warn(f'⚠️  Could not cache base_T_camera transform {frame_pair}: {ex}')
                    return False
            else:
                # Transform already cached, return early
                return True
        else:
            # camera_frame == base_frame, no transform needed
            self.cached_base_to_camera_transform = None
            self.cached_camera_transform_frame_pair = None

        return True

    def _process_pointcloud(self, pose_msg: PoseStamped, pointcloud_msg: PointCloud2):
        """Process pointcloud and update elevation map."""
        # Get pointcloud frame from message header
        pc_frame = pointcloud_msg.header.frame_id

        # Convert pointcloud to numpy array (in pointcloud's own frame)
        # Robust mixed-field parser using structured dtype view
        try:
            pc_data = self._parse_pointcloud_xyz(pointcloud_msg)
            if pc_data.size == 0:
                return
        except Exception as e:
            self.get_logger().warn(f'Failed to parse PointCloud2 xyz: {e}')
            return

        # Transform pointcloud to base_frame if needed
        if pc_frame != self.base_frame:
            # Use cached pointcloud_T_base transform
            if self.cached_pointcloud_to_base_transform is None:
                self.get_logger().error(
                    f'❌ No cached pointcloud_T_base transform available for {pc_frame}->{self.base_frame}. '
                    f'This should have been cached in pointcloud_callback. Please check TF tree.'
                )
                return  # Skip processing this pointcloud

            # Use cached transform
            R_sensor_to_base, t_sensor_to_base = self.cached_pointcloud_to_base_transform
            self.get_logger().debug(f'Using cached pointcloud_T_base transform for {pc_frame}->{self.base_frame}')

            # Transform points: p_base = R * p_sensor + t
            # Note: pc_data is (N,3), R is (3,3), so we need R.T for correct matrix multiplication
            pc_data_in_base = (pc_data @ R_sensor_to_base) + t_sensor_to_base
        else:
            # pointcloud_frame == base_frame, no transform needed
            pc_data_in_base = pc_data

        # The pose topic gives the pose of pose_frame in world_frame
        # We need to transform this to base_frame to get the map center
        pose_data = pose_msg.pose
        position_pose_in_world = np.array([
            pose_data.position.x,
            pose_data.position.y,
            pose_data.position.z
        ])

        # Extract rotation (quaternion to rotation matrix) - pose_frame to world
        q = pose_data.orientation
        rotation_pose_to_world = self._quaternion_to_rotation_matrix(q.w, q.x, q.y, q.z)

        # Transform pose from pose_frame to base_frame if needed
        if self.pose_frame != self.base_frame:
            # Use cached pose_T_base transform
            if self.cached_pose_to_base_transform is None:
                self.get_logger().error(
                    f'❌ No cached pose_T_base transform available. '
                    f'This should have been cached in pose_callback. Please check TF tree.'
                )
                return  # Skip processing this pointcloud

            # Use cached pose to base transform
            R_pose_to_base, t_pose_to_base = self.cached_pose_to_base_transform
            self.get_logger().debug(f'Using cached pose_T_base transform for {self.pose_frame}->{self.base_frame}')

            # Transform pose from pose_frame to base_frame in world frame
            # T_world_base = T_world_pose * T_pose_base
            # For position: p_base_world = p_pose_world + R_pose_world * t_pose_base
            # For rotation: R_base_world = R_pose_world * R_pose_base
            position_base_in_world = position_pose_in_world + (rotation_pose_to_world @ t_pose_to_base)
            rotation_base_to_world = rotation_pose_to_world @ R_pose_to_base
        else:
            # pose_frame == base_frame, no transform needed
            position_base_in_world = position_pose_in_world
            rotation_base_to_world = rotation_pose_to_world

        # Convert to appropriate backend (CuPy or NumPy)
        if HAS_CUPY:
            points = cp.asarray(pc_data_in_base, dtype=cp.float32)
            rotation = cp.asarray(rotation_base_to_world, dtype=cp.float32)
            translation = cp.asarray(position_base_in_world, dtype=cp.float32)
        else:
            points = pc_data_in_base.astype(np.float32)
            rotation = rotation_base_to_world.astype(np.float32)
            translation = position_base_in_world.astype(np.float32)

        # Update map center to robot position in map frame (needs x, y, z)
        if HAS_CUPY:
            robot_position_fp16 = cp.asarray(translation, dtype=cp.float16)  # Full 3D position
        else:
            robot_position_fp16 = translation.astype(np.float16)  # Full 3D position

        # self.semantic_map.update_center(robot_position_fp16)

        # Save map center in node for consistent use
        self.map_center = translation.copy()  # Store as numpy array
        self.get_logger().debug(f'Updated map center: [{self.map_center[0]:.2f}, {self.map_center[1]:.2f}, {self.map_center[2]:.2f}]')

        # Update semantic map with pointcloud
        # Points are in base_link frame, R and t describe base_link to map transform
        # Kernel will transform: p_map = R @ p_base + t
        self.get_logger().debug(
            f'Processing {len(points)} points at pose [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}], '
            f'map_center=[{self.map_center[0]:.2f}, {self.map_center[1]:.2f}, {self.map_center[2]:.2f}]',
            throttle_duration_sec=2.0
        )

        # Removed debug-only GPU→CPU conversion for point range logging

        # Flatten rotation matrix for kernel (3x3 -> 9 elements)
        # R_flat = rotation.flatten()

        # self.semantic_map.update_with_pointcloud(points, R_flat, translation)

        # Publish transformed pointcloud for visualization (only if new data)
        current_stamp_ns = (pointcloud_msg.header.stamp.sec * 1_000_000_000 +
                           pointcloud_msg.header.stamp.nanosec)
        if self.last_published_pointcloud_stamp_ns != current_stamp_ns:
            self._publish_transformed_pointcloud(pc_data_in_base, pointcloud_msg.header)
            self.last_published_pointcloud_stamp_ns = current_stamp_ns
            self.get_logger().debug(
                f'Published {len(pc_data_in_base)} transformed points to /sem_map/pointcloud (stamp: {current_stamp_ns})',
                throttle_duration_sec=2.0
            )

        # Publish map center pose for visualization
        self._publish_map_center_pose()

        # Publish TF transform from world_frame to map_center
        self._publish_map_center_tf()

        # Update elevation map via unified sem_map (if available)
        if self.sem_map is not None and self.sem_map.elevation_map is not None:
            try:
                # Convert points to the format expected by elevation map
                points_np = points.get() if hasattr(points, 'get') else points
                channels = ["x", "y", "z"]

                # Convert pose to float64
                R = rotation.astype(np.float64)
                t = translation.astype(np.float64)

                # Move elevation map to current robot position first
                robot_pos = translation.get() if hasattr(translation, 'get') else translation
                self.sem_map.move_to(
                    robot_pos.astype(np.float64),
                    np.eye(3, dtype=np.float64)
                )

                # Input pointcloud via unified sem_map
                self.sem_map.process_pointcloud(points_np, channels, R, t)

                self._publish_grid_map_cupy()
                self.get_logger().debug(f'Elevation map updated with {len(points)} points')

            except Exception as e:
                self.get_logger().warn(f'⚠️  elevation_map update failed: {e}')

    def _process_image(self, pose_msg: PoseStamped, image_msg: CompressedImage):
        """Process synchronized image data and generate traversability map."""
        if self.sem_map is None:
            # SemMap not initialized, skip processing
            self.get_logger().warn('SemMap not initialized, skipping image processing')
            return

        try:
            # Decompress image using cv_bridge
            import cv2
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                self.get_logger().warn('Failed to decode compressed image')
                return

            # Convert BGR to RGB (OpenCV loads as BGR)
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            height, width, channels = cv_image_rgb.shape

            # Get timestamp in nanoseconds
            stamp_ns = image_msg.header.stamp.sec * 1_000_000_000 + image_msg.header.stamp.nanosec

            # Get camera pose for traversability fusion
            pose_R = None
            pose_t = None
            if pose_msg is not None and self.cached_base_to_camera_transform is not None:
                # Extract base pose from pose_msg
                base_position = np.array([
                    pose_msg.pose.position.x,
                    pose_msg.pose.position.y,
                    pose_msg.pose.position.z
                ])
                base_quat = np.array([
                    pose_msg.pose.orientation.w,
                    pose_msg.pose.orientation.x,
                    pose_msg.pose.orientation.y,
                    pose_msg.pose.orientation.z
                ])
                from scipy.spatial.transform import Rotation
                base_R = Rotation.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()

                # Get cached base-to-camera transform (stored as tuple of (R, t) numpy arrays)
                camera_R_in_base, camera_t_in_base = self.cached_base_to_camera_transform

                # Compute camera pose in world frame
                # camera_position_world = base_position + base_R @ camera_t_in_base
                # camera_R_world = base_R @ camera_R_in_base
                pose_t = base_position + base_R @ camera_t_in_base
                pose_R = base_R @ camera_R_in_base

            # Process through unified semantic map
            # This will generate traversability internally and fuse to elevation map if pose is provided
            sensor_name = image_msg.header.frame_id
            self.sem_map.process_frame(
                cv_image_rgb,
                height,
                width,
                channels,
                sensor_name,
                stamp_ns,
                pose_R,
                pose_t
            )

            self.get_logger().debug(
                f'Processed image for traversability: {width}x{height}',
                throttle_duration_sec=2.0
            )

            # Attempt to publish traversability if available
            self._publish_traversability_if_available(image_msg)

            # Note: Traversability fusion is now handled inside SemMap.process_frame()
            # The redundant _add_traversability_to_elevation_map() call has been removed

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def _publish_traversability_if_available(self, image_msg: CompressedImage):
        """Extract and publish traversability from semantic map."""
        if self.sem_map is None:
            self.get_logger().warn('SemMap not initialized, skipping traversability publication')
            return

        try:
            # Access traversability stored in CPU memory via unified API
            trav_np = self.sem_map.get_last_traversability()
            if trav_np is None:
                return
            trav_map = trav_np
            if trav_map is None:
                return

            # Convert to numpy and scale to 0-65535 for MONO16 encoding
            # trav_map is already numpy array from get_last_traversability()
            trav_map = np.clip(trav_map, 0, 1).astype(np.float32)
            trav_uint16 = (trav_map * 65535.0).astype(np.uint16)

            # Create ROS Image message manually (avoiding cv_bridge)
            trav_msg = Image()
            trav_msg.header = image_msg.header
            trav_msg.height = trav_uint16.shape[0]
            trav_msg.width = trav_uint16.shape[1]
            trav_msg.encoding = "mono16"
            trav_msg.is_bigendian = False
            trav_msg.step = trav_uint16.shape[1] * 2  # width * bytes_per_pixel (2 bytes for uint16)
            trav_msg.data = trav_uint16.tobytes()

            # Publish
            self.traversability_pub.publish(trav_msg)

            # Publish TF transform from world to traversability frame for visualization
            # Note: TF publishing handled separately (camera pose computed in _process_image)

            self.get_logger().info(
                f'Published traversability image (MONO16): {trav_uint16.shape[1]}x{trav_uint16.shape[0]}',
                throttle_duration_sec=5.0
            )
        except Exception as e:
            self.get_logger().debug(f'Traversability not available: {e}')

    def _publish_world_to_trav_tf_with_pose(self, image_msg: CompressedImage, R: np.ndarray, t: np.ndarray, camera_frame: str):
        """Publish TF transform from world frame to traversability frame using calculated camera pose."""
        try:
            # Create transform from world to traversability frame
            transform_stamped = TransformStamped()
            transform_stamped.header.stamp = image_msg.header.stamp
            transform_stamped.header.frame_id = self.world_frame
            transform_stamped.child_frame_id = f"{camera_frame}_traversability"

            # Set position from calculated camera pose in world coordinates
            transform_stamped.transform.translation.x = float(t[0])
            transform_stamped.transform.translation.y = float(t[1])
            transform_stamped.transform.translation.z = float(t[2])

            # Convert rotation matrix to quaternion via SciPy
            rotation = Rotation.from_matrix(R)
            quat = rotation.as_quat()  # [x,y,z,w]

            # Set orientation from calculated camera rotation in world coordinates
            transform_stamped.transform.rotation.x = float(quat[0])
            transform_stamped.transform.rotation.y = float(quat[1])
            transform_stamped.transform.rotation.z = float(quat[2])
            transform_stamped.transform.rotation.w = float(quat[3])

            # Broadcast the transform
            self.tf_broadcaster.sendTransform(transform_stamped)

            self.get_logger().debug(
                f'Published world->trav TF: {self.world_frame}->{camera_frame}_traversability, '
                f'pose=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}], '
                f'quat=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]',
                throttle_duration_sec=5.0
            )

        except Exception as e:
            self.get_logger().debug(f'Failed to publish world->trav TF: {e}')

    def _publish_semantic_layer_map(self):
        """Extract and publish the semantic BEV layer from the elevation map as uint8 image."""
        if self.sem_map is None or self.sem_map.elevation_map is None:
            return

        try:
            # Get semantic layer from elevation map via unified sem_map
            semantic_layer = self.semantic_bev_layer_name
            layer_data = self.sem_map.get_elevation_map_layer(semantic_layer)
            if layer_data is None:
                return

            # Apply rotation and flip to match expected visualization orientation
            layer_data = np.flip(np.rot90(layer_data, k=1), axis=0)  # Rotate 90° CCW then flip vertically

            # Convert to uint16 for MONO16 encoding (scale from 0-1 to 0-65535)
            layer_uint16 = (layer_data * 65535).astype(np.uint16)

            # Create ROS Image message
            semantic_msg = Image()
            semantic_msg.header.stamp = self.get_clock().now().to_msg()
            semantic_msg.header.frame_id = self.world_frame
            semantic_msg.height = layer_uint16.shape[0]
            semantic_msg.width = layer_uint16.shape[1]
            semantic_msg.encoding = "mono16"
            semantic_msg.is_bigendian = False
            semantic_msg.step = layer_uint16.shape[1] * 2  # width * bytes_per_pixel (2 bytes for uint16)
            semantic_msg.data = layer_uint16.tobytes()

            # Publish
            self.semantic_layer_map_pub.publish(semantic_msg)

            self.get_logger().debug(
                f'Published {semantic_layer} map (MONO16): {layer_uint16.shape[1]}x{layer_uint16.shape[0]}, '
                f'range=[{layer_data.min():.3f}, {layer_data.max():.3f}]',
                throttle_duration_sec=5.0
            )

        except Exception as e:
            self.get_logger().debug(f'Failed to publish {self.semantic_bev_layer_name} map: {e}')

    # REMOVED: _add_traversability_to_elevation_map() - redundant method
    # Traversability fusion now handled automatically in SemMap.process_frame() when pose is provided

    def _publish_grid_map_cupy(self):
        """Publish elevation map as GridMap message."""
        if self.sem_map is None or self.sem_map.elevation_map is None:
            return

        try:
            # Get elevation data from unified sem_map
            semantic_layer = self.semantic_bev_layer_name
            elevation_data = self.sem_map.get_elevation_map_layer("elevation")
            variance_data = self.sem_map.get_elevation_map_layer("variance")
            semantic_layer_data = self.sem_map.get_elevation_map_layer(semantic_layer)

            if elevation_data is None or variance_data is None or semantic_layer_data is None:
                return

            # Apply rotation and flip to match expected visualization orientation
            # This matches the pattern from test_robot_movement_scenario.py
            elevation_data = np.flip(np.rot90(elevation_data, k=1), axis=0)  # Rotate 90° CCW then flip vertically
            variance_data = np.flip(np.rot90(variance_data, k=1), axis=0)  # Rotate 90° CCW then flip vertically
            semantic_layer_data = np.flip(np.rot90(semantic_layer_data, k=1), axis=0)  # Rotate 90° CCW then flip vertically

            # Import required message types
            from grid_map_msgs.msg import GridMapInfo
            from geometry_msgs.msg import Pose

            # Create GridMap message
            grid_map_msg = GridMap()

            # Set header
            grid_map_msg.header = Header()
            grid_map_msg.header.stamp = self.get_clock().now().to_msg()
            grid_map_msg.header.frame_id = self.world_frame

            # Create and populate GridMapInfo
            grid_map_msg.info = GridMapInfo()
            grid_map_msg.info.resolution = float(self.param.resolution)
            grid_map_msg.info.length_x = float(self.param.map_size)
            grid_map_msg.info.length_y = float(self.param.map_size)

            # Set map position (center at map_center position)
            grid_map_msg.info.pose = Pose()
            # Use stored map center position
            if self.map_center is not None:
                grid_map_msg.info.pose.position.x = float(self.map_center[0])
                grid_map_msg.info.pose.position.y = float(self.map_center[1])
                grid_map_msg.info.pose.position.z = 0.0
            else:
                # Fallback to origin if map center not set
                grid_map_msg.info.pose.position.x = 0.0
                grid_map_msg.info.pose.position.y = 0.0
                grid_map_msg.info.pose.position.z = 0.0
            grid_map_msg.info.pose.orientation.w = 1.0

            # Set layers
            grid_map_msg.layers = ['elevation', 'variance', semantic_layer]

            # Get grid dimensions
            height, width = elevation_data.shape

            # Flatten in row-major order (C-style) and convert to list
            elevation_flat = elevation_data.flatten(order='C')
            grid_map_msg.data.append(
                self._create_float32_multiarray(elevation_flat, height, width)
            )

            # Add variance layer
            variance_flat = variance_data.flatten(order='C')
            grid_map_msg.data.append(
                self._create_float32_multiarray(variance_flat, height, width)
            )

            # Add semantic layer (traversability data)
            semantic_layer_flat = semantic_layer_data.flatten(order='C')
            grid_map_msg.data.append(
                self._create_float32_multiarray(semantic_layer_flat, height, width)
            )

            # Set indices (default to 0)
            grid_map_msg.outer_start_index = 0
            grid_map_msg.inner_start_index = 0

            # Publish
            self.grid_map_pub.publish(grid_map_msg)

            # Semantic layer publishing disabled (data already included in grid_map)
            # self._publish_semantic_layer_map()

        except Exception as e:
            self.get_logger().warn(f'⚠️  Failed to publish elevation_mapping_cupy grid: {e}')

    def _create_float32_multiarray(self, data_flat, rows, cols):
        """Convert flattened numpy array to std_msgs/Float32MultiArray for GridMap.

        Args:
            data_flat: 1D numpy array of data
            rows: Number of rows in the grid
            cols: Number of columns in the grid

        Returns:
            Float32MultiArray message
        """
        from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

        msg = Float32MultiArray()
        msg.layout.dim = []

        # Row dimension (outer dimension)
        row_dim = MultiArrayDimension()
        row_dim.label = 'column_index'  # GridMap uses column as outer
        row_dim.size = cols
        row_dim.stride = rows * cols
        msg.layout.dim.append(row_dim)

        # Column dimension (inner dimension)
        col_dim = MultiArrayDimension()
        col_dim.label = 'row_index'  # GridMap uses row as inner
        col_dim.size = rows
        col_dim.stride = rows
        msg.layout.dim.append(col_dim)

        msg.layout.data_offset = 0
        msg.data = data_flat.tolist()

        return msg

    def _publish_transformed_pointcloud(self, points_np, original_header):
        """Publish transformed pointcloud for visualization/debugging.

        Args:
            points_np: numpy array of shape (N, 3) with transformed points
            original_header: Header from original pointcloud message
        """
        try:
            # Create PointCloud2 message
            pc_msg = PointCloud2()

            # Set header
            pc_msg.header = original_header
            pc_msg.header.frame_id = self.base_frame  # Points are now in base_link frame
            pc_msg.header.stamp = self.get_clock().now().to_msg()

            # Set pointcloud data
            pc_msg.height = 1
            pc_msg.width = len(points_np)
            pc_msg.is_dense = True

            # Define fields (x, y, z)
            from sensor_msgs.msg import PointField
            pc_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            pc_msg.point_step = 12  # 3 * 4 bytes
            pc_msg.row_step = pc_msg.point_step * pc_msg.width

            # Convert points to bytes
            points_flat = points_np.astype(np.float32).flatten()
            pc_msg.data = points_flat.tobytes()

            # Publish
            self.pointcloud_pub.publish(pc_msg)

        except Exception as e:
            self.get_logger().warn(f'Failed to publish transformed pointcloud: {e}')

    def _publish_map_center_pose(self):
        """Publish map center as PoseStamped message for visualization/debugging."""
        try:
            # Use stored map center position
            if self.map_center is None:
                self.get_logger().warn('Map center not set, cannot publish pose')
                return

            # Create PoseStamped message
            pose_msg = PoseStamped()

            # Set header
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.world_frame  # Map center is in world frame

            # Set position (map center coordinates)
            pose_msg.pose.position.x = float(self.map_center[0])
            pose_msg.pose.position.y = float(self.map_center[1])
            pose_msg.pose.position.z = float(self.map_center[2])

            # Set orientation (identity quaternion - no rotation)
            pose_msg.pose.orientation.w = 1.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0

            # Publish
            self.map_center_pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().warn(f'Failed to publish map center pose: {e}')

    def _publish_map_center_tf(self):
        """Publish TF transform from world_frame to map_center for visualization/debugging."""
        try:
            # Use stored map center position
            if self.map_center is None:
                self.get_logger().warn('Map center not set, cannot publish TF')
                return

            # Create TransformStamped message
            transform_msg = TransformStamped()

            # Set header
            transform_msg.header.stamp = self.get_clock().now().to_msg()
            transform_msg.header.frame_id = self.world_frame  # Parent frame
            transform_msg.child_frame_id = 'map_center'  # Child frame

            # Set translation (map center position)
            transform_msg.transform.translation.x = float(self.map_center[0])
            transform_msg.transform.translation.y = float(self.map_center[1])
            transform_msg.transform.translation.z = float(self.map_center[2])

            # Set rotation (identity quaternion - no rotation)
            transform_msg.transform.rotation.w = 1.0
            transform_msg.transform.rotation.x = 0.0
            transform_msg.transform.rotation.y = 0.0
            transform_msg.transform.rotation.z = 0.0

            # Broadcast the transform
            self.tf_broadcaster.sendTransform(transform_msg)

        except Exception as e:
            self.get_logger().warn(f'Failed to publish map center TF: {e}')

    def get_map_center(self):
        """Get the current map center position.

        Returns:
            numpy.ndarray or None: The map center position [x, y, z] in world frame, or None if not set
        """
        return self.map_center.copy() if self.map_center is not None else None

    @staticmethod
    def _quaternion_to_rotation_matrix(w, x, y, z):
        """
        Convert quaternion to 3x3 rotation matrix.

        Quaternion format: w + xi + yj + zk
        Standard formula for quaternion to rotation matrix conversion.

        Args:
            w, x, y, z: Quaternion components

        Returns:
            3x3 numpy rotation matrix
        """
        # Normalize quaternion
        norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

        # Convert to rotation matrix
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    def __del__(self):
        """Cleanup on node destruction."""
        if self.sem_map is not None:
            try:
                self.sem_map.close()
            except Exception:
                pass


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    try:
        node = SemMapNode()

        # Use MultiThreadedExecutor for better performance with callback groups
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
