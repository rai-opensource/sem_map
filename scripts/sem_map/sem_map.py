# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Unified semantic mapping class combining SemanticPipeline and ElevationMapping.

This class follows the visual-perception-engine pattern of a single Python class
managing multiple components (semantic pipeline + elevation mapping).
"""

import os
import sys
from typing import Optional, Dict, List, Tuple
import numpy as np
import yaml

from .semantic_pipeline import SemanticPipeline
from .parameter import Parameter

# Try to import elevation_mapping_cupy
try:
    from elevation_mapping_cupy import ElevationMap, Parameter as ElevationMappingParameter
    HAS_ELEVATION_MAPPING = True
except ImportError:
    HAS_ELEVATION_MAPPING = False
    ElevationMap = None
    ElevationMappingParameter = None

# Try to import CuPy
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np


def _dbg(msg: str):
    """Debug print to stderr."""
    try:
        sys.stderr.write(f"[sem_map][SemMap] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


class SemMap:
    """Unified semantic mapping class combining SemanticPipeline and ElevationMapping.

    This class manages both semantic processing (RGB → traversability) and
    elevation mapping (pointcloud → elevation map) in a single interface.
    """

    def __init__(self, param: Parameter):
        """Initialize the unified semantic map.

        Args:
            param: Parameter instance containing all configuration
        """
        self.param = param
        self.traversability_temperature = param.get_value('traversability_temperature', 1.0)
        self.semantic_bev_layer_name = param.get_value('semantic_bev_layer_name', 'trail')
        _dbg(f"Traversability temperature set to {self.traversability_temperature}")
        _dbg(f"Semantic BEV layer name: {self.semantic_bev_layer_name}")

        # Track last map center position for smart map moving
        self.last_map_center = None
        self.map_move_threshold = param.get_value('map_move_threshold_m', 1.0)

        # Set up shared CuPy memory pool for efficient GPU memory management
        if HAS_CUPY:
            self.cupy_pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            cp.cuda.set_allocator(self.cupy_pool.malloc)
            _dbg("CuPy memory pool initialized")

        # Initialize semantic pipeline if enabled
        self.semantic_pipeline = None
        if param.get_value('enable_semantic_pipeline', True):
            _dbg("Initializing SemanticPipeline...")
            try:
                self.semantic_pipeline = SemanticPipeline(param)
                _dbg("SemanticPipeline initialized successfully")
            except Exception as e:
                _dbg(f"Failed to initialize SemanticPipeline: {e}")
                raise

        # Initialize elevation mapping if enabled
        self.elevation_map = None
        enable_height_map = param.get_value('enable_height_map', True)
        if enable_height_map and param.backend == "cupy" and HAS_ELEVATION_MAPPING:
            _dbg("Initializing ElevationMap...")
            try:
                # Load elevation mapping configuration
                from ament_index_python.packages import get_package_share_directory
                sem_map_share_dir = get_package_share_directory('sem_map')
                core_param_path = os.path.join(sem_map_share_dir, 'config', 'elevation_mapping', 'core_param.yaml')
                plugin_config_path = os.path.join(sem_map_share_dir, 'config', 'elevation_mapping', 'plugin_config.yaml')

                with open(core_param_path) as f:
                    elevation_config = yaml.safe_load(f)

                elevation_config["plugin_config_file"] = plugin_config_path

                # Create elevation mapping parameter
                param_cupy = ElevationMappingParameter(**elevation_config)
                param_cupy.resolution = param.resolution
                param_cupy.map_length = param.map_size
                param_cupy.plugin_config_file = plugin_config_path
                param_cupy.additional_layers = [self.semantic_bev_layer_name]
                param_cupy.fusion_algorithms = ["image_exponential"]
                param_cupy.image_channel_fusions = {self.semantic_bev_layer_name: "exponential"}

                param_cupy.update()  # Update derived parameters
                self.elevation_map = ElevationMap(param_cupy)

                _dbg("ElevationMap initialized successfully")
            except Exception as e:
                _dbg(f"Failed to initialize ElevationMap: {e}")
                self.elevation_map = None
        elif not HAS_ELEVATION_MAPPING:
            _dbg("elevation_mapping_cupy not available, skipping elevation map initialization")
        else:
            _dbg("Height mapping disabled via enable_height_map parameter")

        # Validate configuration
        enable_bev_fusion = param.get_value('enable_bev_fusion', True)
        if enable_bev_fusion and self.elevation_map is None:
            print("[SemMap] WARNING: enable_bev_fusion=true but elevation map not initialized. BEV fusion will be skipped.")
            print("[SemMap]          Set enable_height_map=true to enable elevation mapping.")

        # Camera info storage for image projection
        self.camera_info = None

    def set_camera_info(self, K: np.ndarray, D: np.ndarray, distortion_model: str,
                       width: int, height: int):
        """Store camera intrinsics for image projection.

        Args:
            K: 3x3 camera intrinsic matrix
            D: Distortion coefficients
            distortion_model: Distortion model name (e.g., "plumb_bob")
            width: Image width
            height: Image height
        """
        self.camera_info = {
            'K': K,
            'D': D,
            'distortion_model': distortion_model,
            'width': width,
            'height': height
        }
        _dbg(f"Camera info stored: {width}x{height}, K={K.flatten()[:3]}")

    def process_frame(self, image_data, height: int, width: int, channels: int,
                     sensor_name: str, stamp_ns: int,
                     pose_R: Optional[np.ndarray] = None,
                     pose_t: Optional[np.ndarray] = None):
        """Process RGB frame through semantic pipeline and optionally fuse to elevation map.

        Args:
            image_data: Raw image data (bytes, bytearray, or numpy array)
            height: Image height in pixels
            width: Image width in pixels
            channels: Number of channels (should be 3 for RGB)
            sensor_name: Name of the sensor for logging
            stamp_ns: Timestamp in nanoseconds
            pose_R: Optional 3x3 rotation matrix (camera to world)
            pose_t: Optional 3-element translation vector (camera position in world)
        """
        # Process through semantic pipeline
        if self.semantic_pipeline is not None:
            self.semantic_pipeline.process_frame(
                image_data, height, width, channels, sensor_name, stamp_ns
            )

            # If BEV fusion enabled, elevation map and pose are available, fuse traversability
            enable_bev_fusion = self.param.get_value('enable_bev_fusion', True)
            if (enable_bev_fusion and
                self.elevation_map is not None and
                pose_R is not None and pose_t is not None and
                self.camera_info is not None and
                self.semantic_pipeline.last_traversability is not None):

                self._fuse_traversability_to_elevation(pose_R, pose_t, height, width)
            # else:
            #     # Debug: show which condition failed
            #     _dbg(f"process_frame: no fusion - elevation_map={self.elevation_map is not None}, "
            #          f"pose_R={pose_R is not None}, pose_t={pose_t is not None}, "
            #          f"camera_info={self.camera_info is not None}, "
            #          f"last_trav={self.semantic_pipeline.last_traversability is not None}")
        # else:
        #     _dbg("process_frame: no semantic pipeline")

    def _fuse_traversability_to_elevation(self, pose_R: np.ndarray, pose_t: np.ndarray,
                                         image_height: int, image_width: int):
        """Fuse traversability from semantic pipeline into elevation map.

        Args:
            pose_R: 3x3 rotation matrix (camera to world)
            pose_t: 3-element translation vector (camera position in world)
            image_height: Height of the original image
            image_width: Width of the original image
        """
        try:
            trav_map = self.semantic_pipeline.last_traversability

            # Get camera intrinsics
            K = self.camera_info['K']
            D = self.camera_info['D']
            distortion_model = self.camera_info['distortion_model']
            cam_width = self.camera_info['width']
            cam_height = self.camera_info['height']

            # Get traversability dimensions
            trav_height, trav_width = trav_map.shape

            # Scale camera intrinsics to match traversability image size
            scale_x = trav_width / cam_width
            scale_y = trav_height / cam_height

            K_scaled = K.copy()
            K_scaled[0, 0] *= scale_x  # fx
            K_scaled[1, 1] *= scale_y  # fy
            K_scaled[0, 2] *= scale_x  # cx
            K_scaled[1, 2] *= scale_y  # cy

            # Convert camera-to-world transform to world-to-camera for projection
            R_world_to_cam = pose_R.T
            t_world_in_cam = -R_world_to_cam @ pose_t

            # Prepare traversability image with correct shape (D, H, W)
            # Zero-copy PyTorch→CuPy via DLPack (GPU-only path)
            if HAS_CUPY:
                # Clamp and ensure contiguous on GPU
                trav_gpu = trav_map.clamp(0, 1).contiguous()
                # Zero-copy conversion via DLPack
                trav_cp = cp.from_dlpack(trav_gpu.__dlpack__())
                trav_image = cp.expand_dims(trav_cp, axis=0)
            else:
                # Fallback to CPU path
                trav_np = trav_map.clamp(0, 1).cpu().numpy().astype(np.float32)
                trav_image = np.expand_dims(trav_np, axis=0)

            # Add traversability to elevation map
            self.elevation_map.input_image(
                image=trav_image,
                channels=[self.semantic_bev_layer_name],
                R=R_world_to_cam.astype(np.float64),
                t=t_world_in_cam.astype(np.float64),
                K=K_scaled,
                D=D,
                distortion_model=distortion_model,
                image_height=trav_height,
                image_width=trav_width,
            )

        except Exception as e:
            _dbg(f"Failed to fuse traversability to elevation map: {e}")

    def process_pointcloud(self, points: np.ndarray, channels: List[str],
                          R: np.ndarray, t: np.ndarray):
        """Process pointcloud and update elevation map.

        Args:
            points: Nx3 numpy array of points in sensor frame
            channels: List of channel names (e.g., ["x", "y", "z"])
            R: 3x3 rotation matrix (base to world)
            t: 3-element translation vector (base position in world)
        """
        if self.elevation_map is None:
            _dbg("process_pointcloud: elevation_map is None")
            return False

        try:
            # Input pointcloud to elevation map
            self.elevation_map.input_pointcloud(
                points, channels, R.astype(np.float64), t.astype(np.float64), 0.0, 0.0
            )

            # Update normal and variance
            self.elevation_map.update_normal(self.elevation_map.elevation_map[0])
            self.elevation_map.update_variance()
            self.elevation_map.update_time()

            return True

        except Exception as e:
            import traceback
            _dbg(f"Failed to process pointcloud: {e}")
            _dbg(f"Traceback: {traceback.format_exc()}")
            return False

    def move_to(self, position: np.ndarray, rotation: np.ndarray):
        """Update map center to robot position (only if moved beyond threshold).

        Args:
            position: 3-element position vector [x, y, z]
            rotation: 3x3 rotation matrix
        """
        if self.elevation_map is None:
            return

        # Check if we need to move the map
        should_move = False
        if self.last_map_center is None:
            # First time - always move
            should_move = True
        else:
            # Calculate distance from last map center
            distance = np.linalg.norm(position[:2] - self.last_map_center[:2])  # 2D distance
            if distance > self.map_move_threshold:
                should_move = True

        if not should_move:
            return

        try:
            self.elevation_map.move_to(
                position.astype(np.float64),
                rotation.astype(np.float64)
            )

            # Update last map center
            self.last_map_center = position.copy()

        except Exception as e:
            _dbg(f"Failed to move map center: {e}")

    def get_last_semantic_results(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get traversability and segmentation from semantic pipeline.

        Returns:
            Tuple of (traversability, segmentation) numpy arrays, or None if unavailable.
        """
        if self.semantic_pipeline is None:
            return None

        trav = self.semantic_pipeline.last_traversability
        seg = self.semantic_pipeline.last_segmentation

        if trav is None or seg is None:
            return None

        # Ensure numpy float32 copies for downstream publishing
        if hasattr(trav, 'detach'):
            trav = trav.detach()
        if hasattr(seg, 'detach'):
            seg = seg.detach()

        trav_np = trav.float().cpu().numpy()
        seg_np = seg.float().cpu().numpy()
        return trav_np, seg_np

    def get_last_traversability(self) -> Optional[np.ndarray]:
        """Backward compatible helper returning traversability only."""
        result = self.get_last_semantic_results()
        if result is None:
            return None
        trav_np, _ = result
        return trav_np

    def get_elevation_map_layer(self, layer_name: str) -> Optional[np.ndarray]:
        """Get specific layer from elevation map.

        Args:
            layer_name: Name of the layer (e.g., "elevation", "variance", "trail")

        Returns:
            Layer data as numpy array, or None if not available
        """
        if self.elevation_map is None:
            return None

        try:
            if HAS_CUPY:
                # Get CuPy array directly (stays on GPU), single GPU→CPU transfer
                m = self.elevation_map.get_map_with_name_ref(layer_name, return_cupy=True)
                if m is None:
                    return None
                return cp.asnumpy(m)
            else:
                cell_n = int(self.elevation_map.cell_n)
                h, w = max(cell_n - 2, 0), max(cell_n - 2, 0)
                layer_data = np.zeros((h, w), dtype=np.float32)
                self.elevation_map.get_map_with_name_ref(layer_name, layer_data)
                return layer_data

        except Exception as e:
            _dbg(f"Failed to get elevation map layer '{layer_name}': {e}")
            return None

    def get_elevation_layers_transformed(self, layer_names: List[str]) -> Optional[Dict[str, np.ndarray]]:
        """Get multiple elevation map layers with rotation and flip applied (for GridMap ROS format).

        Uses return_cupy=True to keep data on GPU, applies transforms there,
        then does a single GPU→CPU transfer per layer.

        Args:
            layer_names: List of layer names to retrieve (e.g., ["elevation", "variance", "trail"])

        Returns:
            Dictionary mapping layer names to transformed numpy arrays (row-major, contiguous),
            or None if elevation map not available.
        """
        if self.elevation_map is None:
            return None

        try:
            result = {}

            for layer_name in layer_names:
                if HAS_CUPY:
                    # Get CuPy array directly (stays on GPU, no CPU transfer)
                    m = self.elevation_map.get_map_with_name_ref(layer_name, return_cupy=True)
                    if m is None:
                        continue
                    # Apply transforms on GPU, then single GPU→CPU transfer
                    transformed = m.T
                    result[layer_name] = cp.asnumpy(cp.ascontiguousarray(transformed))
                else:
                    # CPU fallback
                    cell_n = int(self.elevation_map.cell_n)
                    h, w = max(cell_n - 2, 0), max(cell_n - 2, 0)
                    layer_data = np.zeros((h, w), dtype=np.float32)
                    self.elevation_map.get_map_with_name_ref(layer_name, layer_data)
                    result[layer_name] = np.ascontiguousarray(layer_data.T)

            return result

        except Exception as e:
            _dbg(f"Failed to get transformed elevation map layers: {e}")
            return None

    def close(self):
        """Cleanup resources."""
        _dbg("Closing SemMap...")

        # Close semantic pipeline
        if self.semantic_pipeline is not None:
            try:
                self.semantic_pipeline.close()
            except Exception as e:
                _dbg(f"Error closing semantic pipeline: {e}")

        # Elevation map doesn't need explicit cleanup

        _dbg("SemMap closed")
