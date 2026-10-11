# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Parameter class for semantic mapping pipeline.

Provides reflection-based parameter management for ROS parameter bridge.
Allows C++ to discover parameter names and types, then set values from ROS.
"""


class Parameter:
    """Parameter class with reflection methods for ROS integration."""

    def __init__(self):
        # Semantic pipeline parameters with defaults
        # Model defaults match config/sem_map.yaml and the artifacts download.sh fetches.
        self.backbone_name = "dinov3s"
        self.upsampler_type = "bilinear"
        self.upsample_partial_factor = 4
        self.seg_layer = "dinov3s_sem_seg_trail_v4"
        self.config_path = "config/models.yaml"

        # Additional parameters that might be needed
        self.latent_w = 32
        self.latent_h = 20
        self.patch_size = 16
        self.feature_dims = 384
        self.num_classes = 27

        # Processing parameters
        self.temperature = None
        # Visualization options
        self.enable_pca = True
        # Semantic pipeline
        self.enable_semantic_pipeline = True
        self.publish_traversability = True
        self.publish_segmentation = True
        self.traversability_temperature = 1.0
        self.semantic_bev_layer_name = "trail"
        self.flip_image = False  # Flip RGB images and outputs vertically

        # Elevation mapping
        self.enable_height_map = True
        self.enable_bev_fusion = True
        self.enable_inpaint = True  # Use inpainted elevation layer (elevation_inpaint) instead of raw elevation layer

        # Multiprocessing (Phase 1: experimental)
        self.use_multiprocessing = False  # Enable CUDA IPC multiprocessing architecture

        # Multiprocessing pipeline parameters
        self.image_height = 480
        self.image_width = 640
        self.trav_height = 320
        self.trav_width = 512
        self.pc_max_points = 100000
        self.buffer_size = 10

        # Map parameters
        self.resolution = 0.05
        self.map_size = 20.0
        self.backend = "cupy"
        self.precision = "float16"
        self.map_move_threshold_m = 1.0  # Only move map when robot moves > 1m from center

        # Confidence-based fusion
        self.confidence_fusion_threshold = 0.5  # Only fuse pixels with confidence >= this value (0.0-1.0). Gate for 'exponential' fusion and the _sem_observed inpainting trust stamp
        self.semantic_fusion_method = "exponential"  # BEV image fusion: 'exponential' (fixed alpha, hard gate) or 'confidence_weighted' (per-cell evidence weight)
        self.conf_floor = 0.2  # confidence_weighted: skip pixels below this confidence
        self.conf_weight_cap = 4.0  # confidence_weighted: max accumulated cell weight (rate-independent plasticity floor)
        self.semantic_fusion_range_m = 0.0  # confidence_weighted: distance falloff (inverse-variance), evidence scaled by 1/(1+(d/range)^4). 0 = auto (map_size/4), negative = disabled

        # Processing rates
        self.pointcloud_processing_rate_hz = 10.0
        self.rgb_processing_rate_hz = 30.0
        self.ros_publish_rate_hz = 30.0

        # Removed internal elevation-layer parameters (handled by elevation_mapping_cupy)

    def get_names(self):
        """Return list of parameter names for reflection."""
        return list(self.__dict__.keys())

    def get_types(self):
        """Return list of parameter types for ROS parameter conversion."""
        types = []
        for name in self.get_names():
            val = getattr(self, name)
            if isinstance(val, float):
                types.append("float")
            elif isinstance(val, str):
                types.append("str")
            elif isinstance(val, bool):
                types.append("bool")
            elif isinstance(val, int):
                types.append("int")
            else:
                # Default to string for unknown types
                types.append("str")
        return types

    def set_value(self, name, value):
        """Set a parameter value by name."""
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            # Add new parameter if it doesn't exist
            setattr(self, name, value)

    def get_value(self, name, default=None):
        """Get a parameter value by name, with optional default.

        If the attribute does not exist, or is None, return the provided default.
        """
        if not hasattr(self, name):
            return default
        val = getattr(self, name)
        return default if (val is None and default is not None) else val

    def update(self):
        """Optional: validation, derived parameter calculation, etc."""
        # Validate parameter ranges
        if self.upsample_partial_factor <= 0:
            raise ValueError("upsample_partial_factor must be positive")

        if self.latent_w <= 0 or self.latent_h <= 0:
            raise ValueError("latent dimensions must be positive")

        # Validate map parameters
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")

        if self.map_size <= 0:
            raise ValueError("map_size must be positive")

        if self.backend not in ["numpy", "cupy"]:
            raise ValueError("backend must be 'numpy' or 'cupy'")

        if self.precision not in ["float16", "float32"]:
            raise ValueError("precision must be 'float16' or 'float32'")

        # Set derived parameters
        if self.backbone_name == "dinov3s":
            self.patch_size = 16
            self.feature_dims = 384
        elif self.backbone_name == "dinov2s":
            self.patch_size = 14
            self.feature_dims = 384
        else:
            # Default values
            self.patch_size = 16
            self.feature_dims = 384
