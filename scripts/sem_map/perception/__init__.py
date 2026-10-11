# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Core utilities - lightweight, fast imports only."""

# Common utilities (no heavy deps)
from .common import (
    get_device,
    prepare_image,
    unnormalize,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# Keep heavy ML imports in submodules - users import explicitly:
# from sem_map.perception.models.dinov import load_dinovx_torch
# from sem_map.perception.models.jafar import load_jafar_torch
# from sem_map.perception.models.segmentation import load_seg_head_torch, COCOSTUFF_27_CATEGORIES
