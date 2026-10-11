# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
sem_map Python package

This package contains runtime scripts and utilities for the semantic mapping
pipeline. It is installed under share/sem_map/scripts and imported as
`sem_map` at runtime by the C++ bindings.
"""

# CRITICAL: Set multiprocessing start method for CUDA compatibility
# Must be done ONCE per interpreter, BEFORE any multiprocessing usage
# Guarded to prevent errors in ROS/pybind environments with multiple imports
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

from .parameter import Parameter
from .semantic_pipeline import SemanticPipeline
from .sem_map import SemMap

__all__ = ['Parameter', 'SemanticPipeline', 'SemMap']
