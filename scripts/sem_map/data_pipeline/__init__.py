# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Data Pipeline - CUDA IPC Multiprocessing Infrastructure

Zero-copy GPU data transfer between C++ ROS node and Python worker processes.

Key Components:
- cuda_utils: CUDA IPC base classes and utilities
- cuda_buffer: CUDATimeBuffer for timestamp-based latest-item access
- data_pipeline_manager: Central orchestrator for multiprocessing architecture
"""

from .cuda_utils import checkCudaErrors
from .cuda_buffer import CUDATimeBuffer
from .data_pipeline_manager import DataPipelineManager

__all__ = [
    'checkCudaErrors',
    'CUDATimeBuffer',
    'DataPipelineManager',
]
