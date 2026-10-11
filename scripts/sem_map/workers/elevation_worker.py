# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Elevation Mapping Worker Process

Runs in separate Python process with independent GIL.
Handles:
- Pointcloud ingestion and elevation map updates
- Traversability fusion from semantic worker
- Output of fused elevation map layers
"""

import sys
import time
import signal
from time import sleep
from collections import deque
import torch
import torch.multiprocessing as mp
from cuda.bindings import driver as cuda


def _dbg(msg: str):
    """Print to stderr for visibility in ROS logs"""
    print(f"[ElevationWorker] {msg}", file=sys.stderr, flush=True)


def checkCudaErrors(result):
    """Check CUDA error codes"""
    if result[0].value != 0:
        raise RuntimeError(f"CUDA error code={result[0].value}({result[1]})")


def _extract_elevation_layers(elevation_map, param, cp, np):
    """Extract and transform elevation map layers and map center for output.

    Args:
        elevation_map: ElevationMap instance from elevation_mapping_cupy
        param: Parameter instance
        cp: cupy module
        np: numpy module

    Returns:
        Dictionary containing:
        - layer_name -> torch.Tensor (on CUDA) for each layer
        - 'map_center' -> torch.Tensor (on CUDA) with shape (3,)
    """
    semantic_layer_name = param.get_value('semantic_bev_layer_name', 'trail')
    enable_inpaint = param.get_value('enable_inpaint', True)
    layer_names = ['elevation', 'variance', semantic_layer_name]
    elevation_layer_name = 'elevation_inpaint' if enable_inpaint else 'elevation'

    result = {}
    # NOTE: Order matters. When joint_inpainting is enabled, reading
    # 'elevation_inpaint' triggers the plugin which inpaints semantic layers
    # in-place. Elevation must be read before the semantic layer so that the
    # semantic read picks up the inpainted values.
    for layer_name in layer_names:
        if layer_name != 'elevation' and not elevation_map.exists_layer(layer_name):
            continue

        actual_name = elevation_layer_name if layer_name == 'elevation' else layer_name
        m = elevation_map.get_map_with_name_ref(actual_name, return_cupy=True)
        if m is None:
            continue

        transformed = m.T
        result[layer_name] = torch.as_tensor(cp.ascontiguousarray(transformed), device='cuda')

    # Map center: zero-copy CuPy→PyTorch if possible
    center = elevation_map.center
    if isinstance(center, cp.ndarray):
        result['map_center'] = torch.as_tensor(center, device='cuda').float()
    else:
        result['map_center'] = torch.from_numpy(np.asarray(center, dtype=np.float32)).cuda()

    return result


def elevation_worker_main(pc_buffer, sem_results_buffer, elev_buffer, param, kill_event, ready_event=None, reset_event=None):
    """
    Main loop for elevation mapping worker.

    Args:
        pc_buffer: CUDATimeBuffer for pointcloud input
        sem_results_buffer: CUDATimeBuffer for traversability + segmentation input
        elev_buffer: CUDATimeBuffer for elevation map output
        param: Parameter instance (from sem_map.parameter)
        kill_event: mp.Event for shutdown signal
        ready_event: optional mp.Event signalled once the worker is ready
        reset_event: optional mp.Event; when set, clears the ElevationMap and is
            cleared again so subsequent triggers can request another reset.
    """
    # Register signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        _dbg(f"Received signal {signum}, setting kill_event")
        kill_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize CUDA (let PyTorch create context)
        checkCudaErrors(cuda.cuInit(0))
        torch.cuda.set_device(0)
        # Force primary context creation NOW: set_device alone may leave no
        # current driver context, and cuMemImportFromShareableHandle below
        # requires one (fails with CUDA_ERROR_INVALID_CONTEXT otherwise).
        _ = torch.zeros(1, device='cuda')

        # Receive CUDA IPC handles (CRITICAL!)
        pc_buffer.recv_shareable_handles()
        sem_results_buffer.recv_shareable_handles()
        elev_buffer.recv_shareable_handles()

        # Initialize elevation_mapping_cupy
        import numpy as np
        import cupy as cp
        import yaml
        import os
        from elevation_mapping_cupy import ElevationMap, Parameter as ElevationMappingParameter
        from ament_index_python.packages import get_package_share_directory

        # Load elevation mapping configuration
        sem_map_share_dir = get_package_share_directory('sem_map')
        core_param_path = os.path.join(sem_map_share_dir, 'config', 'elevation_mapping', 'core_param.yaml')
        plugin_config_path = os.path.join(sem_map_share_dir, 'config', 'elevation_mapping', 'plugin_config.yaml')

        with open(core_param_path) as f:
            elevation_config = yaml.safe_load(f)

        elevation_config["plugin_config_file"] = plugin_config_path

        # Create elevation mapping parameter (following sem_map.py pattern)
        param_cupy = ElevationMappingParameter(**elevation_config)
        param_cupy.resolution = param.resolution
        param_cupy.map_length = param.map_size
        param_cupy.plugin_config_file = plugin_config_path
        semantic_layer_name = param.get_value('semantic_bev_layer_name', 'trail')
        param_cupy.additional_layers = [semantic_layer_name]
        # Fusion method: 'exponential' (fixed alpha + hard confidence gate) or
        # 'confidence_weighted' (per-cell evidence weight; see image_confidence_weighted.py)
        fusion_method = param.get_value('semantic_fusion_method', 'exponential')
        param_cupy.fusion_algorithms = [f"image_{fusion_method}"]
        param_cupy.image_channel_fusions = {semantic_layer_name: fusion_method}
        param_cupy.conf_floor = param.get_value('conf_floor', 0.2)
        param_cupy.conf_weight_cap = param.get_value('conf_weight_cap', 4.0)
        param_cupy.semantic_fusion_range_m = param.get_value('semantic_fusion_range_m', 0.0)
        _dbg(f"Semantic BEV fusion method: {fusion_method} "
             f"(distance falloff range: {param_cupy.semantic_fusion_range_m} m, 0=auto)")

        # Ensure subscriber_cfg has channels set for semantic layer registration
        # Semantic layers are only registered if they're in subscriber_cfg channels (see elevation_mapping.py:145-162)
        # Update subscriber_cfg to include the semantic layer in channels
        if "subscriber_cfg" in elevation_config and elevation_config["subscriber_cfg"]:
            param_cupy.subscriber_cfg = elevation_config["subscriber_cfg"]
            # Ensure semantic layer is in channels for image subscribers
            for cfg in param_cupy.subscriber_cfg.values():
                if cfg.get("data_type") == "image":
                    if "channels" not in cfg:
                        cfg["channels"] = []
                    if semantic_layer_name not in cfg["channels"]:
                        cfg["channels"].append(semantic_layer_name)
        else:
            # Create minimal subscriber_cfg if not present
            param_cupy.subscriber_cfg = {
                "image_sub": {
                    "data_type": "image",
                    "channels": [semantic_layer_name]
                }
            }

        # Set confidence threshold from param (loaded from sem_map.yaml)
        param_cupy.confidence_fusion_threshold = param.confidence_fusion_threshold
        _dbg(f"Using confidence_fusion_threshold={param.confidence_fusion_threshold}")

        # CRITICAL: Update derived parameters (computes cell_n, etc.)
        param_cupy.update()

        # Create elevation map instance
        elevation_map = ElevationMap(param_cupy)

        # Metrics - sliding window statistics (last 50 frames)
        pc_frame_count = 0
        trav_fusion_count = 0
        loop_iteration_count = 0
        process_times = deque(maxlen=50)  # Stores last 50 active loop times (excludes idle)
        pc_times = deque(maxlen=50)  # Stores last 50 pointcloud processing times
        fusion_times = deque(maxlen=50)  # Stores last 50 fusion times
        layer_extraction_times = deque(maxlen=50)  # Stores last 50 layer extraction times
        pc_timestamps = deque(maxlen=50)  # Wall-clock timestamps for windowed PC FPS
        fusion_timestamps = deque(maxlen=50)  # Wall-clock timestamps for windowed fusion FPS
        last_logged_pc_count = 0  # Track last logged frame count to prevent duplicate logs

        # Timestamp tracking for freshness (slot write times, perf_counter domain)
        last_pc_timestamp = float('-inf')
        last_trav_timestamp = float('-inf')
        # Last sensor timestamp seen (seconds, sensor/ROS clock domain) - used to
        # stamp output maps when an iteration republishes without new input
        last_sensor_timestamp = None

        # Track if map has been initialized (at least one pointcloud processed)
        map_initialized = False

        # Map move threshold (matching legacy behavior)
        map_move_threshold = param.map_move_threshold_m

        # Cache map center on CPU to avoid GPU→CPU transfer on every check
        cached_map_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Pre-allocate constant GPU arrays to avoid per-frame allocations
        identity_3x3 = cp.eye(3, dtype=elevation_map.data_type)
        D_zeros = cp.zeros(5, dtype=cp.float64)
        pc_channels = ["x", "y", "z"]

        # Rate control: get processing rate from param (default to 10 Hz if not set)
        processing_rate_hz = getattr(param, 'pointcloud_processing_rate_hz', 10.0)
        period = 1.0 / processing_rate_hz if processing_rate_hz > 0 else 0.1

        # Signal that worker is ready
        if ready_event is not None:
            ready_event.set()

        while not kill_event.is_set():
            loop_start = time.perf_counter()

            if reset_event is not None and reset_event.is_set():
                elevation_map.clear()
                reset_event.clear()
                _dbg("ElevationMap reset via reset_event")

            pc_result = None
            trav_result = None

            # ====================================================================
            # PART 1: Process pointcloud updates (if available)
            # ====================================================================
            pc_result = pc_buffer.get(last_pc_timestamp)

            if pc_result is not None:

                # Unpack pointcloud data. The item id carries the sensor
                # timestamp; pc_timestamp is the slot write time (freshness gate).
                pc_sensor_timestamp, pc_timestamp, pc_dict = pc_result
                last_pc_timestamp = pc_timestamp
                last_sensor_timestamp = pc_sensor_timestamp

                pc_start = time.perf_counter()

                # Extract pointcloud and metadata
                pointcloud = pc_dict['pointcloud']  # (N, 3) on CUDA
                num_points = int(pc_dict['num_points'].item())  # Scalar, convert to int for slicing
                pose_R = pc_dict['pose_R']  # (3, 3) on CUDA
                pose_t = pc_dict['pose_t']  # (3,) on CUDA

                # Trim to actual points (remove zero-padding)
                pc_valid = pointcloud[:num_points]  # (actual_N, 3)

                # Convert PyTorch GPU tensors to CuPy via DLPack (zero-copy, stays on GPU)
                # elevation_mapping_cupy accepts CuPy arrays directly
                pc_cp = cp.from_dlpack(pc_valid.contiguous())

                # Undo the transport transpose (data_pipeline_manager transposes for Eigen col-major fix)
                pose_R_cp = cp.from_dlpack(pose_R.contiguous()).T
                pose_t_cp = cp.from_dlpack(pose_t.contiguous())

                # Small CPU copy of pose_t (3 floats) for threshold check against cached_map_center
                pose_t_cpu = pose_t.cpu().numpy()

                # Move map center if robot moved beyond threshold
                should_move = False
                if np.allclose(cached_map_center, 0.0):
                    should_move = True
                else:
                    distance = np.linalg.norm(pose_t_cpu[:2] - cached_map_center[:2])
                    if distance > map_move_threshold:
                        should_move = True

                if should_move:
                    elevation_map.move_to(pose_t_cp, identity_3x3)
                    if isinstance(elevation_map.center, cp.ndarray):
                        cached_map_center[:] = cp.asnumpy(elevation_map.center)
                    else:
                        cached_map_center[:] = np.asarray(elevation_map.center)

                # Input pointcloud to elevation map (all CuPy, no CPU transfer)
                elevation_map.input_pointcloud(
                    pc_cp, pc_channels,
                    pose_R_cp,
                    pose_t_cp,
                    0.0, 0.0  # min/max range (0 = no filtering)
                )

                # Update normal and variance
                elevation_map.update_normal(elevation_map.elevation_map[0])
                elevation_map.update_variance()
                elevation_map.update_time()

                pc_time = (time.perf_counter() - pc_start) * 1000
                pc_times.append(pc_time)  # Append to sliding window
                pc_timestamps.append(time.perf_counter())
                pc_frame_count += 1

                # Mark map as initialized after first pointcloud is processed
                map_initialized = True

            # ====================================================================
            # PART 2: Process traversability fusion (if map is initialized)
            # ====================================================================
            # Track if we processed traversability this iteration (for timestamp selection)
            processed_trav_this_iter = False
            trav_timestamp_this_iter = None

            # Only attempt fusion if map has been initialized (at least one pointcloud processed)
            # Drain all pending trav frames to avoid falling behind when trav rate > loop rate
            while map_initialized:
                fusion_start = time.perf_counter()
                trav_result = sem_results_buffer.get(last_trav_timestamp)

                if trav_result is None:
                    break

                # Unpack semantic data (item id = sensor timestamp of source frame)
                trav_sensor_timestamp, trav_timestamp, trav_dict = trav_result
                last_trav_timestamp = trav_timestamp
                trav_timestamp_this_iter = trav_sensor_timestamp
                last_sensor_timestamp = trav_sensor_timestamp
                processed_trav_this_iter = True

                # Extract traversability, confidence, and camera info
                traversability = trav_dict['traversability']  # (H, W) on CUDA
                confidence = trav_dict.get('confidence', None)  # (H, W) on CUDA, may be None
                trav_pose_R = trav_dict['pose_R']  # (3, 3) on CUDA
                trav_pose_t = trav_dict['pose_t']  # (3,) on CUDA
                intrinsics = trav_dict['intrinsics']  # (3, 3) on CUDA

                # Convert pose/intrinsics to CuPy via DLPack (zero-copy, stays on GPU)
                # Undo transport transpose, then compute world-to-camera
                trav_pose_R_cp = cp.from_dlpack(trav_pose_R.contiguous()).T  # undo transport transpose
                trav_pose_t_cp = cp.from_dlpack(trav_pose_t.contiguous())
                intrinsics_cp = cp.from_dlpack(intrinsics.contiguous()).T    # undo transport transpose

                # Invert pose: camera-to-world → world-to-camera (for camera projection)
                R_world_to_cam = trav_pose_R_cp.T
                t_world_in_cam = -R_world_to_cam @ trav_pose_t_cp

                # Prepare traversability image with correct shape (1, H, W)
                # Zero-copy PyTorch→CuPy via DLPack
                trav_gpu = traversability.clamp(0, 1).contiguous()
                trav_cp = cp.from_dlpack(trav_gpu)
                trav_image = cp.expand_dims(trav_cp, axis=0)

                # Prepare confidence image with correct shape (1, H, W)
                conf_image = None
                if confidence is not None:
                    conf_gpu = confidence.clamp(0, 1).contiguous()
                    conf_cp = cp.from_dlpack(conf_gpu)
                    conf_image = cp.expand_dims(conf_cp, axis=0)

                trav_height, trav_width = traversability.shape

                # Scale intrinsics from input image resolution to traversability resolution
                scale_x = trav_width / param.image_width
                scale_y = trav_height / param.image_height

                K_scaled = intrinsics_cp.copy()
                K_scaled[0, 0] *= scale_x  # fx
                K_scaled[1, 1] *= scale_y  # fy
                K_scaled[0, 2] *= scale_x  # cx
                K_scaled[1, 2] *= scale_y  # cy

                # Add traversability to elevation map (all CuPy, no CPU transfer)
                elevation_map.input_image(
                    image=trav_image,
                    channels=[semantic_layer_name],
                    confidence=conf_image,
                    R=R_world_to_cam,
                    t=t_world_in_cam,
                    K=K_scaled,
                    D=D_zeros,
                    distortion_model="plumb_bob",
                    image_height=trav_height,
                    image_width=trav_width,
                )

                trav_fusion_count += 1
                fusion_time = (time.perf_counter() - fusion_start) * 1000
                fusion_times.append(fusion_time)
                fusion_timestamps.append(time.perf_counter())

            # Get elevation map layers and map center (includes fused traversability)
            # map_center is extracted directly from elevation_map.center (authoritative source)
            # Only extract and publish if map has been initialized (at least one pointcloud processed)
            if map_initialized:
                layer_extraction_start = time.perf_counter()
                layers = _extract_elevation_layers(elevation_map, param, cp, np)
                layer_extraction_time = (time.perf_counter() - layer_extraction_start) * 1000
                layer_extraction_times.append(layer_extraction_time)  # Append to sliding window

                # Write to output buffer (CUDATimeBuffer automatically overwrites old data)
                # Stamp with the sensor timestamp of the newest ingested data so the
                # C++ publisher gets stamps in the sensor/ROS clock domain.
                # Prefer pointcloud, then traversability, then last known sensor time.
                if pc_result is not None:
                    output_timestamp = pc_sensor_timestamp
                elif processed_trav_this_iter and trav_timestamp_this_iter is not None:
                    output_timestamp = trav_timestamp_this_iter
                else:
                    # map_initialized implies at least one pointcloud was processed,
                    # so last_sensor_timestamp is always set here
                    output_timestamp = last_sensor_timestamp
                elev_buffer.put(layers, output_timestamp, sync=True)

            # Record total processing time (only when work was done)
            did_work = pc_result is not None or processed_trav_this_iter
            if did_work:
                process_time = (time.perf_counter() - loop_start) * 1000
                process_times.append(process_time)
            loop_iteration_count += 1

            # Sleep to maintain processing rate
            loop_elapsed = time.perf_counter() - loop_start
            if loop_elapsed < period:
                sleep(period - loop_elapsed)

            # Log metrics periodically (every 50 PC frames)
            if pc_frame_count >= last_logged_pc_count + 50 and pc_frame_count > 0:
                # Calculate averages from sliding windows
                avg_total = sum(process_times) / len(process_times) if process_times else 0.0
                avg_pc = sum(pc_times) / len(pc_times) if pc_times else 0.0
                avg_fusion = sum(fusion_times) / len(fusion_times) if fusion_times else 0.0
                avg_layer_extraction = sum(layer_extraction_times) / len(layer_extraction_times) if layer_extraction_times else 0.0

                # Windowed FPS
                if len(pc_timestamps) >= 2:
                    pc_hz = (len(pc_timestamps) - 1) / (pc_timestamps[-1] - pc_timestamps[0])
                else:
                    pc_hz = 0.0
                if len(fusion_timestamps) >= 2:
                    fusion_hz = (len(fusion_timestamps) - 1) / (fusion_timestamps[-1] - fusion_timestamps[0])
                else:
                    fusion_hz = 0.0

                _dbg(f"Processed {pc_frame_count} PC frames, {trav_fusion_count} fusions "
                     f"(PC: {pc_hz:.1f} Hz, Fusion: {fusion_hz:.1f} Hz) | "
                     f"Total: {avg_total:.2f}ms | PC: {avg_pc:.2f}ms | Fusion: {avg_fusion:.2f}ms | Layers: {avg_layer_extraction:.2f}ms")
                last_logged_pc_count = pc_frame_count

    except Exception as e:
        _dbg(f"FATAL ERROR in elevation_worker_main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up CUDA resources before exit
        try:
            # Synchronize CUDA to ensure all operations complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as e:
            _dbg(f"[ElevationWorker] Cleanup warning: {e}")
