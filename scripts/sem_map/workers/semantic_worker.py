# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Semantic processing worker process.
Reads RGB frames, runs TensorRT inference, outputs traversability.
"""
import time
from time import sleep
import sys
from collections import deque
import torch
# cuda-python 13.x API
from cuda.bindings import driver as cuda

from sem_map.data_pipeline.cuda_utils import checkCudaErrors


def _dbg(msg: str):
    """Debug print to stderr (visible in ROS logs)."""
    try:
        sys.stderr.write(f"[SemanticWorker] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def semantic_worker_main(rgb_buffer, sem_results_buffer, param, kill_event, ready_event=None):
    """
    Main loop for semantic processing worker.

    Args:
        rgb_buffer: CUDATimeBuffer for RGB input
        sem_results_buffer: CUDATimeBuffer for traversability + segmentation output
        param: Parameter instance (from sem_map.parameter)
        kill_event: mp.Event for shutdown signal
        ready_event: mp.Event to signal when worker is ready (optional)
    """
    import signal

    # Handle SIGINT/SIGTERM gracefully
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
        rgb_buffer.recv_shareable_handles()
        sem_results_buffer.recv_shareable_handles()

        # Initialize SemanticPipeline
        from sem_map.semantic_pipeline import SemanticPipeline

        try:
            pipeline = SemanticPipeline(param)
        except Exception as e:
            _dbg(f"ERROR: Failed to load SemanticPipeline: {e}")
            import traceback
            traceback.print_exc()
            return

        # Signal that worker is ready
        if ready_event is not None:
            ready_event.set()

        # Get flip_image parameter (default to False)
        flip_image = param.get_value('flip_image', False)
        if flip_image:
            _dbg("flip_image enabled: RGB images and outputs will be flipped vertically")

        # Metrics - sliding window statistics (last 100 frames)
        frame_count = 0
        loop_times = deque(maxlen=100)  # Stores last 100 loop times
        process_times = deque(maxlen=100)  # Stores last 100 process times
        frame_timestamps = deque(maxlen=100)  # Wall-clock timestamps for windowed FPS

        # Rate control: get processing rate from param (default to 30 Hz if not set)
        processing_rate_hz = getattr(param, 'rgb_processing_rate_hz', 30.0)
        period = 1.0 / processing_rate_hz if processing_rate_hz > 0 else 1.0 / 30.0
        last_timestamp = float('-inf')

        while not kill_event.is_set():
            loop_start = time.perf_counter()

            # Try to get latest RGB frame
            result = rgb_buffer.get(last_timestamp)
            if result is None:
                # No new data, sleep to maintain rate
                loop_elapsed = time.perf_counter() - loop_start
                if loop_elapsed < period:
                    sleep(period - loop_elapsed)
                continue

            # item_id carries the sensor timestamp (seconds); timestamp is the
            # slot's write time (time.perf_counter), used only for freshness gating.
            sensor_timestamp, timestamp, rgb_dict = result
            last_timestamp = timestamp

            try:
                # Extract image data
                # rgb_dict contains: 'image', 'pose_R', 'pose_t', 'intrinsics'
                # Image is float32 [0-255] on CUDA (converted from uint8 to match buffer dtype)
                image_tensor = rgb_dict['image']  # (H, W, 3) float32 [0-255] on CUDA
                pose_R = rgb_dict['pose_R']       # (3, 3) on CUDA
                pose_t = rgb_dict['pose_t']       # (3,) on CUDA
                intrinsics = rgb_dict['intrinsics']  # (3, 3) on CUDA

                # Get image dimensions
                H, W, C = image_tensor.shape

                # Process through semantic pipeline (pass tensor directly for zero-copy)
                # Note: Image flipping is handled inside the pipeline for consistency
                process_start = time.perf_counter()
                pipeline.process_frame(
                    image_tensor,  # Pass GPU tensor directly, no CPU copy!
                    height=H,
                    width=W,
                    channels=C,
                    sensor_name="rgb_camera",
                    stamp_ns=int(sensor_timestamp * 1e9)  # Sensor timestamp back to nanoseconds
                )
                process_time = (time.perf_counter() - process_start) * 1000

                # Get semantic outputs (traversability + segmentation)
                if pipeline.last_traversability is None:
                    _dbg("WARNING: No traversability output from pipeline")
                    continue
                if pipeline.last_segmentation is None:
                    _dbg("WARNING: No segmentation output from pipeline")
                    continue

                # Ensure traversability is contiguous, float32, and on CUDA
                trav_tensor = pipeline.last_traversability
                if trav_tensor.dtype != torch.float32:
                    trav_tensor = trav_tensor.float()
                if not trav_tensor.is_contiguous():
                    trav_tensor = trav_tensor.contiguous()

                seg_tensor = pipeline.last_segmentation
                if seg_tensor.dtype != torch.float32:
                    seg_tensor = seg_tensor.float()
                if not seg_tensor.is_contiguous():
                    seg_tensor = seg_tensor.contiguous()

                # Note: Image and output flipping is handled inside the pipeline
                # No need to flip here as pipeline.last_traversability and pipeline.last_segmentation
                # are already flipped if flip_image is enabled

                # Ensure pose/intrinsics are contiguous and float32
                pose_R_contig = pose_R.float().contiguous() if (pose_R.dtype != torch.float32 or not pose_R.is_contiguous()) else pose_R
                pose_t_contig = pose_t.float().contiguous() if (pose_t.dtype != torch.float32 or not pose_t.is_contiguous()) else pose_t
                intrinsics_contig = intrinsics.float().contiguous() if (intrinsics.dtype != torch.float32 or not intrinsics.is_contiguous()) else intrinsics

                # Prepare confidence tensor (required by buffer schema)
                if pipeline.last_confidence is not None:
                    conf_tensor = pipeline.last_confidence
                    if conf_tensor.dtype != torch.float32:
                        conf_tensor = conf_tensor.float()
                    if not conf_tensor.is_contiguous():
                        conf_tensor = conf_tensor.contiguous()
                else:
                    # Fallback for old models without confidence output: report
                    # FULL confidence so downstream gates/weighting are no-ops and
                    # fusion behaves as it did before confidence existed. (Zeros
                    # would silently block all BEV fusion under any conf gate.)
                    conf_tensor = torch.ones_like(trav_tensor)

                # Prepare output dict for semantic results buffer
                # All keys MUST match the buffer's data_signature
                sem_result = {
                    'traversability': trav_tensor,  # (H, W) float32 on CUDA
                    'segmentation': seg_tensor,     # (H, W) float32 label IDs on CUDA
                    'confidence': conf_tensor,      # (H, W) float32 on CUDA (zeros if unavailable)
                    'pose_R': pose_R_contig,
                    'pose_t': pose_t_contig,
                    'intrinsics': intrinsics_contig,
                }

                # Write to output buffer, carrying the input's sensor timestamp
                # as the item id so it survives to the published message headers
                sem_results_buffer.put(sem_result, sensor_timestamp, sync=True)

                loop_time = (time.perf_counter() - loop_start) * 1000

                # Update metrics - append to sliding windows
                frame_count += 1
                loop_times.append(loop_time)
                process_times.append(process_time)
                frame_timestamps.append(time.perf_counter())

                # Log every 100 frames
                if frame_count % 100 == 0:
                    # Calculate averages from sliding window (uses available measurements)
                    avg_total = sum(loop_times) / len(loop_times) if loop_times else 0.0
                    avg_process = sum(process_times) / len(process_times) if process_times else 0.0
                    if len(frame_timestamps) >= 2:
                        fps = (len(frame_timestamps) - 1) / (frame_timestamps[-1] - frame_timestamps[0])
                    else:
                        fps = 0.0
                    window_size = len(loop_times)
                    _dbg(f"Processed {frame_count} frames (last {window_size} frames: "
                         f"FPS: {fps:.1f}, "
                         f"avg_total: {avg_total:.2f}ms, "
                         f"avg_process: {avg_process:.2f}ms)")

                # Sleep to maintain processing rate
                loop_elapsed = time.perf_counter() - loop_start
                if loop_elapsed < period:
                    sleep(period - loop_elapsed)

            except Exception as e:
                _dbg(f"ERROR processing frame: {e}")
                import traceback
                traceback.print_exc()
                # Continue processing (don't crash on single frame error)
                # Sleep to maintain rate even on error
                loop_elapsed = time.perf_counter() - loop_start
                if loop_elapsed < period:
                    sleep(period - loop_elapsed)

    except Exception as e:
        _dbg(f"FATAL ERROR in semantic_worker_main: {e}")
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
            _dbg(f"[SemanticWorker] Cleanup warning: {e}")
