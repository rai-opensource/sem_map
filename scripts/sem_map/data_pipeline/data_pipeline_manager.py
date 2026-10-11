# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
DataPipelineManager: Orchestrates multiprocessing architecture for sem_map.

Manages:
- CUDA IPC buffers/queues for zero-copy data transfer
- Worker process lifecycle (spawn, health monitoring, shutdown)
- Handle sharing for CUDA IPC

Architecture:
  Process 1 (C++/Python main): RGB/PC input → buffers
  Process 2 (Semantic worker): RGB buffer → Semantic → Trav buffer
  Process 3 (Elevation worker): PC buffer + Trav buffer → Elevation → Elev queue
"""
import sys
import threading
import torch
import torch.multiprocessing as mp
# cuda-python 13.x API
from cuda.bindings import driver as cuda
from typing import Optional, Tuple

from sem_map.data_pipeline.cuda_buffer import CUDATimeBuffer
from sem_map.data_pipeline.cuda_utils import checkCudaErrors


_thread_local = threading.local()

def _ensure_cuda_context(device_id: int = 0):
    """Ensure CUDA primary context is set on the calling thread.

    CUDA driver API contexts are per-thread. When C++ threads call into Python
    via pybind11, they need the primary context set as current before any driver
    API operations (cuMemcpyDtoDAsync, etc.). torch.cuda.set_device() internally
    calls cudaSetDevice() which retains and sets the primary context.
    """
    if not getattr(_thread_local, 'cuda_ctx_set', False):
        torch.cuda.set_device(device_id)
        _thread_local.cuda_ctx_set = True

def _dbg(msg: str):
    """Debug print to stderr (visible in ROS logs)."""
    try:
        sys.stderr.write(f"[DataPipelineManager] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


class DataPipelineManager:
    """
    Manages the multiprocessing data pipeline for sem_map.

    Thread Safety: Relies on VPE's internal mp.Lock per slot in buffers/queues.
    Multiple C++ threads can safely call write_* methods concurrently.
    """

    def __init__(self, config: dict, param=None):
        """
        Initialize buffers/queues (does NOT spawn processes yet).

        Args:
            config: Configuration dict containing:
                - image_height, image_width: RGB image dimensions
                - pc_max_points: Maximum pointcloud size
                - map_cells: Elevation map dimensions
                - enable_semantic_pipeline: bool
                - enable_height_map: bool
            param: Parameter instance (for worker processes), optional
        """
        self.config = config
        self.param = param  # Store for worker processes

        # CUDA IPC workers must be spawned, never forked: forking a CUDA-initialized,
        # multithreaded parent (the ROS component container) causes intermittent
        # hangs/crashes in the children. sem_map/__init__.py sets 'spawn' only if the
        # context is still unset, so fail loudly if something fixed it to 'fork' first.
        start_method = mp.get_start_method(allow_none=True)
        if start_method is None:
            mp.set_start_method('spawn')
        elif start_method != 'spawn':
            raise RuntimeError(
                f"multiprocessing start method is {start_method!r}, but CUDA IPC workers "
                f"require 'spawn'. Something fixed the multiprocessing context before "
                f"sem_map was imported."
            )
        _dbg(f"multiprocessing start method: {mp.get_start_method()}")

        self.kill_event = mp.Event()

        # Worker readiness events (to prevent writing before workers are ready)
        self.semantic_ready = mp.Event()
        self.elevation_ready = mp.Event()

        # Command events to the elevation worker
        self.elevation_reset_event = mp.Event()

        # Extract dimensions from config
        self.img_h = config.get('image_height', 480)
        self.img_w = config.get('image_width', 640)
        self.pc_max_pts = config.get('pc_max_points', 100000)
        self.map_cells = config.get('map_cells', 100)
        self.semantic_bev_layer_name = (
            config.get('semantic_bev_layer_name')
            or (param.get_value('semantic_bev_layer_name', 'trail') if param is not None else 'trail')
        )

        # Traversability output dimensions (segmentation model outputs at ~2/3 scale)
        # Default: 320x512 for 480x640 input (can be overridden in config)
        self.trav_h = config.get('trav_height', 320)
        self.trav_w = config.get('trav_width', 512)

        # Buffer/queue sizes
        self.buffer_size = config.get('buffer_size', 10)

        # Initialize CUDA in main process
        checkCudaErrors(cuda.cuInit(0))

        # Ensure CUDA context exists (required for cuMemsetD8 and other CUDA operations)
        # PyTorch CUDA initialization creates a context if one doesn't exist
        if torch.cuda.is_available():
            # Create a dummy tensor to ensure CUDA context is initialized
            _ = torch.zeros(1, device='cuda')

        # Create buffers/queues
        self._create_buffers()

        # Worker process handles (created in start())
        self.semantic_proc: Optional[mp.Process] = None
        self.elevation_proc: Optional[mp.Process] = None
        self.started = False

        # Timestamp tracking for reading (to avoid reprocessing old data)
        # Initialize to 0.0 to skip any stale data that might be in buffer at startup
        self.last_elevation_timestamp = 0.0
        self.last_semantic_timestamp = 0.0

    def start(self) -> None:
        """
        Spawn worker processes and share CUDA IPC handles.

        Call this after __init__() to start the multiprocessing pipeline.
        """
        if self.started:
            return  # Already started

        # Spawn worker processes
        if self.config.get('enable_semantic_pipeline', True):
            self._spawn_semantic_process()

        if self.config.get('enable_height_map', True):
            self._spawn_elevation_process()

        # Send CUDA IPC handles to workers
        self._share_handles()

        # Wait for workers to be ready before accepting data.
        # A worker that crashed during init (e.g. failed CUDA IPC handle import)
        # must not be reported as ready - fail fast with a clear error instead
        # of coming up as a silently dead pipeline.
        _dbg("Waiting for workers to initialize...")
        if self.config.get('enable_semantic_pipeline', True):
            if self.semantic_ready.wait(timeout=60):
                _dbg("Semantic worker ready")
            else:
                _dbg("WARNING: Semantic worker not ready after 60s "
                     "(model still loading, or worker failed - see logs above)")
        if self.config.get('enable_height_map', True):
            if self.elevation_ready.wait(timeout=60):
                _dbg("Elevation worker ready")
            else:
                _dbg("WARNING: Elevation worker not ready after 60s "
                     "(still initializing, or worker failed - see logs above)")

        healthy, msg = self.check_health()
        if not healthy:
            self.shutdown()
            raise RuntimeError(f"Worker process failed during startup: {msg}")

        self.started = True
        _dbg("Pipeline started")

    def _create_buffers(self):
        """Create CUDA IPC buffers needed for the active pipeline configuration."""
        enable_height_map = self.config.get('enable_height_map', True)

        # Ensure CUDA runtime context is initialized before driver API allocations.
        # The CUDA driver API (used by CUDATimeBuffer) needs a valid context.
        # Without this, memory allocated here may be in a different context than
        # the one used by reader threads (via torch.cuda), causing CUDA_ERROR_INVALID_CONTEXT.
        torch.cuda.init()

        # RGB input buffer: Process 1 (C++) → Process 2 (Semantic)
        # NOTE: Using float32 for all fields to avoid dtype corruption.
        # Image will be converted from uint8→float32 before storage.
        self.rgb_input_buffer = CUDATimeBuffer(
            max_size=self.buffer_size,
            dtype=torch.float32,
            input_device='cuda',  # C++ writes GPU tensors directly
            output_device='cuda',
            data_signature={
                'image': (self.img_h, self.img_w, 3),
                'pose_R': (3, 3),
                'pose_t': (3,),
                'intrinsics': (3, 3),
            },
            add_batch_dim=False
        )

        # Semantic results buffer: Process 2 (Semantic) → Process 3 (Elevation) or C++ publisher
        # When height map disabled, C++ reads directly from this buffer
        self.sem_results_buffer = CUDATimeBuffer(
            max_size=self.buffer_size,
            dtype=torch.float32,
            input_device='cuda',
            output_device='cuda',
            data_signature={
                'traversability': (self.trav_h, self.trav_w),  # Model latent resolution
                'segmentation': (self.trav_h, self.trav_w),     # Label IDs as float32
                'confidence': (self.trav_h, self.trav_w),       # Confidence map (cosine similarity)
                'pose_R': (3, 3),
                'pose_t': (3,),
                'intrinsics': (3, 3),
            },
            add_batch_dim=False
        )

        # Pointcloud and elevation buffers only needed when height mapping is enabled
        self.pc_input_buffer = None
        self.elev_output_buffer = None

        if enable_height_map:
            # Pointcloud input buffer: Process 1 (C++) → Process 3 (Elevation)
            self.pc_input_buffer = CUDATimeBuffer(
                max_size=self.buffer_size,
                dtype=torch.float32,
                input_device='cuda',
                output_device='cuda',
                data_signature={
                    'pointcloud': (self.pc_max_pts, 3),  # Pre-allocated max size
                    'num_points': (),  # Scalar: actual count
                    'pose_R': (3, 3),
                    'pose_t': (3,),
                },
                add_batch_dim=False
            )

            # Elevation output buffer: Process 3 (Elevation) → Process 1 (C++ publisher)
            self.elev_output_buffer = CUDATimeBuffer(
                max_size=self.buffer_size,  # Use buffer_size (same as other buffers)
                dtype=torch.float32,
                input_device='cuda',
                output_device='cpu',  # C++ reads on CPU
                data_signature={
                    'elevation': (self.map_cells, self.map_cells),
                    'variance': (self.map_cells, self.map_cells),
                    self.semantic_bev_layer_name: (self.map_cells, self.map_cells),  # Fused traversability layer
                    'map_center': (3,),  # Map center position [x, y, z] in world frame
                },
                add_batch_dim=False
            )

    def _spawn_semantic_process(self):
        """Spawn semantic processing worker."""
        from sem_map.workers.semantic_worker import semantic_worker_main

        # Use param if available, otherwise convert config to param
        worker_param = self.param
        if worker_param is None:
            # Create minimal param from config (for testing)
            from sem_map.parameter import Parameter
            worker_param = Parameter()
            # Set any config overrides if needed

        self.semantic_proc = mp.Process(
            target=semantic_worker_main,
            args=(
                self.rgb_input_buffer,
                self.sem_results_buffer,
                worker_param,  # Pass Parameter instance, not dict
                self.kill_event,
                self.semantic_ready  # Signal when ready
            ),
            name="SemanticWorker"
        )
        self.semantic_proc.start()

    def _spawn_elevation_process(self):
        """Spawn elevation mapping worker."""
        from sem_map.workers.elevation_worker import elevation_worker_main

        # Use param if available, otherwise convert config to param
        worker_param = self.param
        if worker_param is None:
            # Create minimal param from config (for testing)
            from sem_map.parameter import Parameter
            worker_param = Parameter()
            # Set any config overrides if needed

        self.elevation_proc = mp.Process(
            target=elevation_worker_main,
            args=(
                self.pc_input_buffer,
                self.sem_results_buffer,
                self.elev_output_buffer,  # Changed from queue to buffer
                worker_param,
                self.kill_event,
                self.elevation_ready,  # Signal when ready
                self.elevation_reset_event,  # Reset command from main process
            ),
            name="ElevationWorker"
        )
        self.elevation_proc.start()

    def _share_handles(self):
        """Send CUDA IPC handles to all accessor processes."""
        pids = []

        if self.semantic_proc:
            pids.append(self.semantic_proc.pid)
        if self.elevation_proc:
            pids.append(self.elevation_proc.pid)

        if not pids:
            return  # No workers spawned

        # rgb_input_buffer: accessed by semantic (reads)
        if self.semantic_proc:
            self.rgb_input_buffer.send_shareable_handles([self.semantic_proc.pid])

        # pc_input_buffer: accessed by elevation (reads)
        if self.elevation_proc:
            self.pc_input_buffer.send_shareable_handles([self.elevation_proc.pid])

        # sem_results_buffer: accessed by semantic (writes) AND elevation (reads)
        sem_accessors = []
        if self.semantic_proc:
            sem_accessors.append(self.semantic_proc.pid)
        if self.elevation_proc:
            sem_accessors.append(self.elevation_proc.pid)
        if sem_accessors:
            self.sem_results_buffer.send_shareable_handles(sem_accessors)

        # elev_output_buffer: accessed by elevation (writes)
        # Main process (creator) already has access for reading (CUDATimeBuffer design)
        if self.elevation_proc:
            self.elev_output_buffer.send_shareable_handles([self.elevation_proc.pid])

    # ====== C++ Interface Methods ======

    def write_rgb(self, image, sensor_name: str, timestamp_ns: int,
                  pose_R, pose_t, intrinsics) -> None:
        """
        Write RGB frame to input buffer (called from C++ RGB thread).

        Args:
            image: (H, W, 3) numpy array or torch tensor, uint8
            sensor_name: Camera sensor identifier (e.g., "camera_front")
            timestamp_ns: Nanosecond timestamp for synchronization
            pose_R: (3, 3) rotation matrix (numpy or torch)
            pose_t: (3,) translation vector (numpy or torch)
            intrinsics: (3, 3) camera matrix K (numpy or torch)
        """
        # Convert numpy arrays to torch tensors if needed
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).cuda()
        if not isinstance(pose_R, torch.Tensor):
            pose_R = torch.from_numpy(pose_R).cuda().float()
        if not isinstance(pose_t, torch.Tensor):
            pose_t = torch.from_numpy(pose_t).cuda().float()
        if not isinstance(intrinsics, torch.Tensor):
            intrinsics = torch.from_numpy(intrinsics).cuda().float()

        # Fix Eigen column-major → NumPy/PyTorch row-major layout
        # pybind11 preserves memory layout, so matrices need transpose
        pose_R = pose_R.T
        intrinsics = intrinsics.T

        # Validate image dimensions (will fail with clear error if mismatch)
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be (H, W, 3), got {image.shape}")
        if image.shape[0] != self.img_h or image.shape[1] != self.img_w:
            _dbg(f"WARNING: Image shape mismatch. Expected ({self.img_h}, {self.img_w}, 3), got {image.shape}. "
                 f"This will cause buffer write to fail.")

        # Convert image to float32 to match buffer dtype
        # (Buffer uses float32 for all fields to avoid dtype corruption)
        # Keep [0-255] range - semantic_pipeline will normalize to [0-1]
        if image.dtype == torch.uint8:
            image = image.float()

        # Convert timestamp from nanoseconds to float seconds
        timestamp = float(timestamp_ns) / 1e9

        data = {
            'image': image,
            'pose_R': pose_R,
            'pose_t': pose_t,
            'intrinsics': intrinsics,
        }
        self.rgb_input_buffer.put(data, timestamp, sync=True)

    def write_pointcloud(self, points, channels: list, timestamp_ns: int,
                         pose_R, pose_t) -> None:
        """
        Write pointcloud to input buffer (called from C++ PC thread).

        Args:
            points: (N, C) numpy array or torch tensor, float32
            channels: List of channel names (e.g., ['x', 'y', 'z', 'intensity'])
            timestamp_ns: Nanosecond timestamp for synchronization
            pose_R: (3, 3) rotation matrix (numpy or torch)
            pose_t: (3,) translation vector (numpy or torch)
        """
        if self.pc_input_buffer is None:
            return

        # Convert numpy arrays to torch tensors if needed
        if not isinstance(points, torch.Tensor):
            points = torch.from_numpy(points).cuda().float()
        if not isinstance(pose_R, torch.Tensor):
            pose_R = torch.from_numpy(pose_R).cuda().float()
        if not isinstance(pose_t, torch.Tensor):
            pose_t = torch.from_numpy(pose_t).cuda().float()

        # Fix Eigen column-major → NumPy/PyTorch row-major layout
        # pybind11 preserves memory layout, so matrices need transpose
        pose_R = pose_R.T

        # Convert timestamp from nanoseconds to float seconds
        timestamp = float(timestamp_ns) / 1e9

        # Extract XYZ (first 3 channels)
        points_xyz = points[:, :3]
        num_points = points_xyz.shape[0]

        # Validate pointcloud size
        if num_points > self.pc_max_pts:
            raise ValueError(
                f"Pointcloud size ({num_points}) exceeds maximum buffer size ({self.pc_max_pts}). "
                f"Increase 'pc_max_points' in config or reduce pointcloud size."
            )

        # Pad to max size if smaller
        if num_points < self.pc_max_pts:
            padded = torch.zeros((self.pc_max_pts, 3),
                                dtype=torch.float32, device='cuda')
            padded[:num_points] = points_xyz
            points_xyz = padded

        data = {
            'pointcloud': points_xyz,
            'num_points': torch.tensor(num_points, dtype=torch.float32, device='cuda'),
            'pose_R': pose_R,
            'pose_t': pose_t,
        }

        self.pc_input_buffer.put(data, timestamp, sync=True)

    def read_bev_map(self) -> Optional[Tuple[dict, Tuple[float, float, float], int]]:
        """
        Read BEV map layers and map center from output buffer (called from C++ publish thread).

        Returns:
            (layers_dict, map_center, timestamp_ns) where:
            - layers_dict contains numpy arrays of elevation layers
            - map_center is (x, y, z) tuple in world frame
            - timestamp_ns is timestamp in nanoseconds
            Returns None if buffer is empty or no new data since last read
        """
        try:
            if self.elev_output_buffer is None:
                return None
            _ensure_cuda_context()
            result = self.elev_output_buffer.get(last_timestamp=self.last_elevation_timestamp, sync=True)

            if result is None:
                return None

            # item_id carries the sensor timestamp (seconds); timestamp is the
            # slot write time (perf_counter), used only for freshness gating.
            sensor_timestamp, timestamp, data = result

            # Update last read timestamp (slot write time domain)
            self.last_elevation_timestamp = timestamp

            # Extract map center (required - should always be present)
            if 'map_center' not in data:
                _dbg("WARNING: map_center missing from elevation buffer data")
                return None

            map_center_tensor = data['map_center']
            if isinstance(map_center_tensor, torch.Tensor):
                map_center_np = map_center_tensor.cpu().numpy()
            else:
                import numpy as np
                map_center_np = np.asarray(map_center_tensor)
            map_center = tuple(map_center_np.tolist())

            # Convert torch tensors to numpy for C++
            layers_dict = {}
            for key, tensor in data.items():
                if key == 'map_center':
                    continue  # Skip map_center from layers_dict (handled separately)
                if isinstance(tensor, torch.Tensor):
                    layers_dict[key] = tensor.cpu().numpy()
                else:
                    layers_dict[key] = tensor

            # Convert sensor timestamp to nanoseconds for C++
            timestamp_ns = int(sensor_timestamp * 1e9)

            return (layers_dict, map_center, timestamp_ns)

        except Exception as e:
            _dbg(f"read_bev_map error: {type(e).__name__}: {e}")
            return None

    def read_semantic_results(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int, float]]:
        """
        Read latest semantic results (traversability + segmentation + confidence) from output buffer.

        Returns:
            (traversability_tensor_cpu, segmentation_tensor_cpu, confidence_tensor_cpu, height, width, timestamp) tuple,
            or None if buffer is empty or no new data since last read.
            confidence_tensor_cpu will be None if confidence is not available (old models).

        Note: Tensors are moved to CPU before returning to minimize GIL hold time in C++.
        """
        try:
            _ensure_cuda_context()
            result = self.sem_results_buffer.get(
                last_timestamp=self.last_semantic_timestamp,
                sync=True
            )
            if result is None:
                return None

            # item_id carries the sensor timestamp (seconds); timestamp is the
            # slot write time (perf_counter), used only for freshness gating.
            sensor_timestamp, timestamp, sem_dict = result
            self.last_semantic_timestamp = timestamp

            trav_tensor = sem_dict['traversability']
            seg_tensor = sem_dict['segmentation']

            trav_cpu = trav_tensor.cpu()
            seg_cpu = seg_tensor.cpu()
            h, w = trav_cpu.shape

            # Get confidence if available
            conf_cpu = None
            if 'confidence' in sem_dict:
                conf_tensor = sem_dict['confidence']
                conf_cpu = conf_tensor.cpu()

            return (trav_cpu, seg_cpu, conf_cpu, int(h), int(w), sensor_timestamp)
        except Exception as e:
            _dbg(f"read_semantic_results error: {type(e).__name__}: {e}")
            return None

    def set_camera_info(self, K, D, distortion_model: str, width: int, height: int) -> None:
        """
        Set camera intrinsics (stub for compatibility with legacy interface).

        In multiprocessing mode, camera info is passed per-frame in buffers,
        so this method does nothing.
        """
        pass


    def reset_elevation_map(self) -> None:
        """Signal the elevation worker to clear all map layers on its next tick."""
        self.elevation_reset_event.set()

    def check_health(self) -> Tuple[bool, str]:
        """
        Check if all worker processes are alive.

        Returns:
            (healthy, message) tuple
        """
        if self.semantic_proc and not self.semantic_proc.is_alive():
            return False, f"Semantic process died (exit code: {self.semantic_proc.exitcode})"
        if self.elevation_proc and not self.elevation_proc.is_alive():
            return False, f"Elevation process died (exit code: {self.elevation_proc.exitcode})"
        return True, "OK"

    def shutdown(self, timeout: float = 2.0) -> None:
        """
        Gracefully shutdown all worker processes.

        Args:
            timeout: Maximum seconds to wait for each process to join
        """
        # Signal workers to stop
        self.kill_event.set()

        # Join semantic process
        if self.semantic_proc:
            self.semantic_proc.join(timeout=timeout)
            if self.semantic_proc.is_alive():
                _dbg("WARNING: Semantic process didn't stop, terminating...")
                self.semantic_proc.terminate()
                self.semantic_proc.join(timeout=0.5)

        # Join elevation process
        if self.elevation_proc:
            self.elevation_proc.join(timeout=timeout)
            if self.elevation_proc.is_alive():
                _dbg("WARNING: Elevation process didn't stop, terminating...")
                self.elevation_proc.terminate()
                self.elevation_proc.join(timeout=0.5)

        # Close buffers
        try:
            self.rgb_input_buffer.close()
        except Exception as e:
            _dbg(f"Error closing rgb_input_buffer: {e}")
        if self.pc_input_buffer is not None:
            try:
                self.pc_input_buffer.close()
            except Exception as e:
                _dbg(f"Error closing pc_input_buffer: {e}")
        try:
            self.sem_results_buffer.close()
        except Exception as e:
            _dbg(f"Error closing sem_results_buffer: {e}")
        if self.elev_output_buffer is not None:
            try:
                self.elev_output_buffer.close()
            except Exception as e:
                _dbg(f"Error closing elev_output_buffer: {e}")
