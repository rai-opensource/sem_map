# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import torch
from typing import Dict, Tuple


def _cuda_backend():
    """Use pycuda as the CUDA backend."""
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda  # type: ignore
        return ("pycuda", cuda)
    except Exception as e:
        raise RuntimeError("No CUDA Python backend found (install pycuda)") from e


def _trt_dtype_to_torch(trt_dtype):
    """Convert TensorRT dtype to PyTorch dtype."""
    import tensorrt as trt

    if trt_dtype == trt.float16:
        return torch.float16
    if trt_dtype == trt.float32:
        return torch.float32
    if hasattr(trt, "bfloat16") and trt_dtype == trt.bfloat16:
        return torch.bfloat16
    # Extend as needed (int8, etc.)
    raise TypeError(f"Unsupported TRT dtype: {trt_dtype}")


class TrtRunner:
    """Unified TensorRT runner for all models."""

    def __init__(self, engine_path: str, profile_index: int = 0):
        import tensorrt as trt

        self.trt = trt
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # Use default CUDA stream (0)
        self.stream_handle = 0

        # Pre-allocated output tensors (reused across inference calls)
        self.output_memory_slots: Dict[str, torch.Tensor] = {}

        # Select optimization profile (if any)
        if self.engine.num_optimization_profiles > 0:
            self.profile_index = max(
                0, min(profile_index, self.engine.num_optimization_profiles - 1)
            )
            try:
                if hasattr(self.context, "set_optimization_profile_async"):
                    self.context.set_optimization_profile_async(
                        self.profile_index, int(self.stream_handle)
                    )
                elif hasattr(self.context, "set_optimization_profile"):
                    self.context.set_optimization_profile(self.profile_index)
                elif hasattr(self.context, "active_optimization_profile"):
                    self.context.active_optimization_profile = int(self.profile_index)
            except Exception:
                pass
        else:
            self.profile_index = 0

        # Build tensor maps using TensorRT 10.x tensor API
        if not hasattr(self.engine, "num_io_tensors"):
            raise RuntimeError("TensorRT 10.x tensor I/O API required")

        self.bindings: Dict[str, int] = {}
        self.binding_is_input: Dict[str, bool] = {}
        self.binding_dtypes: Dict[str, torch.dtype] = {}
        num_io = int(self.engine.num_io_tensors)
        for i in range(num_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            is_input = mode == trt.TensorIOMode.INPUT
            self.bindings[name] = i
            self.binding_is_input[name] = is_input
            self.binding_dtypes[name] = _trt_dtype_to_torch(
                self.engine.get_tensor_dtype(name)
            )

        # Pre-allocated device pointers (reused across calls when shapes match)
        self._dev_ptrs: Dict[int, int] = {}
        self._dev_shapes: Dict[int, Tuple[int, ...]] = {}

    def set_bindings(self, input_tensors_dict: Dict[str, torch.Tensor]):
        """Bind input tensors (zero-copy)."""
        # Set input shapes and bind addresses
        for name, tensor in input_tensors_dict.items():
            if name not in self.bindings or not self.binding_is_input[name]:
                continue

            # Align dtype
            target_dtype = self.binding_dtypes[name]
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            if tensor.device.type != "cuda":
                tensor = tensor.cuda(non_blocking=True)

            # Set shape
            try:
                self.context.set_input_shape(name, tuple(tensor.shape))
            except Exception:
                try:
                    self.context.set_tensor_shape(name, tuple(tensor.shape))
                except Exception:
                    pass

            # Bind address
            try:
                self.context.set_tensor_address(name, int(tensor.data_ptr()))
            except Exception:
                if hasattr(self.context, "set_input_tensor_address"):
                    self.context.set_input_tensor_address(name, int(tensor.data_ptr()))

    def infer(self, output_names: list[str]) -> Dict[str, torch.Tensor]:
        """Execute inference and return output tensors (reused from memory slots)."""
        # Allocate and bind ALL output tensors (not just requested ones)
        output_tensors: Dict[str, torch.Tensor] = {}
        for out_name, is_input in self.binding_is_input.items():
            if is_input:
                continue  # Skip input tensors

            # Determine shape
            try:
                out_shape = tuple(self.context.get_tensor_shape(out_name))
            except Exception:
                out_shape = ()

            if not out_shape or any(int(d) < 0 for d in out_shape):
                # Fallback shape - this should be handled by the calling code
                raise RuntimeError(f"Could not determine output shape for {out_name}")

            # Reuse or allocate output tensor
            if out_name not in self.output_memory_slots:
                # First inference or never allocated - create new tensor
                out_dtype = self.binding_dtypes.get(out_name, torch.float16)
                out_tensor = torch.empty(out_shape, dtype=out_dtype, device="cuda")
                self.output_memory_slots[out_name] = out_tensor
            else:
                # Reuse cached tensor if shape matches
                cached = self.output_memory_slots[out_name]
                if cached.shape != out_shape:
                    # Dynamic shape changed - reallocate
                    out_dtype = self.binding_dtypes.get(out_name, torch.float16)
                    out_tensor = torch.empty(out_shape, dtype=out_dtype, device="cuda")
                    self.output_memory_slots[out_name] = out_tensor
                else:
                    # Shape matches - reuse existing tensor
                    out_tensor = cached

            output_tensors[out_name] = out_tensor

            # Bind output address
            try:
                self.context.set_tensor_address(out_name, int(out_tensor.data_ptr()))
            except Exception:
                if hasattr(self.context, "set_output_tensor_address"):
                    self.context.set_output_tensor_address(out_name, int(out_tensor.data_ptr()))

        # Enqueue (async)
        ok = False
        last_err = None
        for meth in ("enqueue_v3", "enqueueV3", "execute_async_v3", "execute_async_v2"):
            if hasattr(self.context, meth):
                try:
                    if meth == "execute_async_v2":
                        # Legacy path
                        ok = bool(getattr(self.context, meth)([], int(self.stream_handle)))
                    else:
                        ok = bool(getattr(self.context, meth)(int(self.stream_handle)))
                    break
                except Exception as e:
                    last_err = e
        if not ok:
            raise RuntimeError(f"TensorRT enqueue failed: {last_err}")

        # Sync to ensure completion
        torch.cuda.synchronize()
        return output_tensors

    def __del__(self):
        """Cleanup device memory."""
        try:
            for dptr in self._dev_ptrs.values():
                if dptr:
                    self.backend.mem_free(dptr)
            self._dev_ptrs.clear()
        except Exception:
            pass


def load_engine(engine_path: str):
    """Load TensorRT engine (for compatibility)."""
    return TrtRunner(engine_path)
