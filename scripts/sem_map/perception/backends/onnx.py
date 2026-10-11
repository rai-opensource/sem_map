# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import numpy as np
import torch
import onnxruntime as ort


def create_session(onnx_path: str, use_cuda: bool = True, fp16: bool = True):
    """Build ONNX session with provider selection.

    Args:
        onnx_path: Path to ONNX model file
        use_cuda: Whether to use CUDA-based execution providers
        fp16: Whether to enable FP16 precision for TensorRT

    Returns:
        ONNX Runtime InferenceSession with optimal execution provider
    """
    available = set(getattr(ort, "get_available_providers", lambda: [])())

    providers = []
    provider_options = []

    # Prioritize TensorRT ExecutionProvider for best performance on Jetson/NVIDIA GPUs
    if use_cuda and "TensorrtExecutionProvider" in available:
        import os
        cache_dir = os.path.join(os.path.dirname(onnx_path), ".trt_cache")
        os.makedirs(cache_dir, exist_ok=True)

        providers.append("TensorrtExecutionProvider")
        provider_options.append({
            "device_id": 0,
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_max_workspace_size": 2147483648,  # 2GB
        })
        print(f"Using TensorrtExecutionProvider (FP16: {fp16})")

    # Fallback to CUDA ExecutionProvider
    if use_cuda and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        provider_options.append({
            "device_id": 0,
        })
        if "TensorrtExecutionProvider" not in available:
            print("Using CUDAExecutionProvider")

    # CPU fallback
    providers.append("CPUExecutionProvider")
    provider_options.append({})

    so = ort.SessionOptions()
    try:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        pass

    try:
        sess = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=providers,
            provider_options=provider_options,
        )
        print(f"Active provider: {sess.get_providers()[0]}")
    except Exception as e:
        print(f"Warning: Failed to create session with specified providers: {e}")
        print("Falling back to default providers...")
        sess = ort.InferenceSession(onnx_path, sess_options=so)
    return sess


def prepare_numpy_input(tensor: torch.Tensor, target_dtype: np.dtype) -> np.ndarray:
    """Convert torch tensor to numpy with correct dtype."""
    # Convert to target dtype
    if target_dtype == np.float16 and tensor.dtype != torch.float16:
        tensor = tensor.half()
    elif target_dtype == np.float32 and tensor.dtype != torch.float32:
        tensor = tensor.float()
    # Preserve original shape (expecting 4D [B,C,H,W] for seg)
    np_arr = tensor.detach().cpu().numpy()
    if np_arr.dtype != target_dtype:
        np_arr = np_arr.astype(target_dtype, copy=False)
    return np_arr


def run_session(session, inputs_dict: dict, output_names: list[str]):
    """Generic ONNX inference."""
    return session.run(output_names, inputs_dict)
