# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort


def _fmt_shape(t):
    """Format tensor shape for display."""
    dims = []
    for d in t.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        elif d.dim_value:
            dims.append(str(d.dim_value))
        else:
            dims.append("?")
    return "[" + ", ".join(dims) + "]"


def _resolve_shape(t, dynamic_values: Dict[str, int]) -> List[int]:
    """Resolve tensor shape, replacing dynamic dimensions with provided values."""
    dims = []
    for d in t.dim:
        if d.dim_param:
            if d.dim_param not in dynamic_values:
                raise ValueError(f"Dynamic dimension '{d.dim_param}' not provided in --dynamic-values")
            dims.append(dynamic_values[d.dim_param])
        elif d.dim_value:
            dims.append(d.dim_value)
        else:
            raise ValueError("Encountered unknown dimension (neither dim_param nor dim_value)")
    return dims


def _onnx_dtype_to_numpy(onnx_dtype: int) -> np.dtype:
    """Convert ONNX data type to numpy dtype."""
    dtype_map = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.INT8: np.int8,
        onnx.TensorProto.UINT16: np.uint16,
        onnx.TensorProto.INT16: np.int16,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.BOOL: np.bool_,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.UINT32: np.uint32,
        onnx.TensorProto.UINT64: np.uint64,
    }
    return dtype_map.get(onnx_dtype, np.float32)


def _generate_random_inputs(model: onnx.ModelProto, dynamic_values: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Generate random input tensors matching model input specifications."""
    inputs = {}
    for vi in model.graph.input:
        name = vi.name
        t = vi.type.tensor_type
        shape = _resolve_shape(t.shape, dynamic_values)
        dtype = _onnx_dtype_to_numpy(t.elem_type)

        # Generate random data based on dtype
        if dtype in [np.float32, np.float16, np.float64]:
            data = np.random.randn(*shape).astype(dtype)
        elif dtype == np.bool_:
            data = np.random.randint(0, 2, size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            data = np.random.randint(0, 100, size=shape).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)

        inputs[name] = data
    return inputs


def _get_available_providers() -> List[str]:
    """Get all available ONNX Runtime execution providers in the system."""
    return ort.get_available_providers()


def _benchmark_session(
    session: ort.InferenceSession,
    inputs: Dict[str, np.ndarray],
    warmup_runs: int,
    benchmark_runs: int,
) -> Tuple[float, float, float]:
    """
    Benchmark an ONNX Runtime session.

    Returns:
        Tuple of (mean_time_ms, std_time_ms, fps)
    """
    # Warmup runs
    for _ in range(warmup_runs):
        session.run(None, inputs)

    # Benchmark runs
    times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        session.run(None, inputs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    mean_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000.0 / mean_time if mean_time > 0 else 0.0

    return mean_time, std_time, fps


def _print_model_info(model: onnx.ModelProto, dynamic_values: Dict[str, int]):
    """Print model input/output information."""
    print("\n" + "="*80)
    print("Model Information")
    print("="*80)

    print("\nInputs:")
    for vi in model.graph.input:
        name = vi.name
        t = vi.type.tensor_type
        elem = onnx.TensorProto.DataType.Name(t.elem_type)
        shape_str = _fmt_shape(t.shape)
        resolved_shape = _resolve_shape(t.shape, dynamic_values)
        print(f"  - {name}: {elem} {shape_str} -> resolved: {resolved_shape}")

    print("\nOutputs:")
    for vi in model.graph.output:
        name = vi.name
        t = vi.type.tensor_type
        elem = onnx.TensorProto.DataType.Name(t.elem_type)
        shape = _fmt_shape(t.shape)
        print(f"  - {name}: {elem} {shape}")


def _print_benchmark_results(provider: str, mean_time: float, std_time: float, fps: float, success: bool, error_msg: str = ""):
    """Print benchmark results for a provider."""
    if success:
        print(f"  ✓ {provider:20s} | Mean: {mean_time:7.2f} ms | Std: {std_time:6.2f} ms | FPS: {fps:6.2f}")
    else:
        print(f"  ✗ {provider:20s} | FAILED: {error_msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX model inference with random inputs across all available backends"
    )
    parser.add_argument("--model", required=True, help="Path to .onnx model")
    parser.add_argument(
        "--dynamic-values",
        type=str,
        default="",
        help="Comma-separated dynamic dimension values (e.g., 'batch_size=1,height=512,width=512')",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup runs before benchmarking (default: 5)",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=100,
        help="Number of benchmark runs to average (default: 100)",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="",
        help="Comma-separated list of providers to test (default: all available). Example: 'CPUExecutionProvider,CUDAExecutionProvider'",
    )
    args = parser.parse_args()

    # Parse dynamic values
    dynamic_values = {}
    if args.dynamic_values:
        for pair in args.dynamic_values.split(","):
            key, value = pair.split("=")
            dynamic_values[key.strip()] = int(value.strip())

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\nLoading model: {model_path}")
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    # Print model info
    _print_model_info(model, dynamic_values)

    # Generate random inputs
    print("\nGenerating random inputs...")
    inputs = _generate_random_inputs(model, dynamic_values)
    for name, data in inputs.items():
        print(f"  - {name}: shape={data.shape}, dtype={data.dtype}")

    # Get providers to test
    available_providers = _get_available_providers()
    if args.providers:
        requested_providers = [p.strip() for p in args.providers.split(",")]
        providers_to_test = [p for p in requested_providers if p in available_providers]
        unavailable = [p for p in requested_providers if p not in available_providers]
        if unavailable:
            print(f"\nWarning: Requested providers not available: {unavailable}")
    else:
        providers_to_test = available_providers

    print(f"\nAvailable providers: {available_providers}")
    print(f"Testing providers: {providers_to_test}")

    # Benchmark each provider
    print("\n" + "="*80)
    print(f"Benchmark Results (warmup={args.warmup_runs}, runs={args.benchmark_runs})")
    print("="*80)

    results = []
    for provider in providers_to_test:
        try:
            # Create session with specific provider
            session = ort.InferenceSession(
                str(model_path),
                providers=[provider]
            )

            # Benchmark
            mean_time, std_time, fps = _benchmark_session(
                session, inputs, args.warmup_runs, args.benchmark_runs
            )

            _print_benchmark_results(provider, mean_time, std_time, fps, True)
            results.append((provider, mean_time, std_time, fps, True, ""))

        except Exception as e:
            error_msg = str(e)[:60]  # Truncate long error messages
            _print_benchmark_results(provider, 0, 0, 0, False, error_msg)
            results.append((provider, 0, 0, 0, False, error_msg))

    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    successful_results = [(p, mt, st, f) for p, mt, st, f, s, _ in results if s]
    if successful_results:
        # Sort by FPS (descending)
        successful_results.sort(key=lambda x: x[3], reverse=True)
        print("\nRanked by FPS (fastest first):")
        for i, (provider, mean_time, std_time, fps) in enumerate(successful_results, 1):
            print(f"  {i}. {provider:20s} | {fps:6.2f} FPS | {mean_time:7.2f} ± {std_time:5.2f} ms")

        fastest = successful_results[0]
        print(f"\n🏆 Fastest provider: {fastest[0]} ({fastest[3]:.2f} FPS)")
    else:
        print("\nNo providers successfully completed the benchmark.")

    # Print failed providers
    failed_results = [(p, e) for p, _, _, _, s, e in results if not s]
    if failed_results:
        print(f"\n❌ Failed providers ({len(failed_results)}):")
        for provider, error in failed_results:
            print(f"  - {provider}: {error}")


if __name__ == "__main__":
    main()
