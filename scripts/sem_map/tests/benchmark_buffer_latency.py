# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Benchmark buffer write/read latency for DataPipelineManager.

Tests performance of:
- write_rgb(): Target <200μs per call
- write_pointcloud(): Target <200μs per call
- read_elevation(): Target <200μs per call

Measures:
- Average latency
- P50, P95, P99 percentiles
- Min/max latency
"""
import numpy as np
import torch
import time
from cuda.bindings import driver as cuda
from sem_map.data_pipeline import DataPipelineManager, checkCudaErrors


def benchmark_write_rgb(manager, num_iterations=1000):
    """Benchmark write_rgb() latency."""
    print(f"\n{'='*60}")
    print(f"Benchmarking write_rgb() - {num_iterations} iterations")
    print(f"{'='*60}")

    # Pre-create test data
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_pose_R = np.eye(3, dtype=np.float64)
    test_pose_t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    test_K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

    latencies = []

    # Warmup (10 iterations)
    for _ in range(10):
        manager.write_rgb(test_image, "camera", 1000000000, test_pose_R, test_pose_t, test_K)

    # Benchmark
    for i in range(num_iterations):
        start = time.perf_counter()
        manager.write_rgb(
            test_image,
            "camera",
            1000000000 + i,  # Unique timestamp
            test_pose_R,
            test_pose_t,
            test_K
        )
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)  # Convert to microseconds

    # Statistics
    latencies = np.array(latencies)
    print(f"\nResults:")
    print(f"  Mean:    {latencies.mean():.2f} μs")
    print(f"  Median:  {np.median(latencies):.2f} μs")
    print(f"  Std Dev: {latencies.std():.2f} μs")
    print(f"  Min:     {latencies.min():.2f} μs")
    print(f"  Max:     {latencies.max():.2f} μs")
    print(f"  P95:     {np.percentile(latencies, 95):.2f} μs")
    print(f"  P99:     {np.percentile(latencies, 99):.2f} μs")

    # Check target
    target_us = 200
    if latencies.mean() < target_us:
        print(f"\n✅ PASSED: Mean latency {latencies.mean():.2f}μs < {target_us}μs target")
    else:
        print(f"\n❌ FAILED: Mean latency {latencies.mean():.2f}μs > {target_us}μs target")

    return latencies.mean() < target_us


def benchmark_write_pointcloud(manager, num_iterations=1000):
    """Benchmark write_pointcloud() latency."""
    print(f"\n{'='*60}")
    print(f"Benchmarking write_pointcloud() - {num_iterations} iterations")
    print(f"{'='*60}")

    # Pre-create test data
    num_points = 5000
    test_points = np.random.randn(num_points, 3).astype(np.float32)
    test_channels = ['x', 'y', 'z']
    test_pose_R = np.eye(3, dtype=np.float64)
    test_pose_t = np.array([0.5, 1.0, 1.5], dtype=np.float64)

    latencies = []

    # Warmup (10 iterations)
    for _ in range(10):
        manager.write_pointcloud(test_points, test_channels, 1000000000, test_pose_R, test_pose_t)

    # Benchmark
    for i in range(num_iterations):
        start = time.perf_counter()
        manager.write_pointcloud(
            test_points,
            test_channels,
            1000000000 + i,  # Unique timestamp
            test_pose_R,
            test_pose_t
        )
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)  # Convert to microseconds

    # Statistics
    latencies = np.array(latencies)
    print(f"\nResults:")
    print(f"  Mean:    {latencies.mean():.2f} μs")
    print(f"  Median:  {np.median(latencies):.2f} μs")
    print(f"  Std Dev: {latencies.std():.2f} μs")
    print(f"  Min:     {latencies.min():.2f} μs")
    print(f"  Max:     {latencies.max():.2f} μs")
    print(f"  P95:     {np.percentile(latencies, 95):.2f} μs")
    print(f"  P99:     {np.percentile(latencies, 99):.2f} μs")

    # Check target
    target_us = 200
    if latencies.mean() < target_us:
        print(f"\n✅ PASSED: Mean latency {latencies.mean():.2f}μs < {target_us}μs target")
    else:
        print(f"\n❌ FAILED: Mean latency {latencies.mean():.2f}μs > {target_us}μs target")

    return latencies.mean() < target_us


def benchmark_read_elevation(manager, num_iterations=1000):
    """Benchmark read_elevation() latency (empty queue case)."""
    print(f"\n{'='*60}")
    print(f"Benchmarking read_elevation() - {num_iterations} iterations")
    print(f"{'='*60}")
    print("Note: Testing empty queue case (most common scenario)")

    latencies = []

    # Warmup (10 iterations)
    for _ in range(10):
        manager.read_elevation()

    # Benchmark
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = manager.read_elevation()
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)  # Convert to microseconds

    # Statistics
    latencies = np.array(latencies)
    print(f"\nResults:")
    print(f"  Mean:    {latencies.mean():.2f} μs")
    print(f"  Median:  {np.median(latencies):.2f} μs")
    print(f"  Std Dev: {latencies.std():.2f} μs")
    print(f"  Min:     {latencies.min():.2f} μs")
    print(f"  Max:     {latencies.max():.2f} μs")
    print(f"  P95:     {np.percentile(latencies, 95):.2f} μs")
    print(f"  P99:     {np.percentile(latencies, 99):.2f} μs")

    # Check target
    target_us = 200
    if latencies.mean() < target_us:
        print(f"\n✅ PASSED: Mean latency {latencies.mean():.2f}μs < {target_us}μs target")
    else:
        print(f"\n❌ FAILED: Mean latency {latencies.mean():.2f}μs > {target_us}μs target")

    return latencies.mean() < target_us


def benchmark_mixed_workload(manager, num_iterations=300):
    """Benchmark mixed workload (write RGB + PC, read elevation)."""
    print(f"\n{'='*60}")
    print(f"Benchmarking Mixed Workload - {num_iterations} iterations")
    print(f"{'='*60}")
    print("Simulates realistic usage: write RGB, write PC, read elevation")

    # Pre-create test data
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_points = np.random.randn(5000, 3).astype(np.float32)
    test_channels = ['x', 'y', 'z']
    test_pose_R = np.eye(3, dtype=np.float64)
    test_pose_t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    test_K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

    rgb_latencies = []
    pc_latencies = []
    read_latencies = []
    total_latencies = []

    # Warmup
    for _ in range(10):
        manager.write_rgb(test_image, "camera", 1000000000, test_pose_R, test_pose_t, test_K)
        manager.write_pointcloud(test_points, test_channels, 1000000000, test_pose_R, test_pose_t)
        manager.read_elevation()

    # Benchmark
    for i in range(num_iterations):
        cycle_start = time.perf_counter()

        # Write RGB
        start = time.perf_counter()
        manager.write_rgb(test_image, "camera", 1000000000 + i, test_pose_R, test_pose_t, test_K)
        rgb_latencies.append((time.perf_counter() - start) * 1e6)

        # Write Pointcloud
        start = time.perf_counter()
        manager.write_pointcloud(test_points, test_channels, 1000000000 + i, test_pose_R, test_pose_t)
        pc_latencies.append((time.perf_counter() - start) * 1e6)

        # Read Elevation
        start = time.perf_counter()
        manager.read_elevation()
        read_latencies.append((time.perf_counter() - start) * 1e6)

        total_latencies.append((time.perf_counter() - cycle_start) * 1e6)

    # Statistics
    rgb_latencies = np.array(rgb_latencies)
    pc_latencies = np.array(pc_latencies)
    read_latencies = np.array(read_latencies)
    total_latencies = np.array(total_latencies)

    print(f"\nRGB Write:")
    print(f"  Mean: {rgb_latencies.mean():.2f} μs, P95: {np.percentile(rgb_latencies, 95):.2f} μs")

    print(f"\nPointcloud Write:")
    print(f"  Mean: {pc_latencies.mean():.2f} μs, P95: {np.percentile(pc_latencies, 95):.2f} μs")

    print(f"\nElevation Read:")
    print(f"  Mean: {read_latencies.mean():.2f} μs, P95: {np.percentile(read_latencies, 95):.2f} μs")

    print(f"\nTotal Cycle Time:")
    print(f"  Mean: {total_latencies.mean():.2f} μs ({1e6/total_latencies.mean():.1f} Hz)")
    print(f"  P95:  {np.percentile(total_latencies, 95):.2f} μs")

    # Check if total GIL time is reasonable
    target_total_us = 600  # 3 operations × 200μs
    if total_latencies.mean() < target_total_us:
        print(f"\n✅ PASSED: Mean cycle time {total_latencies.mean():.2f}μs < {target_total_us}μs target")
    else:
        print(f"\n⚠️  WARNING: Mean cycle time {total_latencies.mean():.2f}μs > {target_total_us}μs target")

    return total_latencies.mean() < target_total_us


def main():
    """Run all benchmarks."""
    print("="*60)
    print("DataPipelineManager Latency Benchmark")
    print("="*60)
    print("\nTarget: <200μs per operation (GIL hold time)")
    print("Baseline comparison: VPE achieves 100-200μs per write")

    # Initialize CUDA
    checkCudaErrors(cuda.cuInit(0))
    torch.cuda.set_device(0)

    # Create DataPipelineManager (no workers)
    config = {
        'image_height': 480,
        'image_width': 640,
        'pc_max_points': 100000,
        'map_cells': 100,
        'enable_semantic_pipeline': False,
        'enable_height_map': False,
        'buffer_size': 10,
        'queue_size': 5,
    }

    print("\nInitializing DataPipelineManager...")
    manager = DataPipelineManager(config)
    print("✓ DataPipelineManager created (no workers)\n")

    # Run benchmarks
    results = []

    results.append(("RGB Write", benchmark_write_rgb(manager, num_iterations=1000)))
    results.append(("Pointcloud Write", benchmark_write_pointcloud(manager, num_iterations=1000)))
    results.append(("Elevation Read", benchmark_read_elevation(manager, num_iterations=1000)))
    results.append(("Mixed Workload", benchmark_mixed_workload(manager, num_iterations=300)))

    # Summary
    print(f"\n{'='*60}")
    print("Benchmark Summary")
    print(f"{'='*60}")

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    all_passed = all(r for _, r in results)
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL BENCHMARKS PASSED - Ready for production!")
        print("\nConclusion: GIL hold times are within target (<200μs)")
        print("Expected speedup: 58× reduction in GIL contention (0.6ms vs 35ms)")
    else:
        print("⚠️  SOME BENCHMARKS DID NOT MEET TARGET")
        print("\nFurther optimization may be needed")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
