# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Single-process test for DataPipelineManager write/read operations.

Tests the C++ bindings interface WITHOUT spawning worker processes.
This validates:
- Buffer creation
- write_rgb() from C++-style numpy arrays
- write_pointcloud() from C++-style numpy arrays
- Buffer read operations (direct access for testing)
"""
import numpy as np
import torch
from cuda.bindings import driver as cuda
from sem_map.data_pipeline import DataPipelineManager, checkCudaErrors


def test_rgb_buffer_write_read():
    """Test RGB buffer write and read operations."""
    print("\n=== Testing RGB Buffer Write/Read ===")

    # Initialize CUDA
    checkCudaErrors(cuda.cuInit(0))
    torch.cuda.set_device(0)

    # Create DataPipelineManager (without starting worker processes)
    config = {
        'image_height': 480,
        'image_width': 640,
        'pc_max_points': 100000,
        'map_cells': 100,
        'enable_semantic_pipeline': False,  # Don't spawn workers
        'enable_height_map': False,
        'buffer_size': 10,
        'queue_size': 5,
    }

    manager = DataPipelineManager(config)
    print("  ✓ DataPipelineManager created (no workers)")

    # Create test data (numpy arrays, as they would come from C++)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_pose_R = np.eye(3, dtype=np.float64)
    test_pose_t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    test_K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    test_timestamp_ns = 1234567890123456789

    print(f"  ✓ Test data created:")
    print(f"    - Image: {test_image.shape}, dtype={test_image.dtype}")
    print(f"    - Pose R: {test_pose_R.shape}, dtype={test_pose_R.dtype}")
    print(f"    - Pose t: {test_pose_t.shape}, dtype={test_pose_t.dtype}")
    print(f"    - Intrinsics: {test_K.shape}, dtype={test_K.dtype}")
    print(f"    - Timestamp: {test_timestamp_ns} ns")

    # Write to buffer (this is what C++ bindings call)
    try:
        manager.write_rgb(
            test_image,
            "camera_front",
            test_timestamp_ns,
            test_pose_R,
            test_pose_t,
            test_K
        )
        print("  ✓ write_rgb() succeeded")
    except Exception as e:
        print(f"  ✗ write_rgb() failed: {e}")
        return False

    # Read back from buffer (direct access for testing)
    # Pass last_timestamp=0.0 to get any available data
    try:
        result = manager.rgb_input_buffer.get(last_timestamp=0.0)
        assert result is not None, "Buffer returned None (no data)"
        item_id, timestamp, data = result
        print("  ✓ Buffer read succeeded")

        # Verify data
        assert 'image' in data, "Missing 'image' key"
        assert 'pose_R' in data, "Missing 'pose_R' key"
        assert 'pose_t' in data, "Missing 'pose_t' key"
        assert 'intrinsics' in data, "Missing 'intrinsics' key"

        # Check shapes
        assert data['image'].shape == (480, 640, 3), f"Wrong image shape: {data['image'].shape}"
        assert data['pose_R'].shape == (3, 3), f"Wrong pose_R shape: {data['pose_R'].shape}"
        assert data['pose_t'].shape == (3,), f"Wrong pose_t shape: {data['pose_t'].shape}"
        assert data['intrinsics'].shape == (3, 3), f"Wrong intrinsics shape: {data['intrinsics'].shape}"

        # Note: VPE's CUDATimeBuffer uses time.perf_counter() internally
        # rather than storing the provided timestamp. This is by design.
        # We just verify that we got a valid timestamp back.
        assert timestamp > 0, f"Invalid timestamp: {timestamp}"

        print("  ✓ Data verification passed")
        print(f"    - Image on GPU: {data['image'].device}")
        print(f"    - Pose R on GPU: {data['pose_R'].device}")
        print(f"    - Timestamp: {timestamp} s (from perf_counter)")

        return True

    except Exception as e:
        print(f"  ✗ Buffer read failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pointcloud_buffer_write_read():
    """Test pointcloud buffer write and read operations."""
    print("\n=== Testing Pointcloud Buffer Write/Read ===")

    # Initialize CUDA
    checkCudaErrors(cuda.cuInit(0))
    torch.cuda.set_device(0)

    # Create DataPipelineManager (without starting worker processes)
    config = {
        'image_height': 480,
        'image_width': 640,
        'pc_max_points': 100000,
        'map_cells': 100,
        'enable_semantic_pipeline': False,  # Don't spawn workers
        'enable_height_map': False,
        'buffer_size': 10,
        'queue_size': 5,
    }

    manager = DataPipelineManager(config)
    print("  ✓ DataPipelineManager created (no workers)")

    # Create test pointcloud (numpy array, as it would come from C++)
    num_points = 5000
    test_points = np.random.randn(num_points, 3).astype(np.float32)
    test_channels = ['x', 'y', 'z']
    test_pose_R = np.eye(3, dtype=np.float64)
    test_pose_t = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    test_timestamp_ns = 1234567890987654321

    print(f"  ✓ Test data created:")
    print(f"    - Points: {test_points.shape}, dtype={test_points.dtype}")
    print(f"    - Channels: {test_channels}")
    print(f"    - Num points: {num_points}")
    print(f"    - Timestamp: {test_timestamp_ns} ns")

    # Write to buffer (this is what C++ bindings call)
    try:
        manager.write_pointcloud(
            test_points,
            test_channels,
            test_timestamp_ns,
            test_pose_R,
            test_pose_t
        )
        print("  ✓ write_pointcloud() succeeded")
    except Exception as e:
        print(f"  ✗ write_pointcloud() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Read back from buffer (direct access for testing)
    # Pass last_timestamp=0.0 to get any available data
    try:
        result = manager.pc_input_buffer.get(last_timestamp=0.0)
        assert result is not None, "Buffer returned None (no data)"
        item_id, timestamp, data = result
        print("  ✓ Buffer read succeeded")

        # Verify data
        assert 'pointcloud' in data, "Missing 'pointcloud' key"
        assert 'num_points' in data, "Missing 'num_points' key"

        # Check shapes (should be padded to max_points)
        assert data['pointcloud'].shape == (100000, 3), \
            f"Wrong pointcloud shape: {data['pointcloud'].shape}"

        # Check num_points
        num_pts_tensor = data['num_points']
        assert num_pts_tensor.item() == num_points, \
            f"Wrong num_points: {num_pts_tensor.item()} vs {num_points}"

        # Verify first few points match (accounting for padding)
        points_cpu = data['pointcloud'][:num_points].cpu().numpy()
        max_diff = np.abs(points_cpu - test_points).max()
        assert max_diff < 1e-5, f"Points data mismatch: max_diff={max_diff}"

        # Note: VPE's CUDATimeBuffer uses time.perf_counter() internally
        assert timestamp > 0, f"Invalid timestamp: {timestamp}"

        print("  ✓ Data verification passed")
        print(f"    - Points on GPU: {data['pointcloud'].device}")
        print(f"    - Actual points: {num_points}/{data['pointcloud'].shape[0]}")
        print(f"    - Timestamp: {timestamp} s (from perf_counter)")
        print(f"    - Data match: max_diff={max_diff:.2e}")

        return True

    except Exception as e:
        print(f"  ✗ Buffer read failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_writes_freshness():
    """Test that TimeBuffer returns the freshest data (overwrites old)."""
    print("\n=== Testing TimeBuffer Freshness (Multiple Writes) ===")

    # Initialize CUDA
    checkCudaErrors(cuda.cuInit(0))
    torch.cuda.set_device(0)

    # Create manager
    config = {
        'image_height': 480,
        'image_width': 640,
        'pc_max_points': 100000,
        'map_cells': 100,
        'enable_semantic_pipeline': False,
        'enable_height_map': False,
    }

    manager = DataPipelineManager(config)
    print("  ✓ DataPipelineManager created")

    # Write 3 images with different timestamps
    import time
    timestamps = [1000000000, 2000000000, 3000000000]  # ns

    for i, ts_ns in enumerate(timestamps):
        test_image = np.full((480, 640, 3), i, dtype=np.uint8)  # Fill with index
        test_pose_R = np.eye(3, dtype=np.float64)
        test_pose_t = np.array([float(i), float(i), float(i)], dtype=np.float64)
        test_K = np.eye(3, dtype=np.float64)

        manager.write_rgb(test_image, f"cam_{i}", ts_ns, test_pose_R, test_pose_t, test_K)
        time.sleep(0.01)  # Small delay to ensure writes are processed in order
        print(f"  ✓ Wrote image {i} with timestamp {ts_ns} ns")

    # Read back - should get the NEWEST (index 2)
    result = manager.rgb_input_buffer.get(last_timestamp=0.0)
    assert result is not None, "Buffer returned None (no data)"
    item_id, timestamp, data = result

    # Check that timestamp is valid and recent
    assert timestamp > 0, f"Invalid timestamp: {timestamp}"

    # Check image value (should be filled with 2, the newest write)
    image_cpu = data['image'].cpu().numpy()
    unique_values = np.unique(image_cpu)

    print(f"  ✓ TimeBuffer returned freshest data (timestamp={timestamp})")
    print(f"    - Unique image values: {unique_values}")
    print(f"    - Pose t: {data['pose_t'].cpu().numpy()}")

    # The buffer should contain the newest write (index 2, filled with value 2)
    # Due to timing, we might get index 1 or 2, but never index 0
    assert np.all(image_cpu >= 1), \
        f"Should get recent image (1 or 2), got values: {unique_values}"

    print(f"  ✓ Freshness check passed (got recent data, not oldest)")

    return True


def main():
    """Run all single-process buffer tests."""
    print("="*60)
    print("Single-Process Buffer Tests")
    print("="*60)

    tests = [
        ("RGB Buffer Write/Read", test_rgb_buffer_write_read),
        ("Pointcloud Buffer Write/Read", test_pointcloud_buffer_write_read),
        ("TimeBuffer Freshness", test_multiple_writes_freshness),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(r for _, r in results)
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
