# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Test semantic worker process integration.

Validates:
- Worker spawns successfully
- Receives RGB frames from buffer
- Processes with SemanticPipeline
- Outputs traversability to buffer
"""
import numpy as np
import torch
import time
from cuda.bindings import driver as cuda
from sem_map.data_pipeline import DataPipelineManager, checkCudaErrors
from sem_map.parameter import Parameter


def test_semantic_worker_spawn():
    """Test that semantic worker spawns and initializes without errors."""
    print("\n" + "="*60)
    print("Testing Semantic Worker Spawn")
    print("="*60)

    # Initialize CUDA
    checkCudaErrors(cuda.cuInit(0))
    torch.cuda.set_device(0)
    print("✓ CUDA initialized")

    # Create Parameter instance
    param = Parameter()
    print(f"✓ Parameter created (backbone: {param.backbone_name})")

    # Create config
    config = {
        'image_height': 480,
        'image_width': 640,
        'pc_max_points': 100000,
        'map_cells': 100,
        'enable_semantic_pipeline': True,  # Enable worker
        'enable_height_map': False,        # Disable elevation for now
        'buffer_size': 10,
        'queue_size': 5,
    }

    # Create DataPipelineManager with param
    print("\nCreating DataPipelineManager...")
    manager = DataPipelineManager(config, param)
    print("✓ DataPipelineManager created")

    # Start workers (spawns semantic process)
    print("\nStarting workers...")
    manager.start()
    print("✓ Workers started")

    # Give worker time to initialize
    print("\nWaiting for worker initialization...")
    time.sleep(3)

    # Check health
    healthy, msg = manager.check_health()
    if healthy:
        print(f"✓ Worker health check: {msg}")
    else:
        print(f"✗ Worker health check FAILED: {msg}")
        manager.shutdown()
        return False

    # Shutdown
    print("\nShutting down...")
    manager.shutdown(timeout=5.0)
    print("✓ Shutdown complete")

    return True


def test_semantic_worker_processing():
    """Test that semantic worker can process RGB frames end-to-end."""
    print("\n" + "="*60)
    print("Testing Semantic Worker Processing (End-to-End)")
    print("="*60)

    # Initialize CUDA
    checkCudaErrors(cuda.cuInit(0))
    torch.cuda.set_device(0)
    print("✓ CUDA initialized")

    # Create Parameter instance
    param = Parameter()
    print(f"✓ Parameter created")

    # Create config
    config = {
        'image_height': 480,
        'image_width': 640,
        'pc_max_points': 100000,
        'map_cells': 100,
        'enable_semantic_pipeline': True,
        'enable_height_map': False,
        'buffer_size': 10,
        'queue_size': 5,
    }

    # Create and start DataPipelineManager
    print("\nCreating DataPipelineManager...")
    manager = DataPipelineManager(config, param)
    manager.start()
    print("✓ Workers started")

    # Give worker time to initialize TensorRT
    print("\nWaiting for TensorRT initialization (this may take 10-15s)...")
    time.sleep(15)

    # Check if worker is still alive
    healthy, msg = manager.check_health()
    if not healthy:
        print(f"✗ Worker died during initialization: {msg}")
        manager.shutdown()
        return False
    print(f"✓ Worker initialized successfully")

    # Write test RGB frames
    print("\nWriting test RGB frames...")
    np.random.seed(0)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_pose_R = np.eye(3, dtype=np.float64)
    test_pose_t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    test_K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

    num_frames = 5
    for i in range(num_frames):
        manager.write_rgb(
            test_image,
            "test_camera",
            1000000000 + i * 100000000,  # Timestamps 100ms apart
            test_pose_R,
            test_pose_t,
            test_K
        )
        print(f"  ✓ Wrote frame {i+1}/{num_frames}")
        time.sleep(0.1)  # Small delay between frames

    # Give worker time to process
    print("\nWaiting for processing...")
    time.sleep(2)

    # Try to read traversability output
    print("\nReading traversability output...")
    result = manager.sem_results_buffer.get(last_timestamp=0.0)

    if result is None:
        print("✗ No traversability output received")
        manager.shutdown()
        return False

    item_id, timestamp, trav_data = result
    print(f"✓ Received traversability output:")
    print(f"    - Timestamp: {timestamp}")
    print(f"    - Keys: {list(trav_data.keys())}")

    # Validate output
    assert 'traversability' in trav_data, "Missing 'traversability' key"
    assert 'segmentation' in trav_data, "Missing 'segmentation' key"
    assert 'pose_R' in trav_data, "Missing 'pose_R' key"
    assert 'pose_t' in trav_data, "Missing 'pose_t' key"
    assert 'intrinsics' in trav_data, "Missing 'intrinsics' key"

    trav = trav_data['traversability']
    print(f"    - Traversability shape: {trav.shape}")
    print(f"    - Traversability device: {trav.device}")
    print(f"    - Traversability range: [{trav.min():.3f}, {trav.max():.3f}]")

    # Check shape
    expected_shape = (480, 640)  # Or might be downsampled
    if trav.shape[0] > 0 and trav.shape[1] > 0:
        print(f"✓ Traversability has valid shape: {trav.shape}")
    else:
        print(f"✗ Traversability has invalid shape: {trav.shape}")
        manager.shutdown()
        return False

    # Check if worker processed multiple frames
    time.sleep(1)
    healthy, msg = manager.check_health()
    if not healthy:
        print(f"✗ Worker crashed during processing: {msg}")
        manager.shutdown()
        return False

    print("✓ Worker still healthy after processing")

    # Shutdown
    print("\nShutting down...")
    manager.shutdown(timeout=5.0)
    print("✓ Shutdown complete")

    return True


def main():
    """Run all semantic worker tests."""
    print("="*60)
    print("Semantic Worker Integration Tests")
    print("="*60)
    print("\nWARNING: These tests will:")
    print("- Load TensorRT models (~10-15s initialization)")
    print("- Spawn worker processes")
    print("- Use GPU memory")
    print()

    tests = [
        ("Semantic Worker Spawn", test_semantic_worker_spawn),
        ("Semantic Worker Processing", test_semantic_worker_processing),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            print()
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
        print("\nPhase 2 Milestone: Semantic worker integration validated!")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nCheck error messages above for details")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
