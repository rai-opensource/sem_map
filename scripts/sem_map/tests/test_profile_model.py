#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Comprehensive test script for sem_map modular segmentation pipeline
Usage: python test_profile_model.py [test_type] [options]
"""

import sys
import os
import subprocess
import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sem_map.perception.common import get_sem_map_root
import yaml
from sem_map.perception.models import load_models_yaml, resolve_artifact_path

class TestResultLogger:
    """Handles logging of test results to JSON and CSV formats"""

    def __init__(self, output_dir: str = None):
        """Initialize logger with output directory"""
        if output_dir is None:
            # Default to output/profile_model/ in the repo root
            output_dir = get_sem_map_root() / "output" / "profile_model"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{timestamp}"
        self.session_dir.mkdir(exist_ok=True)

        self.results = []
        self.session_info = {
            "timestamp": timestamp,
            "start_time": datetime.now().isoformat(),
            "test_type": None,
            "image_path": None,
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0
        }

    def set_session_info(self, test_type: str, image_path: str):
        """Set session information"""
        self.session_info["test_type"] = test_type
        self.session_info["image_path"] = image_path

    def log_result(self, test_config: Dict, success: bool, metrics: Dict = None,
                   error_msg: str = None, stdout: str = None, stderr: str = None):
        """Log a single test result"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "test_config": test_config,
            "success": success,
            "metrics": metrics or {},
            "error_msg": error_msg,
            "stdout": stdout,
            "stderr": stderr
        }
        self.results.append(result)

        if success:
            self.session_info["successful_tests"] += 1
        else:
            self.session_info["failed_tests"] += 1

        self.session_info["total_tests"] += 1

    def save_results(self):
        """Save results to JSON and CSV files"""
        self.session_info["end_time"] = datetime.now().isoformat()
        self.session_info["duration_seconds"] = (
            datetime.fromisoformat(self.session_info["end_time"]) -
            datetime.fromisoformat(self.session_info["start_time"])
        ).total_seconds()

        # Save JSON results
        json_file = self.session_dir / "results.json"
        with open(json_file, 'w') as f:
            json.dump({
                "session_info": self.session_info,
                "results": self.results
            }, f, indent=2)

        # Save CSV results
        csv_file = self.session_dir / "results.csv"
        if self.results:
            fieldnames = [
                "timestamp", "backbone", "upsampler", "upsampler_backend",
                "seg_backend", "jafar_factor", "skip_seg", "success",
                "avg_ms", "fps", "mem_peak_mb", "error_msg"
            ]

            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in self.results:
                    config = result["test_config"]
                    metrics = result["metrics"]
                    row = {
                        "timestamp": result["timestamp"],
                        "backbone": config.get("backbone", ""),
                        "upsampler": config.get("upsampler", ""),
                        "upsampler_backend": config.get("upsampler_backend", ""),
                        "seg_backend": config.get("seg", ""),
                        "jafar_factor": config.get("jafar_factor", ""),
                        "skip_seg": config.get("skip_seg", False),
                        "success": result["success"],
                        "avg_ms": metrics.get("avg_ms", ""),
                        "fps": metrics.get("fps", ""),
                        "mem_peak_mb": metrics.get("mem_peak_mb", ""),
                        "error_msg": result.get("error_msg", "")
                    }
                    writer.writerow(row)

        # Save summary
        summary_file = self.session_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Profile Model Test Session Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Session: {self.session_info['timestamp']}\n")
            f.write(f"Test Type: {self.session_info['test_type']}\n")
            f.write(f"Image: {self.session_info['image_path']}\n")
            f.write(f"Duration: {self.session_info['duration_seconds']:.2f} seconds\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total Tests: {self.session_info['total_tests']}\n")
            f.write(f"  Successful: {self.session_info['successful_tests']}\n")
            f.write(f"  Failed: {self.session_info['failed_tests']}\n")
            f.write(f"  Success Rate: {self.session_info['successful_tests']/max(1, self.session_info['total_tests'])*100:.1f}%\n\n")
            f.write(f"Files:\n")
            f.write(f"  JSON: {json_file}\n")
            f.write(f"  CSV: {csv_file}\n")

        print(f"\nResults saved to: {self.session_dir}")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")
        print(f"  Summary: {summary_file}")

def get_scripts_dir() -> Path:
    """Get the installed scripts root that contains the sem_map package."""
    # During development we expect PYTHONPATH to include <repo>/scripts,
    # where the package folder sem_map/ resides.
    return get_sem_map_root() / "scripts"

def check_environment():
    """Check test environment"""
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)

    # Check directories
    sem_map_root = get_sem_map_root()
    scripts_dir = get_scripts_dir()
    config_dir = sem_map_root / "config"
    artifacts_dir = sem_map_root / "artifacts"

    module_file = scripts_dir / "sem_map" / "cli" / "profile_model.py"
    checks = [
        ("sem_map_root", sem_map_root.exists()),
        ("scripts_dir", scripts_dir.exists()),
        ("config_dir", config_dir.exists()),
        ("artifacts_dir", artifacts_dir.exists()),
        ("profile_model.py", module_file.exists()),
        ("models.yaml", (config_dir / "models.yaml").exists()),
    ]

    for name, exists in checks:
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {exists}")

    # Check Python imports (package import via sem_map)
    try:
        sys.path.insert(0, str(get_scripts_dir()))
        from sem_map.perception import get_device  # noqa: F401
        print("✓ sem_map.perception import: works")
    except ImportError as e:
        print(f"✗ sem_map.perception import: {e}")

    print("=" * 50)

def _get_default_seg_layer(family: str = "dinov3s") -> str | None:
    """Read models.yaml and return the first seg_layer key for the given family using cached loader."""
    try:
        models = load_models_yaml()
        fam = (models or {}).get("families", {}).get(family, {})
        seg_layers = fam.get("seg_layers") or {}
        if isinstance(seg_layers, dict) and seg_layers:
            return next(iter(seg_layers.keys()))
    except Exception:
        pass
    return None

def run_profile_command(backbone: str, upsampler: str, seg: str,
                       image_path: str, jafar_factor: int = None,
                       warmup: int = 5, iters: int = 20,
                       upsampler_backend: str | None = None,
                       skip_seg: bool = False, logger: TestResultLogger = None,
                       seg_layer: str | None = None) -> bool:
    """Run a single profiling command"""
    family = "dinov3s"

    # Preflight: skip combos that require missing artifacts
    try:
        models = load_models_yaml()
        # Backbone
        if backbone in ("onnx", "trt"):
            art = resolve_artifact_path(family, kind="backbone", artifact_type=("onnx" if backbone == "onnx" else "trt"))
            if not art.exists():
                print(f"  SKIP [bb={backbone}] missing backbone artifact: {art}")
                return True
        # Upsampler
        if upsampler in ("jafar_full", "jafar_partial") and (upsampler_backend in ("onnx", "trt")):
            factor = 16 if upsampler == "jafar_full" else (jafar_factor or 4)
            art = resolve_artifact_path(family, kind="upsampler", artifact_type=("onnx" if upsampler_backend == "onnx" else "trt"), upsampler="jafar", upsample_factor=factor)
            if not art.exists():
                print(f"  SKIP [up={upsampler}:{upsampler_backend}] missing jafar artifact: {art}")
                return True
        # Seg head
        if (not skip_seg) and seg == "trt":
            chosen_seg_layer = seg_layer or _get_default_seg_layer(family)
            if chosen_seg_layer:
                art = resolve_artifact_path(family, kind="seg_layer", artifact_type="trt", seg_layer=chosen_seg_layer)
                if not art.exists():
                    print(f"  SKIP [seg=trt] missing seg artifact: {art}")
                    return True
    except Exception:
        pass

    cmd = [
        "python", "-m", "sem_map.cli.profile_model",
        "--image", image_path,
        "--model_name", family,
        "--backbone_backend", backbone,
        "--upsampler_mode", upsampler,
        "--seg_backend", seg,
        "--warmup", str(warmup),
        "--iters", str(iters)
    ]

    if (not skip_seg):
        chosen_seg_layer = seg_layer or _get_default_seg_layer(family)
        if chosen_seg_layer:
            cmd.extend(["--seg_layer", chosen_seg_layer])

    if upsampler == "jafar_partial" and jafar_factor:
        cmd.extend(["--upsample_partial_factor", str(jafar_factor)])
    # Pass explicit upsampler backend if provided (used only for JAFAR modes)
    if upsampler in ("jafar_full", "jafar_partial") and upsampler_backend:
        cmd.extend(["--upsampler_backend", upsampler_backend])
    if skip_seg:
        cmd.append("--skip_seg")

    # Prepare test configuration for logging
    test_config = {
        "backbone": backbone,
        "upsampler": upsampler,
        "upsampler_backend": upsampler_backend,
        "seg": seg,
        "jafar_factor": jafar_factor,
        "skip_seg": skip_seg,
        "warmup": warmup,
        "iters": iters,
        "image_path": image_path
    }

    try:
        env = dict(os.environ)
        # Ensure child process can import sem_map
        env["PYTHONPATH"] = str(get_scripts_dir()) + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        up_tag_print = (
            f"{upsampler}:{upsampler_backend}" if upsampler in ("jafar_full", "jafar_partial") and upsampler_backend
            else upsampler
        )

        if result.returncode == 0:
            # Extract key metrics
            output = result.stdout
            metrics = {}
            if "avg_ms=" in output and "fps=" in output:
                lines = output.strip().split('\n')
                for line in lines:
                    if "avg_ms=" in line:
                        # Parse metrics from output line
                        parts = line.split()
                        for part in parts:
                            if "avg_ms=" in part:
                                try:
                                    metrics["avg_ms"] = float(part.split("=")[1])
                                except (ValueError, IndexError):
                                    pass
                            elif "fps=" in part:
                                try:
                                    metrics["fps"] = float(part.split("=")[1])
                                except (ValueError, IndexError):
                                    pass
                            elif "mem_peak_mb=" in part:
                                try:
                                    metrics["mem_peak_mb"] = float(part.split("=")[1])
                                except (ValueError, IndexError):
                                    pass
                        print(f"  ✓ [bb={backbone} | up={up_tag_print} | seg={seg}] {line.strip()}")
                        break

            # Log successful result
            if logger:
                logger.log_result(test_config, True, metrics, stdout=result.stdout, stderr=result.stderr)
            return True
        else:
            print(f"  ✗ [bb={backbone} | up={up_tag_print} | seg={seg}]: FAILED")
            # Print tail of stdout/stderr for debugging
            out_tail = (result.stdout or "").strip().split('\n')[-20:]
            err_tail = (result.stderr or "").strip().split('\n')[-20:]
            if out_tail and any(line.strip() for line in out_tail):
                print("    └─ stdout (last lines):")
                for line in out_tail:
                    if line.strip():
                        print(f"       {line}")
            if err_tail and any(line.strip() for line in err_tail):
                print("    └─ stderr (last lines):")
                for line in err_tail:
                    if line.strip():
                        print(f"       {line}")

            # Log failed result
            if logger:
                error_msg = f"Command failed with return code {result.returncode}"
                logger.log_result(test_config, False, error_msg=error_msg,
                                stdout=result.stdout, stderr=result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ [bb={backbone} | up={up_tag_print} | seg={seg}]: TIMEOUT")
        if logger:
            logger.log_result(test_config, False, error_msg="Command timed out after 300 seconds")
        return False
    except Exception as e:
        print(f"  ✗ [bb={backbone} | up={up_tag_print} | seg={seg}]: ERROR - {e}")
        if logger:
            logger.log_result(test_config, False, error_msg=str(e))
        return False

def quick_test(image_path: str, logger: TestResultLogger = None):
    """Run quick tests (10 key combinations)"""
    print("=" * 50)
    print("Quick Profiling Tests")
    print("=" * 50)

    tests = [
        # backbone, upsampler_mode, seg, jafar_factor, description, upsampler_backend
        ("torch", "bilinear", "torch", None, "PyTorch Baseline", None),
        ("trt", "bilinear", "torch", None, "TensorRT Backbone", None),
        ("onnx", "bilinear", "torch", None, "ONNX Backbone", None),
        ("torch", "jafar_full", "torch", None, "PyTorch JAFAR Full", "torch"),
        ("trt", "jafar_full", "trt", None, "TensorRT JAFAR Full", "trt"),
        ("trt", "jafar_partial", "trt", 4, "TensorRT JAFAR Partial f4", "trt"),
        ("trt", "jafar_partial", "torch", 4, "TensorRT JAFAR Partial f4", "trt"),
        ("trt", "jafar_full", "torch", None, "TensorRT JAFAR Full (seg TRT)", "trt"),
        ("trt", "bilinear", "trt", None, "Mixed TRT+TRT", None),
        # New: Trt backbone + bilinear upsampling without seg
        ("trt", "bilinear", "trt", None, "TRT Backbone Bilinear (no seg)", None, True),
    ]

    successful = 0
    for i, entry in enumerate(tests, 1):
        # Support optional skip_seg as 7th element
        if len(entry) == 6:
            backbone, upsampler, seg, factor, description, up_backend = entry
            skip_seg = False
        else:
            backbone, upsampler, seg, factor, description, up_backend, skip_seg = entry
        up_tag = (f"{upsampler}:{up_backend}" if upsampler in ("jafar_full", "jafar_partial") and up_backend else upsampler)
        seg_tag = ("no_seg" if skip_seg else seg)
        print(f"{i}. {description}  [bb={backbone} | up={up_tag} | seg={seg_tag}]")
        if run_profile_command(backbone, upsampler, seg, image_path, factor, upsampler_backend=up_backend, skip_seg=skip_seg, logger=logger):
            successful += 1
        print()

    print(f"Quick tests completed: {successful}/{len(tests)} successful")
    print("=" * 50)

def backend_comparison(image_path: str, upsampler: str = "bilinear", logger: TestResultLogger = None):
    """Compare PyTorch vs ONNX vs TensorRT backends"""
    print("=" * 50)
    print(f"Backend Comparison ({upsampler} upsampling)")
    print("=" * 50)

    backends = ["torch", "onnx", "trt"]
    results = {}

    for backend in backends:
        up_tag = (f"{upsampler}:{backend}" if upsampler in ("jafar_full", "jafar_partial") else upsampler)
        print(f"Testing...  [bb={backend} | up={up_tag} | seg={backend}]")
        # Tie upsampler_backend to the tested backend for JAFAR modes
        up_backend = backend if upsampler in ("jafar_full", "jafar_partial") else None
        if run_profile_command(backend, upsampler, backend, image_path, upsampler_backend=up_backend, logger=logger):
            results[backend] = "SUCCESS"
        else:
            results[backend] = "FAILED"
        print()

    print("Results Summary:")
    print("-" * 30)
    for backend, status in results.items():
        print(f"{backend}: {status}")
    print("=" * 50)

def comprehensive_test(image_path: str, output_dir: str = None, logger: TestResultLogger = None):
    """Run comprehensive tests (all combinations)"""
    print("=" * 50)
    print("Comprehensive Profiling Tests")
    print("WARNING: This will take 1-2 hours!")
    print("=" * 50)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    backends = ["torch", "onnx", "trt"]
    upsamplers = ["bilinear", "jafar_full", "jafar_partial"]
    jafar_factors = [2, 4, 8]

    total_combinations = 0
    successful = 0

    # Count total combinations
    for backbone in backends:
        for seg in backends:
            for upsampler in upsamplers:
                if upsampler == "jafar_partial":
                    total_combinations += len(jafar_factors)
                else:
                    total_combinations += 1

    print(f"Total combinations to test: {total_combinations}")
    print()

    # Run all combinations
    for backbone in backends:
        for seg in backends:
            for upsampler in upsamplers:
                if upsampler == "jafar_partial":
                    for factor in jafar_factors:
                        up_tag = f"{upsampler}:{backbone}"  # tie upsampler backend to backbone
                        print(f"Testing: [bb={backbone} | up={up_tag}f{factor} | seg={seg}]")
                        # Tie upsampler_backend to backbone by default for comprehensive grid
                        if run_profile_command(backbone, upsampler, seg, image_path, factor, 10, 100, upsampler_backend=backbone, logger=logger):
                            successful += 1
                        print()
                else:
                    up_tag = (f"{upsampler}:{backbone}" if upsampler in ("jafar_full",) else upsampler)
                    print(f"Testing: [bb={backbone} | up={up_tag} | seg={seg}]")
                    up_backend = backbone if upsampler in ("jafar_full", "jafar_partial") else None
                    if run_profile_command(backbone, upsampler, seg, image_path, None, 10, 100, upsampler_backend=up_backend, logger=logger):
                        successful += 1
                    print()

    print(f"Comprehensive tests completed: {successful}/{total_combinations} successful")
    print("=" * 50)

def smoke_test():
    """Run basic smoke tests"""
    print("=" * 50)
    print("Smoke Tests")
    print("=" * 50)

    scripts_dir = get_scripts_dir()
    sem_map_root = get_sem_map_root()

    tests = [
        ("profile_model help", ["python", "-m", "sem_map.cli.profile_model", "--help"]),
        ("viz_seg help", ["python", "-m", "sem_map.cli.viz_seg", "--help"]),
    ]

    for test_name, cmd in tests:
        print(f"Testing: {test_name}")
        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(get_scripts_dir()) + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
            if result.returncode == 0:
                print(f"  ✓ {test_name}: SUCCESS")
            else:
                print(f"  ✗ {test_name}: FAILED")
        except Exception as e:
            print(f"  ✗ {test_name}: ERROR - {e}")
        print()

    print("Smoke tests completed!")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive test script for sem_map")
    parser.add_argument("test_type", choices=["env", "quick", "backend", "comprehensive", "smoke"],
                       help="Type of test to run")
    parser.add_argument("--image", default="data/custom_datasets/example/images/000000.png",
                       help="Path to test image")
    parser.add_argument("--upsampler", default="bilinear", choices=["bilinear", "jafar_full", "jafar_partial"],
                       help="Upsampler mode for backend comparison")
    parser.add_argument("--output", help="Output directory for comprehensive tests")
    parser.add_argument("--log-dir", help="Custom log directory (default: output/profile_model/)")
    parser.add_argument("--no-log", action="store_true", help="Disable logging to files")

    args = parser.parse_args()

    print("=" * 50)
    print("Sem Map Test Suite")
    print("=" * 50)
    print(f"Test Type: {args.test_type}")
    print(f"Image: {args.image}")
    print("=" * 50)

    # Initialize logger for test types that need it
    logger = None
    if not args.no_log and args.test_type in ["quick", "backend", "comprehensive"]:
        logger = TestResultLogger(args.log_dir)
        logger.set_session_info(args.test_type, args.image)
        print(f"Logging enabled. Results will be saved to: {logger.session_dir}")
        print("=" * 50)

    try:
        if args.test_type == "env":
            check_environment()
        elif args.test_type == "quick":
            if not os.path.exists(args.image):
                print(f"Error: Image file not found: {args.image}")
                sys.exit(1)
            quick_test(args.image, logger)
        elif args.test_type == "backend":
            if not os.path.exists(args.image):
                print(f"Error: Image file not found: {args.image}")
                sys.exit(1)
            backend_comparison(args.image, args.upsampler, logger)
        elif args.test_type == "comprehensive":
            if not os.path.exists(args.image):
                print(f"Error: Image file not found: {args.image}")
                sys.exit(1)
            comprehensive_test(args.image, args.output, logger)
        elif args.test_type == "smoke":
            smoke_test()
    finally:
        # Save results if logger was used
        if logger:
            logger.save_results()

if __name__ == "__main__":
    main()
