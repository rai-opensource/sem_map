# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from sem_map.perception.common import get_sem_map_root
from sem_map.perception.models import load_models_yaml, resolve_artifact_path


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _maybe(path: Path | None) -> Path | None:
    return path if (path is not None and path.exists()) else None


def main():
    p = argparse.ArgumentParser(description="Inspect ONNX and build TRT engines from config models.yaml (families)")
    p.add_argument("--families", nargs="*", help="Families to process; defaults to all in config")
    p.add_argument("--workspace_gb", type=float, default=4.0, help="TRT builder workspace in GB")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 if supported")
    p.add_argument("--only-seg-layers", action="store_true", help="Only build TRT engines for segmentation layers, skip backbone and JAFAR variants")
    p.add_argument("--max-batch-size", type=int, default=8, help="Maximum batch size for dynamic builds (default: 8)")
    args = p.parse_args()

    sem_map_root = get_sem_map_root()
    cfg_path = sem_map_root / "config" / "models.yaml"
    inspect_path = sem_map_root / "scripts"/ "sem_map" / "cli" / "onnx_inspect.py"
    trt_builder = sem_map_root / "scripts"/ "sem_map" / "cli" / "build_trt_engine.py"

    cfg = load_models_yaml(cfg_path)

    families = args.families if args.families else list(cfg.get("families", {}).keys())

    # Collect ONNX and TRT paths from config
    tasks: list[tuple[str, Path, Path]] = []
    for family in families:
        fam_cfg = cfg.get("families", {}).get(family)
        if not fam_cfg:
            print(f"Skipping unknown family '{family}'")
            continue
        if not args.only_seg_layers:
            # backbone
            onnx_bb = resolve_artifact_path(family, kind="backbone", artifact_type="onnx")
            trt_bb = resolve_artifact_path(family, kind="backbone", artifact_type="trt")
            onnx_p = _maybe(onnx_bb.resolve() if onnx_bb else None)
            if onnx_p and trt_bb:
                engine_p = trt_bb.resolve()
                if engine_p.suffix != ".plan":
                    engine_p = engine_p.parent / (engine_p.name + ".plan")
                tasks.append((f"{family}_backbone", onnx_p, engine_p))
            # jafar variants
            for factor in (2, 4, 8, 16):
                onnx_j = resolve_artifact_path(family, kind="upsampler", artifact_type="onnx", upsampler="jafar", upsample_factor=factor)
                trt_j = resolve_artifact_path(family, kind="upsampler", artifact_type="trt", upsampler="jafar", upsample_factor=factor)
                onnx_p = _maybe(onnx_j.resolve() if onnx_j else None)
                if onnx_p and trt_j:
                    engine_p = trt_j.resolve()
                    if engine_p.suffix != ".plan":
                        engine_p = engine_p.parent / (engine_p.name + ".plan")
                    tasks.append((f"{family}_jafar_x{factor}", onnx_p, engine_p))
        # seg heads (all seg_layers)
        seg_layers = fam_cfg.get("seg_layers", {}) or {}
        for seg_key in seg_layers.keys():
            onnx_s = resolve_artifact_path(family, kind="seg_layer", artifact_type="onnx", seg_layer=seg_key)
            trt_s = resolve_artifact_path(family, kind="seg_layer", artifact_type="trt", seg_layer=seg_key)
            onnx_p = _maybe(onnx_s.resolve() if onnx_s else None)
            if onnx_p and trt_s:
                engine_p = trt_s.resolve()
                if engine_p.suffix != ".plan":
                    engine_p = engine_p.parent / (engine_p.name + ".plan")
                tasks.append((f"{family}_seg_{seg_key}", onnx_p, engine_p))

    if not tasks:
        # Exit non-zero. Silently "succeeding" here is worse than useless: the
        # user is told nothing is wrong, then the node fails much later with an
        # engine it cannot load, and nothing connects the two.
        onnx_dir = sem_map_root / cfg.get("onnx_dir", "artifacts/onnx")
        print(
            f"ERROR: no ONNX artifacts found under {onnx_dir}, so no TensorRT "
            f"engines were built.\n"
            f"Fetch the model files first, or export them yourself with "
            f"cli/run_export_all_onnx.py. See the README section on models.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Inspect and build TRT
    for name, onnx_path, engine_path in tasks:
        print(f"Inspecting {name} -> {onnx_path}")
        rc, out_s, err_s = _run(["python3", str(inspect_path), "--model", str(onnx_path)])
        if rc != 0:
            print(f"inspect_failed name={name} err=\n{err_s}")
            continue
        print(out_s, end="")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python3", str(trt_builder),
            "--onnx", str(onnx_path),
            "--engine", str(engine_path),
            "--workspace_gb", str(args.workspace_gb),
        ]
        # Add dynamic batch shapes for DINO and JAFAR (spatial dims are static)
        if name.endswith("_backbone"):
            # Backbone: pixel_values input [B, 3, H, W]
            # Spatial dims from export: latent_h=20, latent_w=32, patch_size=16 -> 320x512
            cmd.extend([
                "--input_name", "pixel_values",
                "--min_shape", f"1,3,320,512",
                "--opt_shape", f"{args.max_batch_size},3,320,512",
                "--max_shape", f"{args.max_batch_size},3,320,512",
            ])
        elif (name.endswith("_seg")) or ("_seg_" in name):
            # Seg head: features input [B, C, H, W] - keep existing dynamic spatial dims
            cmd.extend([
                "--input_name", "features",
                "--min_shape", "1,384,1,1",
                "--opt_shape", "1,384,320,512",
                "--max_shape", "1,384,320,512",
            ])
        if args.fp16:
            cmd.append("--fp16")
        print("$", " ".join(cmd))
        rc, out_s, err_s = _run(cmd)
        if rc != 0:
            print(f"trt_build_failed name={name} rc={rc} err=\n{err_s}")
            continue
        print(out_s, end="")
        if engine_path.exists():
            try:
                print(f"  done: {engine_path.name} size={engine_path.stat().st_size} bytes")
            except Exception:
                pass


if __name__ == "__main__":
    main()
