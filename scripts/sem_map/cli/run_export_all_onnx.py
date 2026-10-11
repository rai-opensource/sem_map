# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import subprocess
from pathlib import Path

import yaml
from sem_map.perception.common import get_sem_map_root, get_workspace_root
from sem_map.perception.models import resolve_artifact_path, get_num_classes, load_models_yaml


def run(cmd: list[str]) -> bool:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  WARNING: command failed (exit {result.returncode}), skipping.")
        return False
    return True


def main():
    p = argparse.ArgumentParser(description="Export all ONNX artifacts for configured models (families)")
    p.add_argument("--outdir", type=str, help="Base output directory", default=f"{get_sem_map_root()}/artifacts/onnx")
    p.add_argument("--families", nargs="*", default=["dinov3s"], help="Model families to export")
    p.add_argument("--latent_w", type=int, default=32)
    p.add_argument("--latent_h", type=int, default=20)
    p.add_argument("--opset", type=int, default=19)
    p.add_argument("--only-seg-layers", action="store_true", help="Only export segmentation layers, skip backbone and JAFAR variants")
    args = p.parse_args()

    sem_map_root = get_sem_map_root()
    repo_root = get_workspace_root()
    cfg_path = sem_map_root / "config" / "models.yaml"
    script_path = sem_map_root / "scripts" / "sem_map" / "cli" / "export_to_onnx.py"
    outdir = (repo_root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    models_cfg = load_models_yaml(cfg_path)

    for family in args.families:
        print(f"Exporting family={family}")
        fam = models_cfg.get("families", {}).get(family)
        if not fam:
            print(f"Skipping unknown family '{family}'")
            continue
        # Use provided latent size args; no longer infer from config
        fam_latent_w, fam_latent_h = args.latent_w, args.latent_h

        if not args.only_seg_layers:
            # Backbone export
            bb_out = outdir / f"{family}_backbone.onnx"
            print(f"Exporting backbone {family} -> {bb_out}")
            run([
                "python", str(script_path),
                "--model", family,
                "--output", str(bb_out),
                "--latent_w", str(fam_latent_w),
                "--latent_h", str(fam_latent_h),
                "--opset", str(args.opset),
                "--dynamic",  # Enable dynamic batch for flexible training
            ])
            try:
                print(f"  done: {bb_out.name} size={bb_out.stat().st_size} bytes")
            except Exception:
                pass

            # JAFAR variants
            for factor in (2, 4, 8, 16):
                jf_out = outdir / f"{family}_jafar_x{factor}.onnx"
                print(f"Exporting jafar_{family} x{factor} -> {jf_out}")
                run([
                    "python", str(script_path),
                    "--model", f"jafar_{family}",
                    "--output", str(jf_out),
                    "--latent_w", str(fam_latent_w),
                    "--latent_h", str(fam_latent_h),
                    "--jafar_upsample_factor", str(factor),
                    "--dynamic",  # Enable dynamic batch for flexible training
                ])
                try:
                    print(f"  done: {jf_out.name} size={jf_out.stat().st_size} bytes")
                except Exception:
                    pass

        # Segmentation heads per family (all seg_layers)
        seg_layers = fam.get("seg_layers") or {}
        for seg_key in seg_layers.keys():
            ckpt_path = resolve_artifact_path(family, kind="seg_layer", artifact_type="torch", seg_layer=seg_key)
            onnx_path = resolve_artifact_path(family, kind="seg_layer", artifact_type="onnx", seg_layer=seg_key)
            onnx_abs = onnx_path.resolve()
            onnx_abs.parent.mkdir(parents=True, exist_ok=True)
            num_classes = int(get_num_classes(family, seg_key) or 27)
            print(f"Exporting seg_head {family}/{seg_key} -> {onnx_abs}")
            run([
                "python", str(script_path),
                "--model", "seg_head",
                "--output", str(onnx_abs),
                "--latent_w", str(fam_latent_w),
                "--latent_h", str(fam_latent_h),
                "--seg_classes", str(num_classes),
                "--dynamic",
                "--seg_checkpoint", str(ckpt_path.resolve()),
            ])
            try:
                print(f"  done: {onnx_abs.name} size={onnx_abs.stat().st_size} bytes")
            except Exception:
                pass


if __name__ == "__main__":
    main()
