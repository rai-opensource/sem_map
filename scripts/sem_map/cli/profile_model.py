# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sem_map.perception import (
    get_device,
)
from sem_map.perception.common import get_sem_map_root
from sem_map.perception.models.dinov import (
    load_dinovx_torch,
    compute_dinovx_feats_from_batch_torch,
    compute_dinovx_bilinear_upsampled_feats_from_batch,
    load_dinovx_onnx,
    compute_dinovx_feats_from_batch_onnx,
    load_dinovx_trt,
    compute_dinovx_feats_from_batch_trt,
)
from sem_map.perception.models.jafar import (
    load_jafar_torch,
    compute_jafar_upscaled_feats_from_batch_torch,
    load_jafar_onnx,
    compute_jafar_upscaled_feats_from_batch_onnx,
    load_jafar_trt,
    compute_jafar_upscaled_feats_from_batch_trt,
)
from sem_map.perception.models.segmentation import (
    load_seg_head_torch,
    load_seg_head_onnx,
    load_seg_head_trt,
    run_seg_from_batch_torch,
    run_seg_from_batch_onnx,
    run_seg_from_batch_trt,
    compute_confidence_from_reconstruction,
)
from sem_map.perception.models import resolve_artifact_path, get_num_classes, load_models_yaml, load_seg_layer_config


def load_model_config(model_name: str) -> dict:
    """Load model paths from cached models.yaml for specified model family."""
    models = load_models_yaml()
    families = models.get("families", {})
    if model_name not in families:
        raise ValueError(f"Model family '{model_name}' not found in config. Available: {list(families.keys())}")
    return families[model_name]


def run_backbone(image_batch, backend, model_torch, sess_onnx, runner_trt, latent_w, latent_h):
    """Run backbone to get lr_feats only."""
    with torch.inference_mode():
        if backend == "trt" and runner_trt is not None:
            lr_feats = compute_dinovx_feats_from_batch_trt(
                runner_trt,
                image_batch,
                latent_w,
                latent_h,
            )
        elif backend == "onnx" and sess_onnx is not None:
            lr_feats = compute_dinovx_feats_from_batch_onnx(
                sess_onnx,
                image_batch,
                latent_w,
                latent_h,
            )
        else:
            lr_feats = compute_dinovx_feats_from_batch_torch(
                image_batch=image_batch,
                model=model_torch,
                latent_w=latent_w,
                latent_h=latent_h,
            )
    return lr_feats


def run_upsampler(image_batch, lr_feats, mode, factor, jafar_model, jafar_sess_onnx, jafar_runner, patch_size, seg_h, seg_w):
    """Run upsampler based on mode:
    - bilinear: bilinearly upsample lr_feats to (seg_h, seg_w)
    - jafar_full: JAFAR upsamples patch_size× from lr_feats using the full image (scale=1)
    - jafar_partial: image downsampled by patch_size/factor, JAFAR upsamples factor× from lr_feats,
                     then bilinear to (seg_h, seg_w)
    """
    if mode == "bilinear":
        lr_h, lr_w = int(lr_feats.shape[2]), int(lr_feats.shape[3])
        f = max(1, int(factor))
        feats = compute_dinovx_bilinear_upsampled_feats_from_batch(lr_feats, lr_h * f, lr_w * f)
        if feats.shape[2] != seg_h or feats.shape[3] != seg_w:
            feats = torch.nn.functional.interpolate(
                feats, size=(seg_h, seg_w), mode="bilinear", align_corners=False,
            )
        return feats

    # jafar_full uses upsample_factor=patch_size → scale=1 (full image fed to JAFAR)
    # jafar_partial uses upsample_factor=factor → scale=patch_size/factor (image downsampled first)
    H, W = int(image_batch.shape[2]), int(image_batch.shape[3])
    jafar_upsample_factor = patch_size if mode == "jafar_full" else factor
    scale = max(1, int(round(float(patch_size) / float(max(1, jafar_upsample_factor)))))
    if scale > 1:
        image_batch_ds = torch.nn.functional.interpolate(
            image_batch,
            size=(max(1, H // scale), max(1, W // scale)),
            mode="bilinear",
            align_corners=False,
        )
    else:
        image_batch_ds = image_batch

    # Run JAFAR using the same lr_feats from backbone
    with torch.inference_mode():
        if jafar_runner is not None:
            j_hr_feats_batched = compute_jafar_upscaled_feats_from_batch_trt(
                jafar_runner, image_batch_ds, lr_feats
            )
        elif jafar_sess_onnx is not None:
            j_hr_feats_batched = compute_jafar_upscaled_feats_from_batch_onnx(
                jafar_sess_onnx, image_batch_ds, lr_feats
            )
        else:
            j_hr_feats_batched = compute_jafar_upscaled_feats_from_batch_torch(
                image_batch=image_batch_ds,
                model=jafar_model,
                lr_feats=lr_feats,
            )

    # Bilinear to target seg resolution if needed.
    # jafar_full attends to the full-res image internally but output features are brought
    # to (seg_h, seg_w) here. jafar_partial similarly adjusts from its native factor output.
    if j_hr_feats_batched.shape[2] != seg_h or j_hr_feats_batched.shape[3] != seg_w:
        j_hr_feats_batched = torch.nn.functional.interpolate(
            j_hr_feats_batched,
            size=(seg_h, seg_w),
            mode="bilinear",
            align_corners=False,
        )

    return j_hr_feats_batched


def run_segmentation(features, backend, seg_head_torch, seg_sess_onnx, seg_runner_trt, temperature):
    """Run segmentation on features. Returns (seg, logits, reconstruction)."""
    reconstruction = None
    features = features.unsqueeze(0) if features.ndim == 3 else features
    if backend == "trt" and seg_runner_trt is not None:
        result = run_seg_from_batch_trt(seg_runner_trt, features, temperature, return_reconstruction=True)
        # Returns (seg, logits, reconstruction) when return_reconstruction=True without probs
        seg, logits, reconstruction = result if len(result) == 3 else (*result, None)
    elif backend == "onnx" and seg_sess_onnx is not None:
        # ONNX backend does not support reconstruction output
        seg, logits = run_seg_from_batch_onnx(seg_sess_onnx, features, temperature)
    else:
        # Ensure dtype compatibility between features and PyTorch head (e.g., fp16 features vs fp32 weights)
        if isinstance(features, torch.Tensor) and seg_head_torch is not None:
            try:
                target_dtype = next(seg_head_torch.parameters()).dtype
                if features.dtype != target_dtype:
                    features = features.to(dtype=target_dtype)
            except StopIteration:
                # No parameters in head; skip dtype alignment
                pass
        with torch.inference_mode():
            raw = seg_head_torch(features)
            if isinstance(raw, tuple):
                logits, reconstruction = raw[0], raw[1]
            else:
                logits, reconstruction = raw, None
            seg = logits.argmax(dim=1)
    return seg, logits, reconstruction


def profile_modular_pipeline(
    model_config: dict,
    backbone_backend: str,
    upsampler_mode: str,
    upsampler_backend: str,
    seg_backend: str,
    upsample_partial_factor: int,
    latent_w: int,
    latent_h: int,
    device: torch.device,
    warmup: int,
    iters: int,
    skip_seg: bool,
    seg_layer: str | None = None,
    seg_layer_config: str | None = None,
    model_family: str = "dinov3s",
    seg_h: int | None = None,
    seg_w: int | None = None,
):
    """Profile the modular pipeline with configurable backends and upsampler modes."""

    # Extract model parameters from config (nested under backbone)
    patch_size = model_config["backbone"]["patch_size"]
    feature_dim = model_config["backbone"]["feat_dims"]

    # Get model paths from config
    if backbone_backend == "torch":
        backbone_path = None
    else:
        # Use resolver to build absolute path for ONNX/TRT backbones
        artifact_type = "onnx" if backbone_backend == "onnx" else "trt"
        backbone_path = resolve_artifact_path(model_family, kind="backbone", artifact_type=artifact_type)

    # Resolve JAFAR artifact path when needed
    jafar_path = None
    if upsampler_mode in ["jafar_full", "jafar_partial"] and upsampler_backend in ("onnx", "trt"):
        factor = patch_size if upsampler_mode == "jafar_full" else int(upsample_partial_factor)
        jafar_path = resolve_artifact_path(
            model_family,
            kind="upsampler",
            artifact_type=("onnx" if upsampler_backend == "onnx" else "trt"),
            upsampler="jafar",
            upsample_factor=factor,
        )

    # Ensure JAFAR variables are defined for downstream calls
    jafar_model = None
    sess_jafar_onnx = None
    trt_jafar_runner = None

    if upsampler_mode in ["jafar_full", "jafar_partial"]:
        # Select engine per factor using new config layout (x2/x4/x8/x16)
        if upsampler_backend == "torch":
            # Load JAFAR torch variant specified under upsamplers.jafar
            jafar_model, _ = load_jafar_torch(model_config["upsamplers"]["jafar"]["torch_variant"])
            # Ensure Torch JAFAR runs in FP16 on CUDA
            if get_device().type == "cuda":
                try:
                    jafar_model.half()
                except Exception:
                    pass
        elif upsampler_backend == "onnx":
            if not jafar_path or not jafar_path.exists():
                raise FileNotFoundError(f"JAFAR ONNX model not found: {jafar_path}")
            sess_jafar_onnx = load_jafar_onnx(str(jafar_path))
        elif upsampler_backend == "trt":
            if not jafar_path or not jafar_path.exists():
                raise FileNotFoundError(f"JAFAR TensorRT engine not found: {jafar_path}")
            trt_jafar_runner = load_jafar_trt(str(jafar_path))

    # Warn and auto-skip seg if no seg_layer provided
    if not skip_seg and not seg_layer:
        print("Warning: --skip_seg not set but --seg_layer not provided; skipping segmentation stage.")
        skip_seg = True

    # Resolve num_classes for seg head
    num_classes = None
    if not skip_seg:
        if seg_layer_config:
            config_dir = get_sem_map_root() / "config" / "seg_layers"
            config_path = Path(seg_layer_config)
            if not config_path.is_absolute():
                config_name = seg_layer_config if seg_layer_config.endswith((".yaml", ".yml")) else f"{seg_layer_config}.yaml"
                config_path = config_dir / config_name
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Seg layer config not found: {config_path}\n"
                    f"Expected location: {config_dir}/ or provide a full path"
                )
            seg_config = load_seg_layer_config(config_path.name, config_path.parent)
            num_classes = seg_config.get("num_classes")
            if num_classes is None:
                raise ValueError(f"Config file {config_path} does not contain 'num_classes' field")
        elif seg_layer:
            num_classes = get_num_classes(model_family, seg_layer)
            if num_classes is None:
                raise ValueError(
                    f"Could not determine num_classes for {model_family}/{seg_layer}. "
                    f"Either add it to models.yaml or provide --seg_layer_config"
                )

    # Resolve segmentation head path
    seg_path = None
    if not skip_seg and seg_layer:
        artifact_type = "torch" if seg_backend == "torch" else ("onnx" if seg_backend == "onnx" else "trt")
        seg_path = resolve_artifact_path(model_family, kind="seg_layer", artifact_type=artifact_type, seg_layer=seg_layer)

    # Load backbone model
    dino_model = None
    sess_backbone_onnx = None
    trt_back_runner = None
    if backbone_backend == "torch":
        dino_model, _ = load_dinovx_torch(model_config["backbone"]["torch_variant"])
        # Ensure Torch backbone runs in FP16 on CUDA
        if get_device().type == "cuda":
            try:
                dino_model.half()
            except Exception:
                pass
    elif backbone_backend == "onnx":
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone ONNX model not found: {backbone_path}")
        sess_backbone_onnx = load_dinovx_onnx(str(backbone_path))
    elif backbone_backend == "trt":
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone TensorRT engine not found: {backbone_path}")
        trt_back_runner = load_dinovx_trt(str(backbone_path))

    # Load segmentation head
    seg_head = None
    seg_sess_onnx = None
    seg_trt_runner = None
    if not skip_seg and seg_path:
        if seg_backend == "torch":
            if not seg_path.exists():
                raise FileNotFoundError(f"Segmentation head checkpoint not found: {seg_path}")
            seg_head = load_seg_head_torch(
                model_feature_dim=feature_dim,
                num_classes=int(num_classes),
                model_name=model_config.get("model_name", "dinov3s"),
                checkpoint_dir=str(seg_path)
            )
        elif seg_backend == "onnx":
            if not seg_path.exists():
                raise FileNotFoundError(f"Segmentation head ONNX model not found: {seg_path}")
            seg_sess_onnx = load_seg_head_onnx(str(seg_path))
        elif seg_backend == "trt":
            if not seg_path.exists():
                raise FileNotFoundError(f"Segmentation head TensorRT engine not found: {seg_path}")
            seg_trt_runner = load_seg_head_trt(str(seg_path))

    # Snapshot free VRAM before any allocations (captures TRT + PyTorch + everything)
    vram_free_before = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        vram_free_before, _ = torch.cuda.mem_get_info(device)

    # Build dummy input tensor (no real image needed for latency profiling)
    input_h = latent_h * patch_size
    input_w = latent_w * patch_size
    # seg_h/seg_w control the upsampler output resolution (and therefore the seg head resolution)
    # defaulting to full res (input_h, input_w) for backward compatibility
    seg_h = seg_h if seg_h is not None else input_h
    seg_w = seg_w if seg_w is not None else input_w
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    image_batch = torch.randn(1, 3, input_h, input_w, dtype=dtype, device=device)

    def _run_pipeline():
        lr_feats = run_backbone(
            image_batch=image_batch,
            backend=backbone_backend,
            model_torch=dino_model,
            sess_onnx=sess_backbone_onnx,
            runner_trt=trt_back_runner,
            latent_w=latent_w,
            latent_h=latent_h,
        )
        upsampled_feats = run_upsampler(
            image_batch=image_batch,
            lr_feats=lr_feats,
            mode=upsampler_mode,
            factor=upsample_partial_factor,
            jafar_model=jafar_model,
            jafar_sess_onnx=sess_jafar_onnx,
            jafar_runner=trt_jafar_runner,
            patch_size=patch_size,
            seg_h=seg_h,
            seg_w=seg_w,
        )
        seg = None
        if not skip_seg:
            seg, _, reconstruction = run_segmentation(
                features=upsampled_feats,
                backend=seg_backend,
                seg_head_torch=seg_head,
                seg_sess_onnx=seg_sess_onnx,
                seg_runner_trt=seg_trt_runner,
                temperature=None,
            )
            if reconstruction is not None:
                compute_confidence_from_reconstruction(
                    upsampled_feats, reconstruction, method="cosine", normalize_range=(-1.0, 1.0)
                )
        return lr_feats, upsampled_feats, seg

    times = []
    for wi in range(warmup):
        lr_feats, upsampled_feats, seg = _run_pipeline()
        if wi == 0:
            if not skip_seg:
                print(f"[{backbone_backend}_{upsampler_mode}_{seg_backend}] input_batch_shape={tuple(image_batch.shape)} lr_feats_shape={tuple(lr_feats.shape)} upsampled_feats_shape={tuple(upsampled_feats.shape)} seg_shape={tuple(seg.shape)}")
            else:
                print(f"[{backbone_backend}_{upsampler_mode}_no_seg] input_batch_shape={tuple(image_batch.shape)} lr_feats_shape={tuple(lr_feats.shape)} upsampled_feats_shape={tuple(upsampled_feats.shape)}")
        torch.cuda.synchronize(device) if device.type == "cuda" else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(iters):
        start = time.perf_counter()
        _run_pipeline()
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        times.append((time.perf_counter() - start) * 1000.0)

    vram_used = None
    peak_mem = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        vram_free_after, _ = torch.cuda.mem_get_info(device)
        vram_used = vram_free_before - vram_free_after
        peak_mem = torch.cuda.max_memory_allocated(device)

    return times, vram_used, peak_mem


def main():
    parser = argparse.ArgumentParser(description="Profile modular segmentation pipeline with configurable backends")
    parser.add_argument("--model_name", type=str, default="dinov3s", help="Model variant to use")
    parser.add_argument("--backbone_backend", type=str, default="torch", choices=["torch", "onnx", "trt"], help="Backbone backend")
    parser.add_argument("--upsampler_mode", type=str, default="bilinear", choices=["bilinear", "jafar_full", "jafar_partial"], help="Upsampler strategy")
    parser.add_argument("--upsampler_backend", type=str, default="torch", choices=["torch", "onnx", "trt"], help="Backend for JAFAR upsampler (used only when upsampler_mode is jafar_*)")
    parser.add_argument("--seg_backend", type=str, default="torch", choices=["torch", "onnx", "trt"], help="Segmentation head backend")
    parser.add_argument("--skip_seg", action="store_true", help="Skip segmentation stage entirely")
    parser.add_argument("--seg_layer", type=str, default=None, help="Override segmentation layer key; loads artifacts/torch/<seg_layer>.pth when using torch backend")
    parser.add_argument("--seg_layer_config", type=str, default=None, help="Seg layer config filename (e.g. 'cityscapes_19') or full path to a YAML in config/seg_layers/. Provides num_classes; takes priority over models.yaml.")
    parser.add_argument("--upsample_partial_factor", type=int, default=4, help="Upsampling factor for partial upsampling (bilinear, JAFAR partial)")
    parser.add_argument("--latent_w", type=int, default=32)
    parser.add_argument("--latent_h", type=int, default=20)
    parser.add_argument("--seg_w", type=int, default=None, help="Width of upsampler output fed to seg head (default: latent_w * patch_size = full res)")
    parser.add_argument("--seg_h", type=int, default=None, help="Height of upsampler output fed to seg head (default: latent_h * patch_size = full res)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    device = get_device()

    # Load model configuration
    model_config = load_model_config(args.model_name)

    # Profile the modular pipeline
    times_ms, vram_used, peak_mem = profile_modular_pipeline(
        model_config=model_config,
        backbone_backend=args.backbone_backend,
        upsampler_mode=args.upsampler_mode,
        upsampler_backend=args.upsampler_backend,
        seg_backend=args.seg_backend,
        upsample_partial_factor=args.upsample_partial_factor,
        latent_w=args.latent_w,
        latent_h=args.latent_h,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        skip_seg=args.skip_seg,
        seg_layer=args.seg_layer,
        seg_layer_config=args.seg_layer_config,
        model_family=args.model_name,
        seg_h=args.seg_h,
        seg_w=args.seg_w,
    )

    # Calculate and print results
    n_iters = len(times_ms)
    avg_ms = sum(times_ms) / n_iters
    std_ms = statistics.stdev(times_ms) if n_iters > 1 else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    # Create descriptive mode string
    mode_parts = [args.backbone_backend, args.upsampler_mode, ("no_seg" if args.skip_seg else args.seg_backend)]
    if args.upsampler_mode in ("jafar_partial", "bilinear"):
        mode_parts.insert(2, f"f{args.upsample_partial_factor}")
    mode = "_".join(mode_parts)

    if device.type == "cuda" and vram_used is not None:
        vram_mb = vram_used / (1024**2)
        torch_mb = peak_mem / (1024**2) if peak_mem is not None else 0.0
        print(f"model={args.model_name} mode={mode} latent={args.latent_w}x{args.latent_h} device={device} avg_ms={avg_ms:.2f} std_ms={std_ms:.2f} n_iters={n_iters} fps={fps:.2f} vram_mb={vram_mb:.1f} torch_mb={torch_mb:.1f}")
    else:
        print(f"model={args.model_name} mode={mode} latent={args.latent_w}x{args.latent_h} device={device} avg_ms={avg_ms:.2f} std_ms={std_ms:.2f} n_iters={n_iters} fps={fps:.2f}")


if __name__ == "__main__":
    main()
