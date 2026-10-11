# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import onnx
import onnx.shape_inference
import onnx.checker
import onnxruntime as ort

from sem_map.perception import get_device
from sem_map.perception.models.dinov import (
    load_dinovx_torch,
    load_dinovx_onnx,
    compute_dinovx_feats_from_batch_onnx,
)
from sem_map.perception.models.jafar import (
    load_jafar_torch,
    compute_jafar_upscaled_feats_from_batch_onnx,
)
from sem_map.perception.models.segmentation import load_seg_head_torch
from sem_map.perception.models import resolve_artifact_path, get_num_classes


def build_dummy_input(latent_w: int, latent_h: int, patch_size: int, device: torch.device) -> torch.Tensor:
    # Construct directly without using prepare_image (expects PIL image)
    H = latent_h * patch_size
    W = latent_w * patch_size
    return torch.randn(1, 3, H, W, device=device)


def export_model_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    opset: int,
    dynamic: bool,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamo: bool = False,
    use_fp16: bool = True,
    sanitize_resize_antialias: bool = False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize inputs and names
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)  # type: ignore
    inputs_tuple: tuple[torch.Tensor, ...] = tuple(inputs)
    if input_names is None:
        input_names = ["pixel_values"] if len(inputs_tuple) == 1 else [f"input_{i}" for i in range(len(inputs_tuple))]
    if output_names is None:
        output_names = ["last_hidden_state"]

    # Only mark dimensions dynamic when requested (and not with dynamo)
    if dynamic:
        dynamic_axes: dict[str, dict[int, str]] = {}
        for name in input_names:
            # Always set batch dynamic
            dynamic_axes[name] = {0: "batch"}
            # Only make spatial dims dynamic for seg_head features, not for backbone/JAFAR pixel_values
            if name == "features":
                dynamic_axes[name].update({2: "height", 3: "width"})
        for name in output_names:
            # Always dynamic batch for outputs
            dynamic_axes[name] = {0: "batch"}
            # If logits or reconstruction are exported as [B, C, H, W], mark H/W dynamic
            if name in ("logits", "reconstruction"):
                dynamic_axes[name].update({2: "height", 3: "width"})
    else:
        dynamic_axes = {}

    model.eval()
    if use_fp16:
        model.half()
        inputs_tuple = tuple(t.half() for t in inputs_tuple)
    else:
        model.float()
        inputs_tuple = tuple(t.float() for t in inputs_tuple)
    def _do_export(opset_version: int, use_dynamo: bool):
        with torch.inference_mode():
            export_kwargs = dict(
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamo=use_dynamo,
            )
            if use_dynamo:
                # Keep weights inline: otherwise a <name>.onnx.data sidecar is
                # written and left behind, since the model is re-saved inline below.
                export_kwargs["external_data"] = False
            if use_dynamo and dynamic:
                # Use new dynamic_shapes API with torch.export.Dim symbols
                try:
                    from torch.export import Dim as _Dim  # PyTorch >=2.3
                except Exception:
                    _Dim = lambda name: name  # fallback to strings
                # dynamic_shapes must be a list/tuple matching inputs when using dynamo
                dyn_list: list[dict[int, object]] = []
                # Check if this is JAFAR (has lr_feats input) - only batch dynamic for pixel_values
                is_jafar = "lr_feats" in input_names
                for idx, name in enumerate(input_names):
                    if name == "pixel_values":
                        if is_jafar:
                            # JAFAR: only batch dynamic, spatial dims static
                            dyn_list.append({0: _Dim("batch")})
                        else:
                            # DINO backbone: batch, height, width dynamic
                            dyn_list.append({0: _Dim("batch"), 2: _Dim("height"), 3: _Dim("width")})
                    elif name == "features":
                        # Seg head: Dynamic height/width for 4D features
                        dyn_list.append({0: _Dim("batch"), 2: _Dim("height"), 3: _Dim("width")})
                    else:
                        # Other inputs (e.g., lr_feats): only batch dynamic
                        dyn_list.append({0: _Dim("batch")})
                if dyn_list:
                    export_kwargs["dynamic_shapes"] = tuple(dyn_list)
            if not use_dynamo and dynamic:
                export_kwargs["dynamic_axes"] = dynamic_axes
            torch.onnx.export(
                model,
                inputs_tuple,
                str(output_path),
                **export_kwargs,
            )

    try:
        _do_export(opset, dynamo)
    except Exception as e:
        raise e

    onnx_model = onnx.load(str(output_path))
    if sanitize_resize_antialias:
        # Disable antialias attribute on Resize nodes for TensorRT compatibility
        modified = False
        for node in onnx_model.graph.node:
            if node.op_type == "Resize":
                for attr in node.attribute:
                    if attr.name == "antialias" and getattr(attr, "i", 0) != 0:
                        attr.i = 0
                        modified = True
        # no special handling needed if not modified
    onnx.checker.check_model(onnx_model)
    inferred = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(inferred, str(output_path))


def quick_ort_check(onnx_path: Path, image_batch_cpu: torch.Tensor, providers: list[str] | None = None):
    if providers is None:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    # Match expected input dtype (e.g., FP16 exports)
    import numpy as np
    in_meta = sess.get_inputs()[0]
    want_fp16 = "float16" in getattr(in_meta, "type", "tensor(float)")
    arr = image_batch_cpu.numpy()
    if want_fp16 and arr.dtype != np.float16:
        arr = arr.astype(np.float16, copy=False)
    if (not want_fp16) and arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    inputs = {in_meta.name: arr}
    _ = sess.run(None, inputs)


def export_segmentation_head_to_onnx(
    feature_shape: tuple[int, int, int, int],
    num_classes: int,
    checkpoint_path: str | None,
    output_path: Path,
    opset: int = 19,
    dynamic: bool = True,
    use_fp16: bool = True,
):
    """Export the segmentation head (1x1 Conv) to ONNX.

    feature_shape: (batch, channels, height, width)
    num_classes: number of output classes for logits
    checkpoint_path: path to the .pth checkpoint (or directory); if None, uses default in load_segmentation_head
    output_path: ONNX file path
    """
    device = get_device()
    batch, channels, height, width = feature_shape

    head = load_seg_head_torch(
        model_feature_dim=channels,
        num_classes=num_classes,
        model_name="jafar",
        checkpoint_dir=checkpoint_path,
        device=device,
    )

    # Wrap head to produce [B, num_classes, H, W] and accept [B, C, H, W] inputs
    # Handle both old format (returns single tensor) and new format (returns tuple)
    class SegHead4D(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module
        def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            # Expect strictly 4D [B,C,H,W]
            assert x.dim() == 4, "features must be 4D [B,C,H,W]"
            output = self.module(x)
            # If module returns tuple (new format: predictions, reconstruction), return both
            if isinstance(output, tuple):
                return output  # Return both predictions and reconstruction
            return output  # Old format: single tensor

    wrapped = SegHead4D(head)

    # Export signature as [B, C, H, W] with dynamic H,W
    dummy_feats = torch.randn(batch, channels, height, width, device=device)

    # Check model dtype and match dummy_feats to it
    # The model might be in FP16 if it was loaded on CUDA
    try:
        model_dtype = next(head.parameters()).dtype
        dummy_feats = dummy_feats.to(dtype=model_dtype)
    except StopIteration:
        # Model has no parameters (shouldn't happen, but handle gracefully)
        pass

    # Determine output names based on model format
    # Check if head returns tuple by doing a test forward pass
    with torch.inference_mode():
        test_output = head(dummy_feats)
        if isinstance(test_output, tuple):
            output_names = ["logits", "reconstruction"]
        else:
            output_names = ["logits"]

    export_model_to_onnx(
        model=wrapped,
        output_path=Path(output_path),
        inputs=(dummy_feats,),
        opset=opset,
        dynamic=dynamic,
        input_names=["features"],
        output_names=output_names,
        dynamo=True,
        use_fp16=use_fp16,
        sanitize_resize_antialias=False,
    )


def _extract_torch_output(model: torch.nn.Module, image_batch: torch.Tensor, output_name: str):
    with torch.inference_mode():
        out = model(image_batch)
    # Handle HF ModelOutput or dict/tuple
    if hasattr(out, output_name):
        return getattr(out, output_name)
    if isinstance(out, dict) and output_name in out:
        return out[output_name]
    if isinstance(out, (list, tuple)) and len(out) > 0:
        return out[0]
    return out


def validate_parity(
    model: torch.nn.Module,
    onnx_path: Path,
    image_batch: torch.Tensor,
    output_name: str,
    use_cuda: bool,
):
    # Match devices for Torch and ORT (provider selection below)
    prev_device = next(model.parameters()).device if any(p is not None for p in model.parameters()) else torch.device("cpu")
    target_device = torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")

    model_dev = model
    try:
        model_dev = model.to(target_device)
    except Exception:
        pass
    model_dev.eval()
    with torch.inference_mode():
        torch_out = _extract_torch_output(model_dev, image_batch.to(target_device), output_name)
    torch_out_cpu = torch_out.detach().float().cpu()

    # ORT provider setup
    providers = ["CUDAExecutionProvider"] if target_device.type == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    # ORT expects numpy on host; cast to expected dtype
    import numpy as np
    in_meta = sess.get_inputs()[0]
    want_fp16 = "float16" in getattr(in_meta, "type", "tensor(float)")
    arr = image_batch.detach().cpu().numpy()
    if want_fp16 and arr.dtype != np.float16:
        arr = arr.astype(np.float16, copy=False)
    if (not want_fp16) and arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    inputs = {in_meta.name: arr}
    ort_out = sess.run([output_name], inputs)[0]

    # Flatten first element in batch (assumes batch>=1)
    torch_arr = torch_out_cpu.reshape(torch_out_cpu.shape[0], -1)[0].numpy()
    ort_arr = ort_out.reshape(ort_out.shape[0], -1)[0]

    # Robust metrics: cast to fp32 and ignore NaNs
    import numpy as np
    ta = np.nan_to_num(torch_arr.astype(np.float32))
    oa = np.nan_to_num(ort_arr.astype(np.float32))

    dot = float(np.dot(ta, oa))
    denom = float(np.linalg.norm(ta) * np.linalg.norm(oa))
    cos_sim = dot / denom if denom > 0 else 1.0

    diff = ta - oa
    rmse = float(np.sqrt(np.mean(diff * diff)))

    # Attempt to move model back to its previous device
    try:
        model.to(prev_device)
    except Exception:
        pass
    return cos_sim, rmse


def _build_jafar_wrapper(model_key: str):
    if model_key == "jafar_dinov2s":
        upsampler, _backbone = load_jafar_torch(backbone_name="vit_small_patch14_reg4_dinov2")
        default_ps = 14
    else:  # jafar_dinov3s
        upsampler, _backbone = load_jafar_torch(backbone_name="vit_small_patch16_dinov3.lvd1689m")
        default_ps = 16

    class JafarUpsamplerOnly(nn.Module):
        def __init__(self, upsampler: nn.Module):
            super().__init__()
            self.upsampler = upsampler

        def forward(self, pixel_values, lr_feats):
            H = pixel_values.shape[2]
            W = pixel_values.shape[3]
            return self.upsampler(pixel_values, lr_feats, (H, W))

    return JafarUpsamplerOnly(upsampler), default_ps, "hr_feats"


def _compute_export_settings(model_key: str, args):
    opset_to_use = args.opset
    # Enable dynamic batch for backbones and JAFAR by default; seg_head uses args.dynamic
    if model_key in ("jafar_dinov2s", "jafar_dinov3s", "dinov2s", "dinov3s"):
        dynamic_flag = True  # Always enable dynamic batch for DINO and JAFAR
        export_dynamo = True
    else:
        dynamic_flag = args.dynamic
        export_dynamo = True
    return dynamic_flag, export_dynamo, opset_to_use


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX (dinov2s, dinov3s, jafar_dinov2s, jafar_dino)")
    parser.add_argument("--model", choices=["dinov2s", "dinov3s", "jafar_dinov2s", "jafar_dinov3s", "seg_head"], required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--latent_w", type=int, default=32)
    parser.add_argument("--latent_h", type=int, default=20)
    parser.add_argument("--patch_size", type=int, help="Override patch size; default 14 for v2, 16 for v3")
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic shapes (uses torch.onnx dynamo exporter)")
    parser.add_argument("--no_check", action="store_true", help="Skip ONNXRuntime smoke check")
    parser.add_argument("--validate", action="store_true", help="Validate parity between PyTorch and ONNX outputs")
    parser.add_argument("--ort_cuda", action="store_true", help="Use ONNX Runtime CUDA provider and run Torch validation on CUDA if available")
    parser.add_argument("--backbone_onnx", type=str, help="Path to DINO ONNX backbone (dinov2s/dinov3s) for JAFAR parity validation")
    parser.add_argument(
        "--jafar_upsample_factor",
        type=int,
        help="For JAFAR exports, intended upsample factor (e.g., 16 default, 4 for x4). pixel_values will be downsampled by round(patch_size/factor).",
    )

    # Seg head specific
    parser.add_argument("--family", type=str, default="dinov3s", help="Family name (dinov3s/dinov2s) for seg_head resolution")
    parser.add_argument("--seg_classes", type=int, default=None, help="Number of segmentation classes for seg_head export (defaults from models.yaml if omitted)")
    parser.add_argument("--seg_checkpoint", type=str, help="Path or directory to seg head checkpoint (.pth). If omitted, resolved via models.yaml + --seg_layer")
    parser.add_argument("--seg_layer", type=str, default=None, help="Segmentation layer key; used to resolve checkpoint and classes via models.yaml when seg_head")
    # New flags
    parser.add_argument("--min_cos_sim", type=float, default=0.9990)
    parser.add_argument("--max_rmse", type=float, default=1e-3)

    args = parser.parse_args()

    device = get_device()

    if args.model == "seg_head":
        # Export the segmentation head only using dummy feature tensor
        feature_c = 384
        feature_shape = (1, feature_c, args.latent_h, args.latent_w)
        # Resolve checkpoint and classes
        if args.seg_checkpoint:
            seg_ckpt_path = args.seg_checkpoint
            num_classes = int(args.seg_classes or 27)
        else:
            if not args.seg_layer:
                raise ValueError("--seg_layer is required when --seg_checkpoint is not provided")
            seg_ckpt_path = str(resolve_artifact_path(args.family, kind="seg_layer", artifact_type="torch", seg_layer=args.seg_layer).resolve())
            num_classes = int(args.seg_classes or (get_num_classes(args.family, args.seg_layer) or 27))
        export_segmentation_head_to_onnx(
            feature_shape=feature_shape,
            num_classes=num_classes,
            checkpoint_path=seg_ckpt_path,
            output_path=Path(args.output),
            opset=args.opset,
            dynamic=True,
            use_fp16=True,
        )
        print(f"exported={args.output} model=seg_head features={feature_shape} classes={num_classes} opset={args.opset} dynamic=True")
        return

    if args.model in ("dinov3s", "dinov2s"):
        model, default_ps = load_dinovx_torch(args.model)
        output_name = "last_hidden_state"
    else:
        model, default_ps, output_name = _build_jafar_wrapper(args.model)

        # Force SDPA to math path for export stability
        try:
            if hasattr(torch.backends, "cuda"):
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    patch_size = args.patch_size or default_ps

    # Dummy input sizes
    H_full = args.latent_h * patch_size
    W_full = args.latent_w * patch_size
    # pixel_values tensor used for export/tracing
    if args.model.startswith("jafar_"):
        # If factor not provided, assume default equals patch_size (no downsample)
        factor = int(args.jafar_upsample_factor) if args.jafar_upsample_factor else int(patch_size)
        scale = max(1, int(round(float(patch_size) / float(max(1, factor)))))
        if scale > 1:
            H_ds = max(1, H_full // scale)
            W_ds = max(1, W_full // scale)
        else:
            H_ds, W_ds = H_full, W_full
        image_batch_full = torch.randn(1, 3, H_full, W_full, device=device)
        image_batch = torch.randn(1, 3, H_ds, W_ds, device=device)
    else:
        image_batch = torch.randn(1, 3, H_full, W_full, device=device)
        image_batch_full = image_batch
    # image_batch already set above (and image_batch_full for JAFAR)


    dynamic_flag, export_dynamo, opset_to_use = _compute_export_settings(args.model, args)

    if args.model.startswith("jafar_"):
        # Export upsampler-only wrapper with two inputs
        latent_h = args.latent_h
        latent_w = args.latent_w
        lr_feats_dummy = torch.randn(1, 384, latent_h, latent_w, device=device)
        export_model_to_onnx(
            model=model,
            output_path=Path(args.output),
            inputs=(image_batch, lr_feats_dummy),
            opset=opset_to_use,
            dynamic=dynamic_flag,  # Use dynamic_flag to enable flexible batch size
            input_names=["pixel_values", "lr_feats"],
            output_names=["hr_feats"],
            dynamo=True,
            use_fp16=(args.model in ("jafar_dinov2s", "jafar_dinov3s")),
            sanitize_resize_antialias=False,
        )
    else:
        export_model_to_onnx(
            model=model,
            output_path=Path(args.output),
            inputs=(image_batch,),
            opset=opset_to_use,
            dynamic=dynamic_flag,
            input_names=["pixel_values"],
            output_names=[output_name],
            dynamo=True,
            use_fp16=(args.model in ("dinov2s", "dinov3s")),
            sanitize_resize_antialias=(args.model == "dinov2s"),
        )

    # Optional ORT check on CPU
    if not args.no_check:
        image_cpu = image_batch.detach().cpu()
        providers = ["CUDAExecutionProvider"] if args.ort_cuda and ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        if args.model.startswith("jafar_"):
            # Two-input smoke test
            sess = ort.InferenceSession(str(args.output), providers=providers)
            in0 = sess.get_inputs()[0].name
            in1 = sess.get_inputs()[1].name
            import numpy as np
            want_fp16 = any("float16" in getattr(i, "type", "") for i in sess.get_inputs())
            latent_h = args.latent_h
            latent_w = args.latent_w
            lr_feats_np = torch.randn(1, 384, latent_h, latent_w).numpy()
            arr0 = image_cpu.numpy()
            if want_fp16:
                arr0 = arr0.astype(np.float16, copy=False)
                lr_feats_np = lr_feats_np.astype(np.float16, copy=False)
            else:
                arr0 = arr0.astype(np.float32, copy=False)
                lr_feats_np = lr_feats_np.astype(np.float32, copy=False)
            _ = sess.run(None, {in0: arr0, in1: lr_feats_np})
        else:
            quick_ort_check(Path(args.output), image_cpu, providers=providers)

        if args.validate:
            if args.model.startswith("jafar_"):
                # Parity for JAFAR upsampler-only using ONNX DINO backbone
                if not args.backbone_onnx:
                    print("warning: --backbone_onnx not provided; skipping JAFAR parity check")
                else:
                    # Build DINO lr_feats via ONNX, then compare upsampler torch vs onnx upsampler
                    providers = ["CUDAExecutionProvider"] if args.ort_cuda and ort.get_device() == "GPU" else ["CPUExecutionProvider"]
                    sess = load_dinovx_onnx(args.backbone_onnx)
                    # Compute ONNX DINO lr/hr (use lr) using full-size pixel_values
                    latent_w = args.latent_w
                    latent_h = args.latent_h
                    device_torch = image_batch_full.device
                    lr_feats = compute_dinovx_feats_from_batch_onnx(
                        sess,
                        image_batch_full.to(device_torch),
                        latent_w,
                        latent_h,
                        int(image_batch_full.shape[2]),
                        int(image_batch_full.shape[3]),
                    )
                    # Torch upsampler output in FP16 to match ONNX fp16 graph
                    model_dev = model.to(device_torch).half()
                    model_dev.eval()
                    with torch.inference_mode():
                        torch_hr = model_dev(image_batch.to(device_torch).half(), lr_feats.half())[0]
                    # ONNX upsampler output via helper
                    sess_u = ort.InferenceSession(str(Path(args.output)), providers=providers)
                    ort_out_t = compute_jafar_upscaled_feats_from_batch_onnx(
                        sess_u,
                        image_batch.to(device_torch),
                        lr_feats,
                    )
                    ort_out = ort_out_t.detach().cpu().numpy()
                    # Flatten fully for both
                    torch_arr = torch_hr.detach().float().cpu().reshape(-1).numpy()
                    ort_arr = ort_out.reshape(-1)
                    import numpy as np
                    ta = np.nan_to_num(torch_arr.astype(np.float32))
                    oa = np.nan_to_num(ort_arr.astype(np.float32))
                    dot = float(np.dot(ta, oa))
                    denom = float(np.linalg.norm(ta) * np.linalg.norm(oa))
                    cos_sim = dot / denom if denom > 0 else 1.0
                    diff = ta - oa
                    rmse = float(np.sqrt(np.mean(diff * diff)))
                print(f"parity (upsampler) cos_sim={cos_sim:.6f} rmse={rmse:.6e} provider={'CUDA' if (args.ort_cuda and ort.get_device() == 'GPU') else 'CPU'}")
                if cos_sim < args.min_cos_sim or rmse > args.max_rmse:
                    raise RuntimeError(f"Validation failed (upsampler): cos={cos_sim:.6f}, rmse={rmse:.3e}")
            else:
                cos_sim, rmse = validate_parity(
                    model,
                    Path(args.output),
                    image_batch,
                    output_name,
                    use_cuda=bool(args.ort_cuda and torch.cuda.is_available()),
                )
                print(f"parity cos_sim={cos_sim:.6f} rmse={rmse:.6e} provider={'CUDA' if (args.ort_cuda and torch.cuda.is_available()) else 'CPU'}")
                if cos_sim < args.min_cos_sim or rmse > args.max_rmse:
                    raise RuntimeError(f"Validation failed: cos={cos_sim:.6f}, rmse={rmse:.3e}")

    print(f"exported={args.output} model={args.model} latent={args.latent_w}x{args.latent_h} patch={patch_size} opset={args.opset} dynamic={args.dynamic}")


if __name__ == "__main__":
    main()
