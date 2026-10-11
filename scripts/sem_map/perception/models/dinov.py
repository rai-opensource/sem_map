# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import torch
from PIL import Image
from transformers import AutoModel

from ..common import prepare_image
from ..backends.pytorch import load_model_from_hf, run_vision_transformer, tokens_to_spatial_features
from ..backends.tensorrt import TrtRunner


# DINOv2/v3 variant configurations
DINOV_CONFIGS = {
    "dinov2s": {
        "patch_size": 14,
        "skip_tokens": 5,
        "feat_dims": 384,
        "hf_id": "facebook/dinov2-with-registers-small"
    },
    "dinov3s": {
        "patch_size": 16,
        "skip_tokens": 5,
        "feat_dims": 384,
        "hf_id": "facebook/dinov3-vits16-pretrain-lvd1689m"
    },
    "dinov3b": {
        "patch_size": 16,
        "skip_tokens": 5,
        "feat_dims": 768,
        "hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m"
    }
}


def load_dinovx_torch(variant: str) -> tuple[AutoModel, int]:
    """Load DINOv2/v3 PyTorch model.

    Args:
        variant: One of {"dinov2s", "dinov3s", "dinov3b"}

    Returns:
        (model, patch_size)
    """
    if variant not in DINOV_CONFIGS:
        raise ValueError(f"Unsupported DINO variant: {variant}")

    config = DINOV_CONFIGS[variant]
    model = load_model_from_hf(config["hf_id"])
    return model, config["patch_size"]


def load_dinovx_onnx(onnx_path: str):
    """Load DINOv ONNX session."""
    from ..backends.onnx import create_session
    return create_session(onnx_path)


def load_dinovx_trt(engine_path: str) -> TrtRunner:
    """Load DINOv TensorRT runner."""
    return TrtRunner(engine_path)


def compute_dinovx_feats_from_batch_torch(
    image_batch: torch.Tensor,
    model: AutoModel,
    latent_w: int,
    latent_h: int,
    skip_tokens: int = 5,
) -> torch.Tensor:
    """Compute DINOv low-resolution features from batch using PyTorch.

    Args:
        image_batch: Tensor (1, 3, H, W) normalized
        model: DINOv model
        latent_w, latent_h: Latent grid dimensions
        skip_tokens: Number of special tokens to skip

    Returns:
        lr_feats: Low-resolution features (B, C, latent_h, latent_w)
    """
    all_tokens, patch_tokens = run_vision_transformer(model, image_batch, skip_tokens)
    batch_size = patch_tokens.shape[0]
    patch_tokens = patch_tokens.permute(0, 2, 1)  # (B, C, num_patches)
    lr_feats = patch_tokens.reshape(batch_size, -1, latent_h, latent_w)
    return lr_feats


def compute_dinovx_feats_from_batch_onnx(
    session,
    image_batch: torch.Tensor,
    latent_w: int,
    latent_h: int,
    skip_tokens: int = 5,
    output_name: str = "last_hidden_state",
) -> torch.Tensor:
    """Compute DINOv low-resolution features from batch using ONNX."""
    from ..backends.onnx import prepare_numpy_input, run_session
    import numpy as np

    # Prepare input: handle static models that expect specific H/W order
    in_meta = session.get_inputs()[0]
    in_name = in_meta.name
    in_type = getattr(in_meta, "type", "tensor(float)")
    want_fp16 = "float16" in in_type

    img = image_batch.detach()
    if want_fp16 and img.dtype != torch.float16:
        img = img.half()
    elif (not want_fp16) and img.dtype != torch.float32:
        img = img.float()

    # If model has static shape, reorder if necessary to match expected H/W
    try:
        shp = tuple(getattr(in_meta, "shape", []))
        if len(shp) == 4 and all(isinstance(d, int) and d > 0 for d in shp):
            _, _, exp_h, exp_w = shp
            cur_h = int(image_batch.shape[2])
            cur_w = int(image_batch.shape[3])
            if (exp_h, exp_w) == (cur_w, cur_h):
                # Swap H/W to match static model (e.g., 512x320 expected instead of 320x512)
                img = torch.nn.functional.interpolate(
                    img, size=(exp_h, exp_w), mode="bilinear", align_corners=False
                )
    except Exception:
        pass

    np_arr = prepare_numpy_input(img, np.float16 if want_fp16 else np.float32)
    inputs = {in_name: np_arr}

    # Run inference
    ort_out = run_session(session, inputs, [output_name])[0]
    device = image_batch.device
    all_tokens = torch.from_numpy(ort_out).to(device)
    batch_size = all_tokens.shape[0]

    # Skip special tokens, reshape to latent grid
    patch_tokens = all_tokens[:, skip_tokens:, :]
    patch_tokens = patch_tokens.permute(0, 2, 1)
    lr_feats = patch_tokens.reshape(batch_size, -1, latent_h, latent_w)
    return lr_feats


def compute_dinovx_feats_from_batch_trt(
    runner: TrtRunner,
    image_batch: torch.Tensor,
    latent_w: int,
    latent_h: int,
    skip_tokens: int = 5,
) -> torch.Tensor:
    """Compute DINOv low-resolution features from batch using TensorRT."""

    # Set input bindings
    input_name = next(
        n for n, is_in in runner.binding_is_input.items() if is_in
    )
    runner.set_bindings({input_name: image_batch})

    # Determine output name
    tokens_name = (
        "last_hidden_state" if "last_hidden_state" in runner.bindings else None
    )
    if tokens_name is None:
        tokens_name = next(
            n for n in runner.bindings if not runner.binding_is_input[n]
        )

    # Run inference
    outputs = runner.infer([tokens_name])
    all_tokens = outputs[tokens_name]
    batch_size = all_tokens.shape[0]

    # Post-process tokens
    patch_tokens = all_tokens[:, skip_tokens:, :]
    patch_tokens = patch_tokens.permute(0, 2, 1).contiguous()
    lr_feats = patch_tokens.view(batch_size, patch_tokens.shape[1], latent_h, latent_w)
    return lr_feats


def compute_dinovx_bilinear_upsampled_feats_from_batch(
    lr_feats: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Bilinearly upsample low-resolution features to target dimensions.

    Args:
        lr_feats: Low-resolution features (B, C, latent_h, latent_w)
        target_h, target_w: Target spatial dimensions

    Returns:
        hr_feats: High-resolution upsampled features (B, C, target_h, target_w)
    """
    import torch.nn.functional as F

    hr_feats = F.interpolate(
        lr_feats, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return hr_feats
