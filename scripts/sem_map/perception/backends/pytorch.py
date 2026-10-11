# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import torch
import torch.nn.functional as F
from transformers import AutoModel


def load_model_from_hf(model_id: str, device_map: str = "auto") -> AutoModel:
    """Load a HuggingFace AutoModel."""
    return AutoModel.from_pretrained(model_id, device_map=device_map)


def run_vision_transformer(model: AutoModel, image_batch: torch.Tensor, skip_tokens: int = 5):
    """Generic ViT forward pass.

    Returns (all_tokens, patch_tokens)
    """
    # Don't use inference_mode here - let caller control gradient context
    # This allows more efficient batching and avoids double context wrapping
    outputs = model(image_batch)
    all_tokens = outputs.last_hidden_state
    patch_tokens = all_tokens[:, skip_tokens:, :]
    return all_tokens, patch_tokens


def tokens_to_spatial_features(patch_tokens: torch.Tensor, latent_h: int, latent_w: int, upsample_h: int, upsample_w: int):
    """Reshape patch tokens to spatial features and interpolate.

    Args:
        patch_tokens: (B, num_patches, C)
        latent_h, latent_w: Target spatial dimensions for low-res features
        upsample_h, upsample_w: Target spatial dimensions for high-res features

    Returns:
        lr_feats: (B, C, latent_h, latent_w)
        hr_feats: (B, C, upsample_h, upsample_w)
    """
    batch_size = patch_tokens.shape[0]
    patch_tokens = patch_tokens.permute(0, 2, 1)  # (B, C, num_patches)
    lr_feats = patch_tokens.reshape(batch_size, -1, latent_h, latent_w)
    hr_feats = F.interpolate(
        lr_feats, size=(upsample_h, upsample_w), mode="bilinear", align_corners=False
    )
    return lr_feats, hr_feats


def run_conv_head(head: torch.nn.Module, features: torch.Tensor, temperature: float | None = None):
    """Generic Conv2d head inference.

    Args:
        head: Conv2d head module (old format) or SegmentationHead (new format)
        features: Input features (B, C, H, W)
        temperature: Optional temperature scaling for softmax

    Returns:
        segmentation: (B, H, W) - argmax predictions
        logits: (B, num_classes, H, W) - raw logits
    """
    with torch.inference_mode():
        output = head(features)
        # Handle both old format (single tensor) and new format (tuple: predictions, reconstruction)
        if isinstance(output, tuple):
            logits = output[0]  # Extract predictions from tuple
        else:
            logits = output  # Old format: single tensor
        if (temperature is not None) and (float(temperature) > 0.0):
            _ = F.softmax(logits / float(temperature), dim=1)  # optional use, keep logits for argmax
        segmentation = logits.argmax(dim=1)
    return segmentation, logits
