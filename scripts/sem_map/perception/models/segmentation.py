# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F

from ..common import get_device, get_sem_map_root
from ..backends.pytorch import run_conv_head
from ..backends.tensorrt import TrtRunner


def _generate_color(class_id: int) -> list[int]:
    """Generate deterministic RGB color from class ID when not in config.

    Args:
        class_id: Integer class ID

    Returns:
        RGB color as [R, G, B] list with values in [0, 255]
    """
    # Use a simple hash-based color generation for consistency
    import hashlib
    hash_val = int(hashlib.md5(str(class_id).encode()).hexdigest()[:6], 16)
    r = (hash_val >> 16) & 0xFF
    g = (hash_val >> 8) & 0xFF
    b = hash_val & 0xFF
    return [r, g, b]


def traversability_from_probs(
    probs: torch.Tensor,
    family: str,
    seg_layer: str,
    scores: dict[int, float] | list[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-pixel traversability as expectation over class probs.

    Args:
        probs: Tensor of shape (B, C, H, W) with class probabilities per pixel.
        family: Model family name (e.g., 'dinov3s')
        seg_layer: Segmentation layer name (e.g., 'dinov3s_sem_seg_city_v1')
        scores: Optional override - mapping/list/tensor of length C with traversability per class in [0,1].
                If None, loads from config via get_traversability_scores(family, seg_layer).

    Returns:
        Tensor of shape (B, H, W) on the same device/dtype as probs.
    """
    if probs.ndim != 4:
        raise AssertionError("probs must be 4D [B,C,H,W]")

    B, C, H, W = probs.shape

    # Load scores from config if not provided
    if scores is None:
        from . import get_traversability_scores
        scores = get_traversability_scores(family, seg_layer)

    if isinstance(scores, dict):
        score_list = [float(scores.get(c, 0.0)) for c in range(C)]
        weight_vec = torch.tensor(score_list, device=probs.device, dtype=probs.dtype)
    elif isinstance(scores, (list, tuple)):
        if len(scores) != C:
            raise AssertionError("scores length must match number of classes")
        weight_vec = torch.tensor(scores, device=probs.device, dtype=probs.dtype)
    else:
        # torch.Tensor
        if scores.numel() != C:
            raise AssertionError("scores length must match number of classes")
        weight_vec = scores.to(device=probs.device, dtype=probs.dtype)

    weights = weight_vec.view(1, C, 1, 1)
    traversability = (probs * weights).sum(dim=1)
    return traversability


class ResBlock(nn.Module):
    """Basic residual block: two conv layers with GroupNorm and a residual add.

    Derived from JAFAR (https://github.com/PaulCouairon/JAFAR), licensed under
    the Apache License, Version 2.0; JAFAR in turn credits magvit1. Modified
    here so the shortcut projection is applied whenever the residual and output
    shapes differ, not only when the channel counts do. See the NOTICE file.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        num_groups=8,
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
    ):
        super().__init__()
        self.use_conv_shortcut = use_conv_shortcut
        self.norm1 = norm_fn(num_groups, in_channels) if norm_fn is not None else nn.Identity()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, padding_mode=pad_mode, bias=False,
        )
        self.norm2 = norm_fn(num_groups, out_channels) if norm_fn is not None else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, padding_mode=pad_mode, bias=False,
        )
        self.activation_fn = activation_fn()
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                padding=0, padding_mode=pad_mode, bias=False,
            )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        if self.shortcut is not None and (self.use_conv_shortcut or residual.shape != x.shape):
            residual = self.shortcut(residual)
        return x + residual


class SegmentationHead(nn.Module):
    """Segmentation head module with prediction and reconstruction branches.

    This module takes high-resolution features and produces:
    - Segmentation predictions (logits)
    - Feature reconstructions (for auxiliary loss)

    Optional ResBlocks refine upsampled features before the MLP.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        intermediate_dim: int = 128,
        num_resblocks: int = 0,
        resblock_num_groups: int = 8,
    ):
        """
        Args:
            feature_dim: Input feature dimension from upsampler
            num_classes: Number of segmentation classes
            intermediate_dim: Dimension of intermediate representation (default: 128)
            num_resblocks: Number of ResBlocks to refine features before MLP (default: 0)
            resblock_num_groups: GroupNorm num_groups for ResBlocks (default: 8)
        """
        super().__init__()

        # ResBlocks to refine upsampled features (before MLP)
        resblock_list = []
        for _ in range(num_resblocks):
            resblock_list.append(
                ResBlock(
                    in_channels=feature_dim,
                    out_channels=feature_dim,
                    kernel_size=3,
                    num_groups=resblock_num_groups,
                    norm_fn=nn.GroupNorm,
                    activation_fn=nn.SiLU,
                    use_conv_shortcut=False,
                )
            )
        self.resblocks = nn.Sequential(*resblock_list) if resblock_list else nn.Identity()

        # Project & Compress layer with BatchNorm
        self.project = nn.Sequential(
            nn.Conv2d(feature_dim, intermediate_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_dim),
            nn.ReLU(inplace=True)
        )

        # Prediction branch: intermediate -> num_classes
        self.classifier = nn.Conv2d(intermediate_dim, num_classes, kernel_size=1)

        # Reconstruction branch: intermediate -> feature_dim
        self.reconstruction = nn.Conv2d(intermediate_dim, feature_dim, kernel_size=1)

    def forward(self, hr_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through segmentation head.

        Args:
            hr_feats: High-resolution features from upsampler, shape (B, feature_dim, H, W)

        Returns:
            tuple of (predictions, reconstruction):
                - predictions: Segmentation logits, shape (B, num_classes, H, W)
                - reconstruction: Reconstructed features, shape (B, feature_dim, H, W)
        """
        x = self.resblocks(hr_feats)
        features = self.project(x)
        predictions = self.classifier(features)
        reconstruction = self.reconstruction(features)
        return predictions, reconstruction


def load_seg_head_torch(
    model_feature_dim: int,
    num_classes: int,
    model_name: str = "jafar",
    checkpoint_dir: str | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Load a segmentation head trained via train_segmentation.py or train_probes.py.

    Supports both old format (simple Conv2d) and new format (SegmentationHead with project/classifier/reconstruction).
    Expects checkpoints saved with key "model_state_dict".
    Looks for checkpoints/seg/{model_name}.pth under repo root or CWD.
    """
    if device is None:
        device = get_device()

    if checkpoint_dir is not None:
        # Allow passing a full file path or a directory
        p = Path(checkpoint_dir)
        ckpt_path = p if p.suffix == ".pth" else (p / f"{model_name}_sem_seg.pth")
    else:
        sem_map_root = get_sem_map_root()
        ckpt_path = sem_map_root / "artifacts" / "torch" / f"{model_name}_sem_seg.pth"

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Segmentation head checkpoint not found: {ckpt_path}")

    state = torch.load(str(ckpt_path), weights_only=False, map_location=device)
    state_dict = state["model_state_dict"]

    # Check if this is the new format (has "project", "classifier", "reconstruction" keys)
    # or old format (just "weight", "bias" keys)
    has_new_format = any(key.startswith("project.") or key.startswith("classifier.") or key.startswith("reconstruction.")
                         for key in state_dict.keys())

    if has_new_format:
        # New format: use SegmentationHead — read resblock params from checkpoint metadata
        num_resblocks = state.get("num_resblocks", 0)
        resblock_num_groups = state.get("resblock_num_groups", 8)
        head = SegmentationHead(
            feature_dim=model_feature_dim,
            num_classes=num_classes,
            intermediate_dim=128,
            num_resblocks=num_resblocks,
            resblock_num_groups=resblock_num_groups,
        )
    else:
        # Old format: simple Conv2d
        head = nn.Conv2d(model_feature_dim, num_classes, kernel_size=1, padding=0, stride=1)

    head.to(device)
    head.load_state_dict(state_dict)
    head.to(device)
    # Keep seg head in FP16 on CUDA to avoid fp16->fp32 casts downstream
    if device.type == "cuda":
        try:
            head.half()
        except Exception:
            pass
    head.eval()
    return head


def load_seg_head_onnx(onnx_path: str):
    """Load segmentation head ONNX session."""
    from ..backends.onnx import create_session
    return create_session(onnx_path)


def load_seg_head_trt(engine_path: str) -> TrtRunner:
    """Load segmentation head TensorRT runner."""
    return TrtRunner(engine_path)


def run_seg_from_batch_torch(
    features: torch.Tensor,
    seg_head: torch.nn.Module,
    temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the trained segmentation head on a batch of features using PyTorch.

    Args:
        features: Tensor (B, C, H, W)
        seg_head: Segmentation head module
        temperature: Optional temperature scaling for softmax

    Returns:
        (segmentation, logits)
    """
    return run_conv_head(seg_head, features, temperature)


def run_seg_from_batch_onnx(
    session,
    features: torch.Tensor,
    temperature: float | None = None,
    output_name: str = "logits",
    min_confidence: float | None = None,
    unknown_label: int = 255,
    return_probs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run segmentation head on features using ONNX with 4D [B,C,H,W] input/output."""
    from ..backends.onnx import prepare_numpy_input, run_session
    import numpy as np

    if features.ndim != 4:
        raise AssertionError("features must be 4D [B,C,H,W]")

    if not hasattr(session, "get_inputs"):
        raise RuntimeError("ONNX session expected")

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    if len(input_names) < 1 or len(output_names) < 1:
        raise RuntimeError("Seg head session must have at least 1 input and 1 output")
    in_name = "features" if "features" in input_names else input_names[0]
    out_name = output_name if output_name in output_names else output_names[0]

    def _onnx_dtype_to_torch(dt_str):
        if "float16" in dt_str:
            return torch.float16
        if "float32" in dt_str:
            return torch.float32
        return torch.float16

    in_meta = next(inp for inp in session.get_inputs() if inp.name == in_name)
    in_dtype = _onnx_dtype_to_torch(in_meta.type)
    dev = features.device
    x = features.to(in_dtype)
    if dev.type != "cuda":
        x = x.cuda(non_blocking=True)

    np_arr = prepare_numpy_input(x, np.float16 if in_dtype == torch.float16 else np.float32)
    inputs = {in_name: np_arr}

    ort_out = run_session(session, inputs, [out_name])[0]
    device = features.device
    logits = torch.from_numpy(ort_out).to(device)

    # Compute probabilities with optional temperature scaling
    if (temperature is not None) and (float(temperature) > 0.0):
        probs = F.softmax(logits / float(temperature), dim=1)
    else:
        probs = F.softmax(logits, dim=1)

    # Derive labels from probabilities
    segmentation = probs.argmax(dim=1)

    # Optional in-place masking of low-confidence pixels
    if (min_confidence is not None) and (float(min_confidence) > 0.0):
        max_conf, _ = probs.max(dim=1)
        mask = max_conf < float(min_confidence)
        segmentation[mask] = int(unknown_label)

    if return_probs:
        return segmentation, logits, probs
    return segmentation, logits


def run_seg_from_batch_trt(
    runner: TrtRunner,
    features: torch.Tensor,
    temperature: float | None = None,
    min_confidence: float | None = None,
    unknown_label: int = 255,
    return_probs: bool = False,
    return_reconstruction: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run segmentation head on features using TensorRT with 4D [B,C,H,W] input/output.

    Args:
        runner: TensorRT runner for segmentation head
        features: Input features (B, C, H, W)
        temperature: Optional temperature scaling for softmax
        min_confidence: Optional minimum confidence threshold
        unknown_label: Label for low-confidence pixels
        return_probs: If True, return probabilities
        return_reconstruction: If True, return reconstruction output (new architecture only)

    Returns:
        Tuple of (segmentation, logits) or (segmentation, logits, probs) or
        (segmentation, logits, probs, reconstruction) depending on flags
    """
    if features.ndim != 4:
        raise AssertionError("features must be 4D [B,C,H,W]")

    input_names = [n for n, is_in in runner.binding_is_input.items() if is_in]
    output_names = [n for n, is_in in runner.binding_is_input.items() if not is_in]
    if len(input_names) < 1 or len(output_names) < 1:
        raise RuntimeError("Seg head engine must have at least 1 input and 1 output")
    in_name = "features" if "features" in input_names else input_names[0]
    out_name = "logits" if "logits" in output_names else output_names[0]

    in_dtype = runner.binding_dtypes.get(in_name, torch.float16)
    dev = features.device
    x = features.to(in_dtype)
    if dev.type != "cuda":
        x = x.cuda(non_blocking=True)

    # Determine which outputs to request
    outputs_to_request = [out_name]
    has_reconstruction = "reconstruction" in output_names
    if return_reconstruction and has_reconstruction:
        outputs_to_request.append("reconstruction")

    runner.set_bindings({in_name: x})
    outputs = runner.infer(outputs_to_request)
    logits = outputs[out_name]
    reconstruction = outputs.get("reconstruction") if return_reconstruction and has_reconstruction else None

    # Compute probabilities with optional temperature scaling
    if (temperature is not None) and (float(temperature) > 0.0):
        probs = F.softmax(logits / float(temperature), dim=1)
    else:
        probs = F.softmax(logits, dim=1)

    # Derive labels from probabilities (same as argmax on logits)
    segmentation = probs.argmax(dim=1)

    # Optional in-place masking of low-confidence pixels without cloning
    if (min_confidence is not None) and (float(min_confidence) > 0.0):
        max_conf, _ = probs.max(dim=1)
        mask = max_conf < float(min_confidence)
        segmentation[mask] = int(unknown_label)

    # Return based on requested outputs (keep on GPU if dev is cuda)
    if dev.type == "cuda":
        if return_reconstruction:
            if return_probs:
                return segmentation, logits, probs, reconstruction
            return segmentation, logits, reconstruction
        else:
            if return_probs:
                return segmentation, logits, probs
            return segmentation, logits

    # Move to CPU only if original device was CPU
    if return_reconstruction:
        if return_probs:
            return segmentation.cpu(), logits.cpu(), probs.cpu(), reconstruction.cpu() if reconstruction is not None else None
        return segmentation.cpu(), logits.cpu(), reconstruction.cpu() if reconstruction is not None else None
    else:
        if return_probs:
            return segmentation.cpu(), logits.cpu(), probs.cpu()
        return segmentation.cpu(), logits.cpu()


def compute_confidence_from_reconstruction(
    hr_feats: torch.Tensor,
    reconstruction: torch.Tensor,
    method: str = "cosine",
    normalize_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Compute per-pixel confidence from feature reconstruction.

    All operations stay on GPU to avoid CPU-GPU transfer overhead.

    Args:
        hr_feats: Original high-resolution features (B, C, H, W) on GPU
        reconstruction: Reconstructed features (B, C, H, W) on GPU
        method: Confidence method - "cosine" or "mse"
        normalize_range: Expected range for normalization (e.g., [-1, 1] for cosine)

    Returns:
        Confidence map (B, H, W) in range [0, 1] on GPU
    """
    if method == "cosine":
        # Normalize features across channel dimension (GPU operation)
        hr_feats_norm = F.normalize(hr_feats, dim=1)  # (B, C, H, W)
        recon_norm = F.normalize(reconstruction, dim=1)  # (B, C, H, W)

        # Compute cosine similarity: dot product of normalized vectors (GPU operation)
        confidence = (hr_feats_norm * recon_norm).sum(dim=1)  # (B, H, W)

        # Map from normalize_range to [0, 1] (GPU operation)
        # Cosine similarity is in [-1, 1], map to [0, 1]
        min_val, max_val = normalize_range
        confidence = (confidence - min_val) / (max_val - min_val)
        confidence = confidence.clamp(0.0, 1.0)

    elif method == "mse":
        # Reconstruction loss per pixel (MSE across feature dimension) (GPU operation)
        mse = torch.square(reconstruction - hr_feats).mean(dim=1)  # (B, H, W)
        # Invert so higher confidence = lower loss
        # Normalize using min/max (GPU operation)
        mse_min = mse.min()
        mse_max = mse.max()
        if mse_max > mse_min:
            confidence = 1.0 - (mse - mse_min) / (mse_max - mse_min)
        else:
            confidence = torch.ones_like(mse)

    return confidence
