# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Semantic pipeline class for processing RGB images through TRT models.

Consolidates all the logic from trt_pipeline_wrapper.py into a stateful class
that can be instantiated by C++ with parameter configuration.
"""

from pathlib import Path
from typing import Optional
import time
import os
import sys

import torch
import numpy as np
import yaml

from .perception.models.dinov import (
    load_dinovx_trt,
    compute_dinovx_feats_from_batch_trt,
    compute_dinovx_bilinear_upsampled_feats_from_batch,
)
from .perception.models.jafar import (
    load_jafar_trt,
    compute_jafar_upscaled_feats_from_batch_trt,
)
from .perception.models.segmentation import (
    load_seg_head_trt,
    run_seg_from_batch_trt,
    traversability_from_probs,
)
from .perception.common import (
    get_device,
    IMAGENET_MEAN,
    IMAGENET_STD,
    prepare_image_trt,
    get_pca_dir,
)
from .perception.visualization import (
    pca,
    load_torch_pca_fit as _load_torch_pca_fit,
)


def _dbg(msg: str):
    try:
        sys.stderr.write(f"[sem_map][py] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


class SemanticPipeline:
    """End-to-end TRT semantic pipeline for processing RGB images.

    This class encapsulates all the semantic processing logic that was previously
    in trt_pipeline_wrapper.py, providing a clean interface for C++ integration.
    """

    def __init__(self, param):
        """Initialize the semantic pipeline with parameters.

        Args:
            param: Parameter instance containing all configuration
        """
        self.param = param
        self.device = get_device()

        # Extract parameters
        self.backbone_name = param.get_value('backbone_name')
        self.upsampler_type = param.get_value('upsampler_type')
        self.upsample_partial_factor = param.get_value('upsample_partial_factor')
        self.config_path = param.get_value('config_path')
        self.seg_layer = param.get_value('seg_layer')
        self.traversability_temperature = param.get_value('traversability_temperature', 1.0)
        # Optional features
        self.enable_pca = param.get_value('enable_pca', True)
        self.flip_image = param.get_value('flip_image', False)  # Flip RGB images and outputs vertically

        # Model configuration
        self.patch_size = param.get_value('patch_size')
        self.feature_dim = param.get_value('feature_dims')
        self.latent_h = param.get_value('latent_h')
        self.latent_w = param.get_value('latent_w')

        # Initialize models and components
        self._load_model_config()
        self._load_trt_engines()
        self._setup_normalization()
        self._load_pca_fit()

        # Store last traversability for external access
        self.last_traversability = None
        # Store last high-resolution features for external access
        self.last_hr_feats = None
        # Store last low-resolution features for external access
        self.last_lr_feats = None
        # Store last segmentation for external access
        self.last_segmentation = None
        # Store last PCA visualization for external access
        self.last_pca = None
        # Store last confidence map for external access
        self.last_confidence = None

        # Cache traversability scores to avoid loading config every frame
        from .perception.models import get_traversability_scores
        self.traversability_scores = get_traversability_scores(self.backbone_name, self.seg_layer)

    def _load_model_config(self):
        """Load model configuration from YAML file."""
        from .perception.models import load_models_yaml

        cfg = load_models_yaml()

        if self.backbone_name not in cfg.get("families", {}):
            raise ValueError(
                f"Model '{self.backbone_name}' not found in families. "
                f"Available: {list(cfg.get('families', {}).keys())}"
            )

        self.model_cfg = cfg["families"][self.backbone_name]

    def _load_trt_engines(self):
        """Load TensorRT engines for backbone, JAFAR, and segmentation.

        Engines are loaded in parallel using threads since TRT deserialization
        is a C-level call that releases the GIL.
        """
        from concurrent.futures import ThreadPoolExecutor
        from .perception.models import resolve_artifact_path

        # Resolve all engine paths first (fast, no heavy I/O)
        self.backbone_engine = resolve_artifact_path(
            self.backbone_name, kind="backbone", artifact_type="trt"
        )

        jafar_engine_path = None
        self.trt_jafar = None
        if self.upsampler_type in ("jafar_full", "jafar_partial"):
            factor = self.patch_size if self.upsampler_type == "jafar_full" else self.upsample_partial_factor
            try:
                jafar_engine_path = str(resolve_artifact_path(
                    self.backbone_name,
                    kind="upsampler",
                    artifact_type="trt",
                    upsampler="jafar",
                    upsample_factor=int(factor)
                ))
            except Exception as e:
                raise KeyError(f"Missing JAFAR TRT engine for factor 'x{int(factor)}'") from e

        seg_engine_path = str(resolve_artifact_path(
            self.backbone_name,
            kind="seg_layer",
            artifact_type="trt",
            seg_layer=self.seg_layer
        ))

        # Load all engines in parallel
        futures = {}
        n_engines = 2 + (1 if jafar_engine_path else 0)
        with ThreadPoolExecutor(max_workers=n_engines) as pool:
            futures["backbone"] = pool.submit(load_dinovx_trt, str(self.backbone_engine))
            if jafar_engine_path:
                futures["jafar"] = pool.submit(load_jafar_trt, jafar_engine_path)
            futures["seg"] = pool.submit(load_seg_head_trt, seg_engine_path)

            # Collect results (re-raises any exception from the worker thread)
            self.trt_backbone = futures["backbone"].result()
            if jafar_engine_path:
                self.trt_jafar = futures["jafar"].result()
            self.trt_seg = futures["seg"].result()

    def _setup_normalization(self):
        """Setup normalization tensors."""
        self.mean_t = torch.tensor(IMAGENET_MEAN, device=self.device).view(3, 1, 1)
        self.std_t = torch.tensor(IMAGENET_STD, device=self.device).view(3, 1, 1)
        self.use_fp16 = (self.device.type == "cuda")

    def _load_pca_fit(self):
        """Load PCA fit for feature visualization (optional)."""
        self.pca_fit = None
        if (not self.enable_pca):
            return

        try:
            pca_dir = get_pca_dir()
            pca_file = pca_dir / f"{self.backbone_name}_hr_lat{self.latent_w}x{self.latent_h}_ps{self.patch_size}.pt"
            if pca_file.exists():
                self.pca_fit = _load_torch_pca_fit(pca_file)
        except Exception:
            # PCA is optional, continue without it
            pass

    def _to_tensor(self, data_ptr, h: int, w: int, c: int) -> torch.Tensor:
        """Convert raw data to tensor.

        Args:
            data_ptr: Raw image data (bytes, bytearray, numpy array, or torch tensor)
            h, w, c: Image dimensions (used for validation if tensor is provided)

        Returns:
            Normalized CHW tensor on device
        """
        # Fast path: already a torch tensor (e.g., from CUDA IPC)
        if isinstance(data_ptr, torch.Tensor):
            img_tensor = data_ptr

            # Validate shape
            if img_tensor.ndim == 3:
                if img_tensor.shape != (h, w, c):
                    _dbg(f"WARNING: Tensor shape {img_tensor.shape} doesn't match expected ({h}, {w}, {c})")
            else:
                raise ValueError(f"Expected 3D tensor (H,W,C), got shape {img_tensor.shape}")

            # Move to correct device if needed
            if img_tensor.device != self.device:
                img_tensor = img_tensor.to(self.device, non_blocking=True)

            # Permute HWC → CHW
            if img_tensor.shape[2] == c:  # HWC format
                img_chw = img_tensor.permute(2, 0, 1).contiguous()
            else:  # Assume already CHW
                img_chw = img_tensor

        # Legacy path: numpy/bytes/bytearray
        elif isinstance(data_ptr, (bytes, bytearray)):
            arr = np.frombuffer(data_ptr, dtype=np.uint8)
            # Ensure the array is writable
            if not arr.flags.writeable:
                arr = arr.copy()
            arr = arr.reshape(h, w, c)
            # Convert to CHW on device
            img_chw = torch.from_numpy(arr).to(self.device, non_blocking=True).permute(2, 0, 1).contiguous()

        elif isinstance(data_ptr, np.ndarray):
            arr = data_ptr
            # Ensure the array is writable
            if not arr.flags.writeable:
                arr = arr.copy()
            arr = arr.reshape(h, w, c)
            # Convert to CHW on device
            img_chw = torch.from_numpy(arr).to(self.device, non_blocking=True).permute(2, 0, 1).contiguous()

        else:
            raise TypeError(f"data must be bytes/bytearray/ndarray/torch.Tensor, got {type(data_ptr)}")

        # Normalize to [0, 1] based on value range
        if img_chw.dtype == torch.uint8:
            # uint8 [0-255] → float32 [0-1]
            img_chw = img_chw.float().mul_(1.0 / 255.0)
        elif img_chw.dtype in (torch.float32, torch.float16):
            # Check if values are in [0-255] range (from buffer) or already [0-1]
            if img_chw.max() > 1.0:
                # float32 [0-255] → [0-1]
                img_chw = img_chw.mul_(1.0 / 255.0)
            # else: already normalized [0-1], skip
        else:
            # Convert other dtypes to float32
            img_chw = img_chw.float()

        # Convert to fp16 if needed
        if self.use_fp16 and img_chw.dtype == torch.float32:
            img_chw = img_chw.half()

        return img_chw

    def process_frame(self, image_data, height: int, width: int, channels: int,
                     sensor_name: str, stamp_ns: int) -> None:
        """Process a single RGB frame through the semantic pipeline.

        Args:
            image_data: Raw image data (bytes, bytearray, numpy array, or torch.Tensor)
                       If torch.Tensor, should be (H,W,C) on GPU for zero-copy processing
            height: Image height in pixels
            width: Image width in pixels
            channels: Number of channels (should be 3 for RGB)
            sensor_name: Name of the sensor for logging
            stamp_ns: Timestamp in nanoseconds
        """
        # Optional debug bypass to isolate CUDA/TRT issues
        if os.environ.get("SEM_MAP_DEBUG_SKIP_COMPUTE", "").lower() in ("1", "true", "yes"):
            return

        # Convert to tensor
        img_chw = self._to_tensor(image_data, int(height), int(width), int(channels))

        # Flip image vertically if flip_image is enabled
        # img_chw is in CHW format (C, H, W), so we flip along dimension 1 (height)
        if self.flip_image:
            img_chw = torch.flip(img_chw, dims=[1])  # Flip along height dimension

        # Normalize + resize on-device
        unorm_img, image_batch, input_h, input_w = prepare_image_trt(
            img_chw,
            latent_w=self.latent_w,
            latent_h=self.latent_h,
            patch_size=self.patch_size,
            device=self.device,
            mean=self.mean_t,
            std=self.std_t,
        )

        try:
            # Backbone → lr_feats
            lr_feats = compute_dinovx_feats_from_batch_trt(
                runner=self.trt_backbone,
                image_batch=image_batch,
                latent_w=self.latent_w,
                latent_h=self.latent_h,
            )

            # Cache last low-resolution features
            self.last_lr_feats = lr_feats

            # Upsampler → hr_feats
            if self.upsampler_type == "bilinear":
                # Honour upsample_partial_factor: bilinearly upsample latent by
                # `factor` (relative to patch_size). factor == patch_size → full
                # input resolution; smaller factor keeps hr_feats partial and
                # the post-processing block below upsamples seg/trav/conf to
                # full resolution.
                factor = max(1, int(self.upsample_partial_factor))
                scale = max(1, int(round(float(self.patch_size) / float(factor))))
                target_h = max(1, input_h // scale)
                target_w = max(1, input_w // scale)
                hr_feats = compute_dinovx_bilinear_upsampled_feats_from_batch(lr_feats, target_h, target_w)
            else:
                # JAFAR upsampling
                factor = self.patch_size if self.upsampler_type == "jafar_full" else self.upsample_partial_factor
                scale = max(1, int(round(float(self.patch_size) / float(max(1, factor)))))

                if scale > 1:
                    H, W = int(image_batch.shape[2]), int(image_batch.shape[3])
                    image_batch_ds = torch.nn.functional.interpolate(
                        image_batch,
                        size=(max(1, H // scale), max(1, W // scale)),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    image_batch_ds = image_batch

                hr_feats = compute_jafar_upscaled_feats_from_batch_trt(
                    runner=self.trt_jafar,
                    image_batch=image_batch_ds,
                    lr_feats=lr_feats
                )
                # For jafar_partial: segmentation runs at JAFAR x4 resolution,
                # outputs are upsampled to full resolution afterwards

            # Cache last high-resolution features
            self.last_hr_feats = hr_feats

            # Segmentation at JAFAR feature resolution (+ probabilities + reconstruction)
            seg, logits, probs, reconstruction = run_seg_from_batch_trt(
                self.trt_seg,
                hr_feats,
                temperature=self.traversability_temperature,
                return_probs=True,
                return_reconstruction=True,
            )

            # Traversability expectation per pixel (pass cached scores to avoid 20ms YAML load)
            trav_map = traversability_from_probs(probs, self.backbone_name, self.seg_layer, scores=self.traversability_scores)[0]

            # Compute confidence from reconstruction if available (all GPU operations)
            if reconstruction is not None:
                from .perception.models.segmentation import compute_confidence_from_reconstruction
                confidence_map = compute_confidence_from_reconstruction(
                    hr_feats,
                    reconstruction,
                    method="cosine",
                    normalize_range=(-1.0, 1.0)
                )[0]  # Extract first batch element
            else:
                confidence_map = None

            # For partial-resolution upsamplers (jafar_partial, bilinear with
            # factor < patch_size): upsample segmentation outputs to full res
            if self.upsampler_type in ("jafar_partial", "bilinear") and hr_feats.shape[-2:] != (input_h, input_w):
                # Segmentation labels: nearest interpolation to preserve class IDs
                seg = torch.nn.functional.interpolate(
                    seg.unsqueeze(1).float(),
                    size=(input_h, input_w),
                    mode="nearest-exact",
                ).squeeze(1).to(seg.dtype)
                # Traversability: bilinear for smooth output
                trav_map = torch.nn.functional.interpolate(
                    trav_map.unsqueeze(0).unsqueeze(0),
                    size=(input_h, input_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                # Confidence: bilinear for smooth output
                if confidence_map is not None:
                    confidence_map = torch.nn.functional.interpolate(
                        confidence_map.unsqueeze(0).unsqueeze(0),
                        size=(input_h, input_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)

            # Flip outputs vertically if flip_image is enabled
            # trav_map is (H, W), seg is batch tensor (B, H, W) or (1, H, W), flip along dimension 1 (height)
            if self.flip_image:
                trav_map = torch.flip(trav_map, dims=[0])  # Flip along height dimension
                # Flip segmentation batch: seg is (B, H, W), flip along dimension 1 (height)
                seg = torch.flip(seg, dims=[1])
                # Flip confidence if available
                if confidence_map is not None:
                    confidence_map = torch.flip(confidence_map, dims=[0])

            # Store results for external access
            # Keep traversability on GPU (will be moved to CPU only when needed by C++ ROS)
            self.last_traversability = trav_map.detach()
            self.last_segmentation = seg[0].detach().to(torch.uint8)
            # Keep confidence on GPU (will be moved to CPU only when needed by C++ ROS)
            self.last_confidence = confidence_map.detach() if confidence_map is not None else None

            # Compute and store PCA visualization if enabled
            if self.enable_pca:
                try:
                    if self.pca_fit is not None:
                        hr_pca, _ = pca([hr_feats.float()], dim=3, fit_pca=self.pca_fit, use_torch_pca=True)
                        pca_img = hr_pca[0][0].permute(1, 2, 0).detach()
                        self.last_pca = pca_img
                except Exception:
                    # PCA is optional, continue without it
                    self.last_pca = None
        except Exception as e:
            raise

    def close(self) -> None:
        """Deterministically release GPU/TRT resources.

        Helps avoid finalize-time destructor ordering issues when Python is
        embedded and the interpreter is torn down by C++.
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        try:
            if hasattr(self, "trt_backbone"):
                self.trt_backbone = None
            if hasattr(self, "trt_jafar"):
                self.trt_jafar = None
            if hasattr(self, "trt_seg"):
                self.trt_seg = None
            if hasattr(self, "mean_t"):
                self.mean_t = None
            if hasattr(self, "std_t"):
                self.std_t = None
            if hasattr(self, "last_hr_feats"):
                self.last_hr_feats = None
            if hasattr(self, "last_lr_feats"):
                self.last_lr_feats = None
            if hasattr(self, "last_segmentation"):
                self.last_segmentation = None
            if hasattr(self, "last_pca"):
                self.last_pca = None
        except Exception:
            pass

        try:
            import gc
            gc.collect()
        except Exception:
            pass
