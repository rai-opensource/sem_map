#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Training script for segmentation head using DINOv3 and an optional upsampler in PyTorch.

This script:
- Uses PyTorch models for the DINOv3 backbone and (optionally) a JAFAR upsampler
- Trains a PyTorch segmentation head (classifier) on top of the features
- Supports plain bilinear feature upsampling as an alternative to JAFAR
- Follows the structure of train_probes_sem_map.py but uses PyTorch for inference

Design Decision - Hydra:
This script uses argparse instead of Hydra for configuration management because:
1. Simpler dependency management (no Hydra required)
2. Easier to understand and modify for users not familiar with Hydra
3. The sem_map package doesn't currently use Hydra extensively
4. For training scripts, argparse provides sufficient flexibility

If you need more complex configuration management (nested configs, config composition, etc.),
you can easily migrate to Hydra by:
- Adding @hydra.main decorator
- Creating config files in a config/ directory
- Using instantiate() for dataset/dataloader creation
"""

import datetime
import sys
import random
from pathlib import Path
import argparse
import warnings

# Suppress matplotlib warnings (harmless, from dependencies)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for training
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn)
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy, JaccardIndex
from tqdm import tqdm
import wandb


from sem_map.perception.models.dinov import (
    load_dinovx_torch,
    compute_dinovx_feats_from_batch_torch,
    DINOV_CONFIGS,
)
from sem_map.perception.models.jafar import (
    load_jafar_torch,
    compute_jafar_upscaled_feats_from_batch_torch,
)
from sem_map.perception.models import (
    resolve_artifact_path,
    get_num_classes,
    load_models_yaml,
    load_seg_layer_config,
)
from sem_map.perception.common import (
    get_device,
    get_sem_map_root,
    create_segmentation_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from sem_map.perception.models.segmentation import ResBlock, SegmentationHead
from sem_map.perception.visualization import label_to_color, unnormalize_image
from sem_map.training.datasets import SimpleSegmentationDataset, _create_dataloaders, parse_img_size
from sem_map.training.utils import EarlyStopping, create_val_figure

LOG_INTERVAL = 100


class SegmentationTrainer:
    """Trainer for segmentation head using PyTorch backbone and upsampler."""

    def __init__(
        self,
        backbone_model,
        upsampler_model,
        device,
        num_classes,
        feature_dim,
        patch_size,
        skip_tokens,
        writer,
        console,
        config,
        use_wandb=False,
        categories=None,
    ):
        self.backbone_model = backbone_model
        self.upsampler_model = upsampler_model
        self.device = device
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.skip_tokens = skip_tokens
        self.writer = writer
        self.console = console
        self.config = config
        self.use_wandb = use_wandb
        self.categories = categories

        # Initialize segmentation head module
        num_resblocks = getattr(config, "num_resblocks", 0)
        resblock_num_groups = getattr(config, "resblock_num_groups", 8)
        self.segmentation_head = SegmentationHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            intermediate_dim=128,
            num_resblocks=num_resblocks,
            resblock_num_groups=resblock_num_groups,
        ).to(device)

        # Metrics
        self.accuracy_metric = Accuracy(
            num_classes=num_classes, task="multiclass"
        ).to(device)
        self.iou_metric = JaccardIndex(
            num_classes=num_classes, task="multiclass"
        ).to(device)

    def set_optimizer(self, lr, weight_decay=1e-5, num_epochs=20):
        """Set up optimizer and scheduler."""
        params = list(self.segmentation_head.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )

        num_params = sum(p.numel() for p in params if p.requires_grad)
        self.log_print(
            f"[bold cyan]Number of optimized parameters: {num_params:,}[/bold cyan]"
        )

    def log_print(self, *args, **kwargs):
        """Log to both file and terminal with immediate updates."""
        Console(force_terminal=True).print(*args, **kwargs)
        self.console.print(*args, **kwargs)
        if hasattr(self.console, "file") and self.console.file:
            self.console.file.flush()

    def log_tensorboard(self, step, loss=None, metrics=None):
        """Log losses and metrics to TensorBoard."""
        if loss is not None:
            self.writer.add_scalar("Loss/Step", loss, step)
        if metrics is not None:
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, step)

    def process_batch(self, image_batch, target, is_training=True):
        """Process batch through PyTorch models and return predictions/targets."""
        B, C, H, W = image_batch.shape

        # Use configured target size (accounts for seg_scale)
        target_size = self.config.target_size
        if isinstance(target_size, tuple):
            target_H, target_W = target_size
        else:
            target_H = target_W = target_size

        # Downscale labels to segmentation resolution if needed
        if target.shape[-2:] != (target_H, target_W):
            target_dtype = target.dtype
            target = (
                F.interpolate(
                    target.unsqueeze(1).float(),
                    size=(target_H, target_W),
                    mode="nearest-exact",
                )
                .squeeze(1)
                .to(target_dtype)
            )

        # Compute latent dimensions
        latent_h = H // self.patch_size
        latent_w = W // self.patch_size

        upsampler_type = getattr(self.config, 'upsampler_type', 'jafar')

        # Downscale image for JAFAR guide if seg_scale < 16 (not full res)
        seg_scale = getattr(self.config, 'seg_scale', 16)
        if upsampler_type == 'jafar' and seg_scale < 16:
            jafar_image = F.interpolate(image_batch, size=(target_H, target_W), mode="bilinear", align_corners=False)
        else:
            jafar_image = image_batch

        with torch.no_grad():
            # Run DINOv3 backbone through PyTorch (supports batches)
            lr_feats = compute_dinovx_feats_from_batch_torch(
                image_batch,
                self.backbone_model,
                latent_w,
                latent_h,
                skip_tokens=self.skip_tokens,
            )

            if upsampler_type == 'bilinear':
                # Simple bilinear upsampling of latent features to target size
                hr_feats = F.interpolate(
                    lr_feats,
                    size=(target_H, target_W),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                # Run JAFAR upsampler through PyTorch (supports batches)
                # Pass downscaled image as guide, output matches target size directly
                hr_feats = compute_jafar_upscaled_feats_from_batch_torch(
                    jafar_image,
                    self.upsampler_model,
                    lr_feats,
                    output_size=(target_H, target_W),
                )

        # Run segmentation head (PyTorch, trainable)
        pred, reconstruction = self.segmentation_head(hr_feats)

        # Create mask for valid pixels (not 255)
        valid_mask = target != 255

        # Create mask for non-background pixels for reconstruction loss
        # This mask excludes invalid pixels (255) and background pixels (if a background class exists)
        # Background class ID is determined from seg_layer config (by category name "background")
        # This mask is in original target shape (B, H, W)
        bg_id = self.config.background_class_id
        non_background_mask = target != 255
        if bg_id is not None:
            non_background_mask = non_background_mask & (target != bg_id)

        # Reshape predictions and targets
        pred = rearrange(pred, "b c h w -> (b h w) c")
        target = rearrange(target, "b h w -> (b h w)")
        valid_mask = rearrange(valid_mask, "b h w -> (b h w)")

        # Apply mask to both pred and target
        pred = pred[valid_mask]
        target = target[valid_mask]

        return pred, target, reconstruction, hr_feats, non_background_mask

    def _prepare_batch(self, batch):
        """Prepare batch for processing."""
        batch["image"] = batch["image"].to(self.device)
        batch["label"] = batch["label"].to(self.device)
        return batch["image"], batch["label"]

    def train(self, train_dataloader, progress, epoch, start_time):
        """Train for one epoch."""
        self.log_print(f"[yellow]Training model epoch {epoch+1}...[/yellow]")
        # Set models to appropriate modes
        self.backbone_model.eval()
        if self.upsampler_model is not None:
            self.upsampler_model.eval()
        self.segmentation_head.train()

        epoch_task = progress.add_task(
            f"Epoch {epoch+1}/{self.config.num_epochs}",
            total=len(train_dataloader),
            loss=0.0,
            step=0,
        )
        total_loss = 0
        total_ce_loss = 0
        total_reconstruction_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            image_batch, target = self._prepare_batch(batch)

            # Random horizontal flip augmentation
            if random.random() < 0.5:
                image_batch = torch.flip(image_batch, dims=[3])
                target = torch.flip(target, dims=[2])

            self.optimizer.zero_grad()

            pred, target_masked, reconstruction, hr_feats, non_background_mask = self.process_batch(
                image_batch, target, is_training=True
            )

            # Cross-entropy loss for segmentation
            ce_loss = F.cross_entropy(pred, target_masked)

            # Reconstruction loss (MSE between reconstructed and original features)
            # Exclude background pixels (class 0) from reconstruction loss
            reconstruction_diff = torch.square(reconstruction - hr_feats)  # (B, C, H, W)
            # Apply non-background mask: expand mask to match feature dimensions
            non_background_mask_expanded = non_background_mask.unsqueeze(1)  # (B, 1, H, W)
            reconstruction_diff_masked = reconstruction_diff * non_background_mask_expanded
            # Compute mean only over non-background pixels
            # Sum over all dimensions and divide by number of non-background pixels * feature dims
            num_non_background = non_background_mask.sum().float()
            feature_dim = reconstruction.shape[1]
            if num_non_background > 0:
                reconstruction_loss = reconstruction_diff_masked.sum() / (num_non_background * feature_dim)
            else:
                # Fallback if no non-background pixels (shouldn't happen in practice)
                reconstruction_loss = torch.tensor(0.0, device=reconstruction.device)

            # Combined loss
            reconstruction_weight = getattr(self.config, 'reconstruction_weight', 0.1)
            loss = ce_loss + reconstruction_weight * reconstruction_loss

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()

            avg_loss = total_loss / (batch_idx + 1)
            avg_ce_loss = total_ce_loss / (batch_idx + 1)
            avg_reconstruction_loss = total_reconstruction_loss / (batch_idx + 1)

            # Update progress and log
            if (batch_idx + 1) % LOG_INTERVAL == 0 or batch_idx == len(train_dataloader) - 1:
                elapsed_time = datetime.datetime.now() - start_time
                elapsed_str = str(elapsed_time).split(".")[0]
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Update progress bar (always advance by LOG_INTERVAL)
                progress.update(
                    epoch_task,
                    advance=LOG_INTERVAL,
                    loss=avg_loss,
                    step=batch_idx + 1,
                )

                self.log_print(
                    f"[cyan]Iteration {batch_idx + 1}[/cyan] - "
                    f"Loss: {avg_loss:.6f} (CE: {avg_ce_loss:.6f}, Recon: {avg_reconstruction_loss:.6f}) - "
                    f"LR: {current_lr:.5e} - "
                    f"Elapsed Time: {elapsed_str}"
                )

                if hasattr(self.console, "file"):
                    self.console.file.flush()

                progress.refresh()

                # Log loss to TensorBoard
                step = epoch * len(train_dataloader) + batch_idx
                self.log_tensorboard(
                    step,
                    loss=avg_loss,
                    metrics={
                        "CE_Loss": avg_ce_loss,
                        "Reconstruction_Loss": avg_reconstruction_loss
                    }
                )

                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/ce_loss": avg_ce_loss,
                        "train/reconstruction_loss": avg_reconstruction_loss,
                        "train/learning_rate": current_lr,
                        "train/step": step,
                        "train/epoch": epoch + 1,
                    })

            if self.config.sanity:
                break

        self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]
        self.log_print(
            f"[bold cyan]Epoch {epoch+1} Summary:[/bold cyan] "
            f"Loss = {avg_loss:.6f} - "
            f"LR = {current_lr:.2e}"
        )

        # Log epoch summary to wandb
        if self.use_wandb:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch_ce_loss": avg_ce_loss,
                "train/epoch_reconstruction_loss": avg_reconstruction_loss,
                "train/epoch_learning_rate": current_lr,
                "train/epoch": epoch + 1,
            })

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.config.num_epochs - 1,
            "model_state_dict": self.segmentation_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "task": "seg",
            "backbone": "dinov3s",
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "num_resblocks": getattr(self.config, "num_resblocks", 0),
            "resblock_num_groups": getattr(self.config, "resblock_num_groups", 8),
            "upsampler_type": getattr(self.config, "upsampler_type", "jafar"),
        }
        torch.save(checkpoint, checkpoint_path)
        self.log_print(
            f"[bold green]Training completed. Model saved at: {checkpoint_path}[/bold green]"
        )
        self.console.file.flush() if hasattr(self.console, "file") else None

    @torch.inference_mode()
    def log_wandb_images(self, val_dataset, vis_indices, epoch):
        """Log validation sample visualizations to wandb.

        For each sample, creates a side-by-side figure with input image,
        ground truth segmentation, and predicted segmentation.
        """
        if not self.use_wandb or self.categories is None:
            return

        self.segmentation_head.eval()
        images = []

        for idx in vis_indices:
            sample = val_dataset[idx]
            image = sample["image"].unsqueeze(0).to(self.device)
            label = sample["label"]

            # Get prediction at configured segmentation resolution
            B, C, H, W = image.shape
            target_size = self.config.target_size
            if isinstance(target_size, tuple):
                target_H, target_W = target_size
            else:
                target_H = target_W = target_size
            latent_h = H // self.patch_size
            latent_w = W // self.patch_size

            upsampler_type = getattr(self.config, 'upsampler_type', 'jafar')
            seg_scale = getattr(self.config, 'seg_scale', 16)
            if upsampler_type == 'jafar' and seg_scale < 16:
                jafar_image = F.interpolate(image, size=(target_H, target_W), mode="bilinear", align_corners=False)
            else:
                jafar_image = image

            lr_feats = compute_dinovx_feats_from_batch_torch(
                image, self.backbone_model, latent_w, latent_h,
                skip_tokens=self.skip_tokens,
            )
            if upsampler_type == 'bilinear':
                hr_feats = F.interpolate(
                    lr_feats, size=(target_H, target_W),
                    mode="bilinear", align_corners=False,
                )
            else:
                hr_feats = compute_jafar_upscaled_feats_from_batch_torch(
                    jafar_image, self.upsampler_model, lr_feats,
                    output_size=(target_H, target_W),
                )
            pred_logits, reconstruction = self.segmentation_head(hr_feats)
            pred = pred_logits.argmax(dim=1)[0]  # (target_H, target_W)

            # Cosine similarity between reconstruction and HR features as confidence
            hr_norm = F.normalize(hr_feats, dim=1)
            recon_norm = F.normalize(reconstruction, dim=1)
            cosine_sim = (hr_norm * recon_norm).sum(dim=1)  # (B, H, W)

            # Upscale prediction and confidence to input resolution for visualization
            label_H, label_W = label.shape[-2:]
            if pred.shape[-2:] != (label_H, label_W):
                pred = F.interpolate(
                    pred.unsqueeze(0).unsqueeze(0).float(),
                    size=(label_H, label_W),
                    mode="nearest-exact",
                ).squeeze(0).squeeze(0).to(pred.dtype)
                cosine_sim = F.interpolate(
                    cosine_sim.unsqueeze(1),  # (B, H, W) -> (B, 1, H, W)
                    size=(label_H, label_W),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)  # back to (B, H, W)

            pred = pred.cpu().numpy()

            # Normalize cosine similarity to [0, 1] for visualization
            cs = cosine_sim[0] if cosine_sim.ndim == 3 else cosine_sim
            cs_min, cs_max = cs.min(), cs.max()
            if cs_max > cs_min:
                confidence = ((cs - cs_min) / (cs_max - cs_min)).cpu().numpy()
            else:
                confidence = torch.ones_like(cs).cpu().numpy()

            # Unnormalize image → numpy HWC uint8
            raw_img = unnormalize_image(image[0].cpu(), IMAGENET_MEAN, IMAGENET_STD)
            raw_img = (raw_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Colorize labels
            gt_colored = label_to_color(label.numpy(), self.categories)
            pred_colored = label_to_color(pred, self.categories)

            # Create side-by-side figure
            fig = create_val_figure(raw_img, gt_colored, pred_colored, confidence, confidence_title="Confidence (cosine)")
            images.append(wandb.Image(fig, caption=f"Sample {idx}"))
            plt.close(fig)

        wandb.log({"val/predictions": images, "val/epoch": epoch + 1})

    @torch.inference_mode()
    def evaluate(self, dataloader, epoch):
        """Evaluate on validation set."""
        self.log_print("[yellow]Evaluating model...[/yellow]")
        torch.cuda.empty_cache()

        self.segmentation_head.eval()

        # Reset metrics
        self.accuracy_metric.reset()
        self.iou_metric.reset()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            image_batch, target = self._prepare_batch(batch)

            pred, target_masked, _, _, _ = self.process_batch(
                image_batch, target, is_training=False
            )

            self.accuracy_metric(pred, target_masked)
            self.iou_metric(pred, target_masked)

            if self.config.sanity and batch_idx == 0:
                break

        metrics = {
            "accuracy": self.accuracy_metric.compute().item(),
            "iou": self.iou_metric.compute().item(),
        }

        # Log metrics to TensorBoard
        self.log_tensorboard(step=epoch, metrics=metrics)

        # Log metrics to wandb
        if self.use_wandb:
            wandb.log({
                "val/accuracy": metrics["accuracy"],
                "val/iou": metrics["iou"],
                "val/epoch": epoch + 1,
            })

        self.log_print(f"[bold green]Results: {metrics}[/bold green]")
        return metrics



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train segmentation head using PyTorch DINOv3 and upsampler"
    )

    # Model configuration
    parser.add_argument(
        "--model_family",
        type=str,
        default="dinov3s",
        choices=["dinov2s", "dinov3s"],
        help="Model family (default: dinov3s)",
    )
    parser.add_argument(
        "--seg_tag",
        type=str,
        required=True,
        help="Segmentation layer tag (e.g., 'city_v1' - will be prefixed with '{model_family}_sem_seg_')",
    )
    parser.add_argument(
        "--seg_layer",
        type=str,
        default=None,
        help="Full segmentation layer name (overrides seg_tag if provided, e.g., 'dinov3s_sem_seg_city_v1')",
    )
    parser.add_argument(
        "--seg_layer_config",
        type=str,
        default=None,
        help="Name of seg_layer config file (e.g., 'cityscapes_19' or 'cityscapes_19.yaml' or full path). "
             ".yaml extension is added automatically if not provided. "
             "If provided, num_classes will be loaded from this file instead of models.yaml",
    )
    parser.add_argument(
        "--upsampler_factor",
        type=int,
        default=16,
        help="Upsampler upsampling factor (default: 16)",
    )
    parser.add_argument(
        "--upsampler_type",
        type=str,
        default="jafar",
        choices=["jafar", "bilinear"],
        help="Feature upsampler to use during training: 'jafar' (default) uses the "
             "JAFAR learned upsampler; 'bilinear' simply bilinearly upsamples the "
             "DINO latent features to the segmentation target resolution.",
    )

    # Paths (optional - models are loaded from HuggingFace/upsampler configs)
    parser.add_argument(
        "--backbone_torch_path",
        type=str,
        default=None,
        help="Path to DINOv3 PyTorch model checkpoint (optional, loads from HuggingFace if not provided)",
    )
    parser.add_argument(
        "--upsampler_torch_path",
        type=str,
        default=None,
        help="Path to upsampler PyTorch model checkpoint (optional, loads from upsampler config if not provided)",
    )

    # Training configuration
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (default: 1e-5)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size for training (default: 2)",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1,
        help="Batch size for validation (default: 1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader worker processes (default: 8). Set to 0 for single-process loading.",
    )
    parser.add_argument(
        "--img_size",
        type=parse_img_size,
        default=(320, 512),
        help="Input image size. Can be an integer (e.g., 512) for square resize+crop, "
             "or a tuple (H,W) for aspect-ratio preserving resize (e.g., '320,512') (default: 320,512)",
    )
    parser.add_argument(
        "--seg_scale",
        type=int,
        default=16,
        choices=[1, 2, 4, 8, 16],
        help="JAFAR upsampling factor from latent resolution (patch_size=16). "
             "E.g. --seg_scale 4 upsamples latent by 4x → 80x128 for 320x512 input. "
             "--seg_scale 16 = full resolution (default). "
             "Labels are downscaled with nearest interpolation to match.",
    )

    # Data configuration
    parser.add_argument(
        "--train_dataset",
        type=str,
        required=True,
        help="Path to training dataset root directory. Expected structure:\n"
             "  root/\n"
             "    images/          (PNG image files)\n"
             "    segmentations/   (PNG segmentation masks)\n"
             "    train_images.txt (list of image indices, one per line)",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Name of the training split file (without '_images.txt' suffix). "
             "E.g. 'train_kmeans_k200' will load train_kmeans_k200_images.txt (default: train)",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="val",
        help="Name of the validation split file (without '_images.txt' suffix) (default: val)",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default=None,
        help="Path to validation dataset root directory (optional). "
             "If not provided, uses train_dataset with val_images.txt split file. "
             "Expected structure same as train_dataset but with val_images.txt",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="simple",
        choices=["simple"],
        help="Dataset loader type: 'simple' uses basic loader (default: simple)",
    )

    # Output configuration - defaults to training folder/runs
    training_folder = Path(__file__).parent.resolve()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Base directory for this training run (default: {training_folder}/runs/<run_name>)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (overrides output_dir if set)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for logs (overrides output_dir if set)",
    )
    parser.add_argument(
        "--tb_dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs (overrides output_dir if set)",
    )

    # Other
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run sanity check (single batch)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    # Early stopping configuration
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of epochs with no improvement after which training will be stopped. "
             "If None, no early stopping is used",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="iou",
        choices=["iou", "accuracy"],
        help="Metric to monitor for early stopping",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.001,
        help="Minimum change in monitored metric to qualify as improvement",
    )
    parser.add_argument(
        "--reconstruction_weight",
        type=float,
        default=0.1,
        help="Weight for reconstruction loss term (default: 0.1)",
    )
    parser.add_argument(
        "--img_log_freq",
        type=int,
        default=2,
        help="Log wandb validation images every N epochs (default: 2)",
    )
    parser.add_argument(
        "--num_resblocks",
        type=int,
        default=0,
        help="Number of ResBlocks to refine upsampled features before MLP (default: 0)",
    )
    parser.add_argument(
        "--resblock_num_groups",
        type=int,
        default=8,
        help="GroupNorm num_groups for ResBlocks (default: 8)",
    )

    # Wandb configuration
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity (team/user name); defaults to your wandb account",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sem_map_seg",
        help="Wandb project name (default: sem_map)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated from seg_tag and timestamp)",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default=str(Path.home() / ".wandb"),
        help="Directory for wandb run data (default: ~/.wandb)",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Wandb group name for organizing related runs (e.g., experiment name)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build full seg_layer name from tag if not provided
    seg_tag = args.seg_tag
    seg_layer = args.seg_layer if args.seg_layer else f"{args.model_family}_sem_seg_{seg_tag}"

    # Create run-specific output directories (use tag for cleaner folder names)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    split_suffix = f"_{args.train_split}" if args.train_split != "train" else ""
    config_suffix = f"_{Path(args.seg_layer_config).stem}" if args.seg_layer_config else ""
    scale_suffix = f"_x{args.seg_scale}" if args.seg_scale != 16 else ""
    upsampler_suffix = f"_{args.upsampler_type}" if args.upsampler_type != "jafar" else ""
    run_name = f"seg_{seg_tag}{config_suffix}{split_suffix}{scale_suffix}{upsampler_suffix}_{timestamp}"

    # Determine base output directory
    training_folder = Path(__file__).parent.resolve()
    base_output_dir = Path(args.output_dir) if args.output_dir else training_folder / "runs" / run_name

    # Set up directories (use explicit paths if provided, otherwise use run folder)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else base_output_dir / "checkpoints"
    log_dir = Path(args.log_dir) if args.log_dir else base_output_dir / "logs"
    tb_dir = Path(args.tb_dir) if args.tb_dir else base_output_dir / "tb"

    # Create all directories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    terminal_console = Console()
    log_file = log_dir / f"train_seg_{seg_tag}_{timestamp}.log"
    file_console = Console(file=open(log_file, "w"))

    def log_print(*args, **kwargs):
        terminal_console.print(*args, **kwargs)
        file_console.print(*args, **kwargs)
        file_console.file.flush()

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(tb_dir))

    # Initialize wandb if enabled
    wandb_run = None
    if args.use_wandb:
        # Create wandb directory if it doesn't exist
        wandb_dir = Path(args.wandb_dir).expanduser()
        wandb_dir.mkdir(parents=True, exist_ok=True)

        wandb_run_name = args.wandb_run_name or run_name
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=wandb_run_name,
            group=args.wandb_group,
            dir=str(wandb_dir),
            config={
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "num_epochs": args.num_epochs,
                "train_batch_size": args.train_batch_size,
                "val_batch_size": args.val_batch_size,
                "model_family": args.model_family,
                "seg_layer": seg_layer,
                "seg_tag": seg_tag,
                "num_classes": None,  # Will be set after loading
                "feature_dim": None,  # Will be set after loading
                "patch_size": None,  # Will be set after loading
                "upsampler_factor": args.upsampler_factor,
                "upsampler_type": args.upsampler_type,
                "reconstruction_weight": args.reconstruction_weight,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_metric": args.early_stopping_metric,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "img_size": str(args.img_size),
                "seg_scale": args.seg_scale,
                "seed": args.seed,
                "num_resblocks": args.num_resblocks,
                "resblock_num_groups": args.resblock_num_groups,
            },
        )
        # Define custom x-axes so runs with different dataset sizes are comparable
        wandb.define_metric("train/step")
        wandb.define_metric("train/epoch")
        wandb.define_metric("val/epoch")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("train/epoch_*", step_metric="train/epoch")
        wandb.define_metric("val/*", step_metric="val/epoch")

        wandb_run = wandb.run
        log_print(f"[bold cyan]Wandb initialized: {args.wandb_project}/{wandb_run_name}[/bold cyan]")
        log_print(f"[bold cyan]Wandb directory: {wandb_dir}[/bold cyan]")

    # Start logging
    log_print(f"\n[bold blue]{'='*50}[/bold blue]")
    log_print(f"[bold blue]Starting training at {datetime.datetime.now()}[/bold blue]")
    log_print(f"[bold cyan]Run directory: {base_output_dir}[/bold cyan]")
    log_print(f"[bold cyan]Checkpoints: {checkpoint_dir}[/bold cyan]")
    log_print(f"[bold cyan]Logs: {log_dir}[/bold cyan]")
    log_print(f"[bold cyan]TensorBoard: {tb_dir}[/bold cyan]")
    log_print(f"[bold green]Configuration:[/bold green]")
    for key, value in vars(args).items():
        log_print(f"  {key}: {value}")

    device = get_device()
    log_print(f"[bold yellow]Using device: {device}[/bold yellow]")

    # Update wandb config with device and model info
    if args.use_wandb:
        wandb.config.update({
            "device": str(device),
        }, allow_val_change=True)

    # Load model configuration
    models_config = load_models_yaml()
    model_family = args.model_family
    # seg_layer is already set above

    # Get model parameters
    family_config = models_config["families"][model_family]
    patch_size = DINOV_CONFIGS[model_family]["patch_size"]
    skip_tokens = DINOV_CONFIGS[model_family]["skip_tokens"]
    feature_dim = family_config["backbone"]["feat_dims"]

    # Determine num_classes: from config file if provided, otherwise from models.yaml
    if args.seg_layer_config:
        # Load num_classes directly from config file
        from sem_map.perception.common import get_sem_map_root
        config_dir = get_sem_map_root() / "config" / "seg_layers"

        config_path = Path(args.seg_layer_config)
        if not config_path.is_absolute():
            # Relative path - assume it's a filename in config/seg_layers
            # Add .yaml extension if not present
            config_name = args.seg_layer_config
            if not config_name.endswith(('.yaml', '.yml')):
                config_name = f"{config_name}.yaml"
            config_path = config_dir / config_name

        if not config_path.exists():
            raise FileNotFoundError(
                f"Seg layer config file not found: {config_path}\n"
                f"Expected location: {config_dir}/ or provide full path"
            )

        seg_config = load_seg_layer_config(config_path.name, config_path.parent)
        num_classes = seg_config.get("num_classes")
        if num_classes is None:
            raise ValueError(
                f"Config file {config_path} does not contain 'num_classes' field"
            )
        log_print(f"[cyan]Loaded num_classes={num_classes} from config: {config_path}[/cyan]")

        # Load raw_to_train_id mapping if present in config
        raw_to_train_id = seg_config.get("raw_to_train_id")
        if raw_to_train_id:
            log_print(f"[cyan]Loaded raw_to_train_id mapping from config: {raw_to_train_id}[/cyan]")
        else:
            raw_to_train_id = None

        # Scan categories for a class named "background" to get its ID
        background_class_id = None
        for cat in seg_config.get("categories", []):
            if cat.get("name", "").lower() == "background":
                background_class_id = cat["id"]
                break

        # Extract categories for visualization
        categories = seg_config.get("categories", None)
    else:
        # Try to get from models.yaml
        num_classes = get_num_classes(model_family, seg_layer, models_config)
        if num_classes is None:
            raise ValueError(
                f"Could not determine num_classes for {model_family}/{seg_layer}. "
                f"Either add it to models.yaml or provide --seg_layer_config"
            )
        raw_to_train_id = None
        background_class_id = 0  # Default assumption when no config file
        categories = None

    log_print(f"[bold cyan]Model: {model_family}[/bold cyan]")
    log_print(f"[bold cyan]Segmentation layer: {seg_layer}[/bold cyan]")
    log_print(f"[bold cyan]Number of classes: {num_classes}[/bold cyan]")
    log_print(f"[bold cyan]Feature dimension: {feature_dim}[/bold cyan]")
    log_print(f"[bold cyan]Patch size: {patch_size}[/bold cyan]")
    if background_class_id is not None:
        log_print(f"[bold cyan]Background class ID: {background_class_id} (excluded from reconstruction loss)[/bold cyan]")
    else:
        log_print(f"[bold cyan]No background class found — reconstruction loss uses all valid pixels[/bold cyan]")

    # Update wandb config with model info
    if args.use_wandb:
        wandb.config.update({
            "num_classes": num_classes,
            "feature_dim": feature_dim,
            "patch_size": patch_size,
        }, allow_val_change=True)

    log_print(f"[cyan]Loading backbone (PyTorch) for variant: {model_family}[/cyan]")

    # Load PyTorch models
    backbone_model, _ = load_dinovx_torch(model_family)
    backbone_model.to(device)
    backbone_model.eval()
    backbone_model.requires_grad_(False)

    if args.upsampler_type == "jafar":
        # Get upsampler backbone name from models.yaml
        upsampler_backbone_name = family_config["upsamplers"]["jafar"]["torch_variant"]
        log_print(f"[cyan]Loading upsampler (PyTorch) with backbone: {upsampler_backbone_name}[/cyan]")
        upsampler_model, _ = load_jafar_torch(backbone_name=upsampler_backbone_name)
        upsampler_model.to(device)
        upsampler_model.eval()
        upsampler_model.requires_grad_(False)
    else:
        log_print("[cyan]Using bilinear feature upsampling (no JAFAR upsampler loaded)[/cyan]")
        upsampler_model = None

    # Setup data loaders
    class SimpleConfig:
        def __init__(self):
            self.img_size = args.img_size
            self.seg_scale = args.seg_scale
            # Compute segmentation target size: latent * seg_scale
            # latent = img_size / patch_size (16), so target = img_size * seg_scale / 16
            patch_size = 16
            if isinstance(args.img_size, tuple):
                self.target_size = (args.img_size[0] * args.seg_scale // patch_size,
                                    args.img_size[1] * args.seg_scale // patch_size)
            else:
                self.target_size = args.img_size * args.seg_scale // patch_size
            self.sanity = args.sanity
            self.num_epochs = args.num_epochs
            self.reconstruction_weight = args.reconstruction_weight
            self.num_resblocks = args.num_resblocks
            self.resblock_num_groups = args.resblock_num_groups
            self.background_class_id = background_class_id
            self.upsampler_type = args.upsampler_type

    cfg = SimpleConfig()

    if args.seg_scale < 16:
        log_print(f"[bold yellow]Segmentation scale: x{args.seg_scale} → target resolution {cfg.target_size}[/bold yellow]")

    # Data loading implementation
    val_root = args.val_dataset or args.train_dataset
    log_print(f"[cyan]Train dataset: {args.train_dataset}[/cyan]")
    log_print(f"[cyan]Val dataset: {val_root} (split: {args.val_split})[/cyan]")

    # Load datasets using SimpleSegmentationDataset
    log_print("[cyan]Using SimpleSegmentationDataset loader[/cyan]")
    try:
        image_transform, label_transform = create_segmentation_transforms(args.img_size)

        # Use raw_to_train_id from seg_layer_config if available
        # Otherwise, SimpleSegmentationDataset will auto-detect (subtract 1 if labels start from 1)
        if raw_to_train_id:
            log_print(f"[cyan]Using label remapping from seg_layer_config: {raw_to_train_id}[/cyan]")
        else:
            log_print(f"[cyan]Auto-detecting label remapping (will subtract 1 if labels start from 1)[/cyan]")

        train_dataset = SimpleSegmentationDataset(
            root=args.train_dataset,
            split=args.train_split,
            transform=image_transform,
            target_transform=label_transform,
            img_size=args.img_size,
            raw_to_train_id=raw_to_train_id,
        )
        val_dataset = SimpleSegmentationDataset(
            root=args.val_dataset or args.train_dataset,
            split=args.val_split,
            transform=image_transform,
            target_transform=label_transform,
            img_size=args.img_size,
            raw_to_train_id=raw_to_train_id,
        )

        # Create dataloaders with explicit batch sizes
        train_batch_size = args.train_batch_size
        val_batch_size = args.val_batch_size
        num_workers = args.num_workers
        train_loader, val_loader = _create_dataloaders(
            train_dataset, val_dataset, train_batch_size, val_batch_size, num_workers
        )

        log_print(f"[green]Loaded {len(train_dataset)} training samples[/green]")
        log_print(f"[green]Loaded {len(val_dataset)} validation samples[/green]")
        log_print(f"[cyan]Train DataLoader: num_workers={train_loader.num_workers}, persistent_workers={train_loader.persistent_workers}[/cyan]")
        log_print(f"[cyan]Val DataLoader: num_workers={val_loader.num_workers}, persistent_workers={val_loader.persistent_workers}[/cyan]")

        # Log dataset info to wandb
        if args.use_wandb:
            wandb.config.update({
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
                "train_dataset_path": args.train_dataset,
                "val_dataset_path": val_root,
            }, allow_val_change=True)
    except Exception as e:
        log_print(
            f"[bold red]ERROR: Failed to load dataset: {e}[/bold red]\n"
            "[yellow]Please ensure your dataset follows the expected structure:[/yellow]\n"
            "  root/\n"
            "    images/          (PNG files: 000000.png, 000001.png, ...)\n"
            "    segmentations/   (PNG mask files with same names)\n"
            "    train_images.txt (one index per line: 000000, 000001, ...)\n"
            "    val_images.txt   (one index per line)"
        )
        file_console.file.close()
        writer.close()
        return

    # Setup trainer
    trainer = SegmentationTrainer(
        backbone_model=backbone_model,
        upsampler_model=upsampler_model,
        device=device,
        num_classes=num_classes,
        feature_dim=feature_dim,
        patch_size=patch_size,
        skip_tokens=skip_tokens,
        writer=writer,
        console=file_console,
        config=cfg,
        use_wandb=args.use_wandb,
        categories=categories,
    )

    trainer.set_optimizer(
        lr=args.lr, weight_decay=args.weight_decay, num_epochs=args.num_epochs
    )

    # Checkpoint path (use tag for cleaner filenames)
    checkpoint_path = checkpoint_dir / f"{seg_tag}.pth"

    # Training loop
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[yellow]Loss: {task.fields[loss]:.6f}"),
        TextColumn("[green]Step: {task.fields[step]:.6e}"),
        console=file_console,
    )

    start_time = datetime.datetime.now()

    # Initialize early stopping if patience is provided
    early_stopping = None
    if args.early_stopping_patience is not None:
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            metric_name=args.early_stopping_metric,
            mode="max"  # Both iou and accuracy should be maximized
        )
        log_print(
            f"[cyan]Early stopping enabled: patience={args.early_stopping_patience}, "
            f"metric={args.early_stopping_metric}, min_delta={args.early_stopping_min_delta}[/cyan]"
        )

    # Pick 5 fixed validation indices for wandb image logging
    num_vis_samples = min(5, len(val_dataset))
    rng = np.random.RandomState(args.seed)
    vis_indices = rng.choice(len(val_dataset), size=num_vis_samples, replace=False).tolist()

    with progress:
        log_print(f"[yellow]Training for {args.num_epochs} epochs[/yellow]\n")
        best_metric = 0.0
        best_checkpoint_path = None
        best_epoch = 0

        for epoch in range(args.num_epochs):
            trainer.train(train_loader, progress, epoch, start_time)

            if val_loader:
                metrics = trainer.evaluate(val_loader, epoch)
                if epoch % args.img_log_freq == 0 or epoch == args.num_epochs - 1:
                    trainer.log_wandb_images(val_dataset, vis_indices, epoch)

                # Determine which metric to use for best model selection
                # Default to "iou" if available, otherwise use "accuracy"
                if early_stopping is not None:
                    metric_for_best = args.early_stopping_metric
                else:
                    metric_for_best = "iou" if "iou" in metrics else "accuracy"

                current_metric = metrics.get(metric_for_best, 0.0)

                # Check if this is a new best model
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch + 1

                    # Save best checkpoint
                    best_checkpoint_path = checkpoint_dir / f"{seg_tag}.best.pth"
                    trainer.save_checkpoint(best_checkpoint_path)

                    log_print(
                        f"[bold green]New best {metric_for_best}: {best_metric:.4f} at epoch {best_epoch}[/bold green]"
                    )

                    # Upload best model to wandb as artifact
                    if args.use_wandb:
                        wandb.run.summary[f"best_{metric_for_best}"] = best_metric
                        wandb.run.summary["best_epoch"] = best_epoch
                        wandb.run.summary["best_checkpoint"] = str(best_checkpoint_path)

                        # Create and log artifact for best model
                        artifact = wandb.Artifact(
                            name=f"best_model_{seg_tag}",
                            type="model",
                            description=f"Best model checkpoint with {metric_for_best}={best_metric:.4f} at epoch {best_epoch}",
                            metadata={
                                "epoch": best_epoch,
                                metric_for_best: best_metric,
                                "accuracy": metrics.get("accuracy", 0.0),
                                "iou": metrics.get("iou", 0.0),
                                "seg_tag": seg_tag,
                                "seg_layer": seg_layer,
                            }
                        )
                        artifact.add_file(str(best_checkpoint_path))
                        wandb.log_artifact(artifact)
                        log_print(f"[bold cyan]Uploaded best model to wandb: {best_checkpoint_path}[/bold cyan]")

                # Check early stopping
                if early_stopping is not None:
                    # Check if should stop
                    if early_stopping(current_metric):
                        log_print(
                            f"[bold yellow]Early stopping triggered after {epoch+1} epochs[/bold yellow]\n"
                            f"[yellow]Best {args.early_stopping_metric}: {early_stopping.best_score:.4f}[/yellow]\n"
                            f"[yellow]No improvement for {early_stopping.patience} epochs[/yellow]"
                        )
                        if args.use_wandb:
                            wandb.run.summary["early_stopped"] = True
                            wandb.run.summary["early_stop_epoch"] = epoch + 1
                        break
                    else:
                        log_print(
                            f"[cyan]Early stopping: {early_stopping.counter}/{early_stopping.patience} "
                            f"(best {args.early_stopping_metric}: {early_stopping.best_score:.4f})[/cyan]"
                        )

            if args.sanity:
                break

        # Save final checkpoint
        trainer.save_checkpoint(checkpoint_path)

        # Log final checkpoint path to wandb
        if args.use_wandb:
            wandb.run.summary["final_checkpoint"] = str(checkpoint_path)
            wandb.run.summary["total_epochs"] = epoch + 1

            # If we have a best checkpoint, log summary
            if best_checkpoint_path and best_checkpoint_path.exists():
                metric_name = args.early_stopping_metric if early_stopping is not None else "iou"
                log_print(f"[bold cyan]Best model saved at: {best_checkpoint_path}[/bold cyan]")
                log_print(f"[bold cyan]Best {metric_name}: {best_metric:.4f} at epoch {best_epoch}[/bold cyan]")

    # Finish wandb run before closing files
    if args.use_wandb:
        wandb.finish()
        log_print(f"[bold cyan]Wandb run finished[/bold cyan]")

    file_console.file.close()
    writer.close()


if __name__ == "__main__":
    main()
