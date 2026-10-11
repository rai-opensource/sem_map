# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Shared dataset utilities for segmentation training scripts.

Provides:
- SimpleSegmentationDataset: Standard directory-based segmentation dataset
- _create_dataloaders: Train/val DataLoader factory
- parse_img_size: Argparse type for image size (int or H,W tuple)
"""

import argparse
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms.functional as TF


class SimpleSegmentationDataset(torch.utils.data.Dataset):
    """Simple dataset loader for segmentation datasets with standard directory structure."""

    def __init__(self, root, split="train", transform=None, target_transform=None, img_size: int | tuple[int, int] = (320, 512), raw_to_train_id: dict[int, int] | None = None, ignore_label: int = 255):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
        self.raw_to_train_id = raw_to_train_id  # Mapping from raw label IDs to training IDs (e.g., {1: 0, 2: 1, 3: 2})
        self.ignore_label = ignore_label

        images_dir = self.root / "images"
        seg_dir = self.root / "segmentations"
        split_file = self.root / f"{split}_images.txt"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            indices = [line.strip() for line in f if line.strip()]

        self.image_files = [images_dir / f"{idx}.png" for idx in indices]
        self.label_files = [seg_dir / f"{idx}.png" for idx in indices]

        # Verify files exist
        for img_p, lab_p in zip(self.image_files, self.label_files):
            if not img_p.exists():
                raise FileNotFoundError(f"Image missing: {img_p}")
            if not lab_p.exists():
                raise FileNotFoundError(f"Label missing: {lab_p}")

        # Cache label mapping mode to avoid per-sample min() scans
        self._labels_need_subtract = None
        if self.raw_to_train_id is None:
            self._labels_need_subtract = self._probe_labels_need_subtract()

    def _probe_labels_need_subtract(self) -> bool:
        """Probe first few label files to determine if labels start from 1 (need subtract)."""
        from PIL import Image
        import torchvision.transforms.functional as TF
        for i in range(min(5, len(self.label_files))):
            label_img = Image.open(self.label_files[i])
            label = TF.to_tensor(label_img)
            label = (label * 255).to(torch.uint8).squeeze(0)
            mask = label != self.ignore_label
            valid = label[mask]
            if valid.numel() > 0:
                return valid.min().item() > 0
        return False

    def __len__(self):
        return len(self.image_files)

    def _map_raw_labels(self, label_tensor):
        """Map raw label IDs to training IDs using torch operations.

        If raw_to_train_id is provided, use explicit mapping.
        Otherwise, if labels start from 1 (not 0), subtract 1 (decision cached at init).
        """
        if self.raw_to_train_id is not None:
            # Use explicit mapping if provided. Pixels whose raw value is not a
            # key in raw_to_train_id remain ignore_label (set by full_like).
            # Do NOT force `pixel == ignore_label → ignore_label` afterwards:
            # that would clobber legitimate mappings whose raw key happens to
            # equal ignore_label (e.g. binary masks using 255 as the positive class).
            mapped = torch.full_like(label_tensor, self.ignore_label, dtype=torch.long)
            for raw_id, train_id in self.raw_to_train_id.items():
                mapped[label_tensor == raw_id] = train_id
            return mapped

        if self._labels_need_subtract:
            mask = label_tensor != self.ignore_label
            mapped = label_tensor.clone()
            mapped[mask] = mapped[mask] - 1
            return mapped

        return label_tensor

    def __getitem__(self, idx):
        # Load images
        image = Image.open(self.image_files[idx]).convert("RGB")
        label_img = Image.open(self.label_files[idx])

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label_img = self.target_transform(label_img)

        # Convert label to tensor if not already
        if isinstance(label_img, Image.Image):
            label = TF.to_tensor(label_img)  # Returns (C, H, W) in [0, 1]
            label = (label * 255).to(torch.uint8).squeeze(0)  # (H, W) uint8
        else:
            # Already a tensor from transform
            label = label_img
            if label.ndim == 3 and label.shape[0] == 1:
                label = label.squeeze(0)
            label = label.to(torch.uint8)

        # Apply label remapping (stays in torch)
        label = self._map_raw_labels(label)
        label = label.long()  # Convert to long for cross_entropy

        return {"image": image, "label": label}


def _create_dataloaders(train_dataset, val_dataset, batch_size, val_batch_size, num_workers=8):
    """Create train and validation dataloaders."""
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        persistent_workers=True if num_workers > 0 else False,
    )
    return train_loader, val_loader


def parse_img_size(value: str) -> int | tuple[int, int]:
    """Parse img_size argument: supports int or comma-separated tuple (H,W).

    Examples:
        "512" -> 512
        "320,512" -> (320, 512)
        "512,320" -> (512, 320)
    """
    value = value.strip()
    if ',' in value:
        # Parse as tuple (H, W)
        parts = value.split(',')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"img_size tuple must have 2 values (H,W), got: {value}")
        try:
            h, w = int(parts[0].strip()), int(parts[1].strip())
            return (h, w)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"img_size tuple values must be integers, got: {value}") from e
    else:
        # Parse as int
        try:
            return int(value)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"img_size must be an integer or tuple (H,W), got: {value}") from e
