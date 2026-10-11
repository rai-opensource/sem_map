# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import os

import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import importlib


# --------------------------- Device helpers ---------------------------


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------- Image helpers ---------------------------

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def prepare_image(
    image: Image.Image,
    latent_w: int,
    latent_h: int,
    patch_size: int,
    device: torch.device | None = None,
):
    """Resize+normalize PIL image and return unnormalized image as well.

    Returns (unorm_img, image_batch, input_h, input_w)
    - unorm_img: Tensor (H, W, 3) in [0,1]
    - image_batch: Tensor (1, 3, H, W) normalized
    """
    if device is None:
        device = get_device()

    input_w = latent_w * patch_size
    input_h = latent_h * patch_size

    mean = torch.tensor(IMAGENET_MEAN, device=device)
    std = torch.tensor(IMAGENET_STD, device=device)
    transform = T.Compose(
        [
            T.Resize((input_h, input_w)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    image_batch = transform(image).unsqueeze(0).to(device)
    unorm_img_batch = unnormalize(image_batch, mean, std)
    return unorm_img_batch[0].permute(1, 2, 0), image_batch, input_h, input_w


def unnormalize(tensor, mean, std):
    """
    Unnormalizes a 4D tensor of shape (B, 3, H, W).
    """
    assert len(tensor.shape) == 4
    device = tensor.device
    dtype = tensor.dtype
    mean_t = (
        mean
        if isinstance(mean, torch.Tensor)
        else torch.tensor(mean, device=device, dtype=dtype)
    )
    std_t = (
        std
        if isinstance(std, torch.Tensor)
        else torch.tensor(std, device=device, dtype=dtype)
    )
    mean_t = mean_t.view(1, 3, 1, 1)
    std_t = std_t.view(1, 3, 1, 1)
    return tensor * std_t + mean_t  # Unnormalize


def prepare_image_trt(
    image_chw: torch.Tensor,
    latent_w: int,
    latent_h: int,
    patch_size: int,
    device: torch.device | None = None,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
):
    """
    Resize+normalize a CHW tensor and return unnormalized image as well.

    Args:
        image_chw: Tensor (3, H, W) or (1, 3, H, W), uint8 or float in [0,1] or [0,255]
        latent_w, latent_h, patch_size: target latent grid and patch size
        device: optional device (defaults to best device)

    Returns (unorm_img, image_batch, input_h, input_w)
        - unorm_img: Tensor (H, W, 3) in [0,1]
        - image_batch: Tensor (1, 3, H, W) normalized
    """
    if device is None:
        device = get_device()

    input_w = latent_w * patch_size
    input_h = latent_h * patch_size

    x = image_chw.to(device, non_blocking=True)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    # Convert to float in [0,1]
    if x.dtype == torch.uint8:
        x = x.float().mul_(1.0 / 255.0)
    elif x.dtype in (torch.float16, torch.float32, torch.bfloat16):
        pass
    else:
        x = x.float()

    # Resize to target resolution
    x = torch.nn.functional.interpolate(
        x, size=(input_h, input_w), mode="bilinear", align_corners=False
    )

    # Use provided mean/std if available; otherwise create once per call
    if mean is None:
        mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
    else:
        mean = mean.view(1, 3, 1, 1).to(device=device, dtype=x.dtype)
    if std is None:
        std = torch.tensor(IMAGENET_STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    else:
        std = std.view(1, 3, 1, 1).to(device=device, dtype=x.dtype)

    # Normalized batch (1,3,H,W)
    x_norm = (x - mean) / std

    # Unnormalized image: we already have x in [0,1] after resize; avoid recomputing
    unorm = x[0].permute(1, 2, 0)
    return unorm, x_norm, input_h, input_w


# --------------------------- Path helpers ---------------------------

def get_sem_map_root() -> Path:
    """Return cached sem_map package root path."""
    _sem_map_mod = importlib.import_module("sem_map")
    return Path(_sem_map_mod.__file__).resolve().parents[2]


def get_workspace_root() -> Path:
    """Return the repo root path (override with SEM_MAP_WS_ROOT).

    The sem_map repo is self-contained, so the "workspace" defaults to the
    repo root itself. Set SEM_MAP_WS_ROOT to point outputs/data elsewhere.
    """
    env = os.environ.get("SEM_MAP_WS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return get_sem_map_root()


def get_pca_dir() -> Path:
    """Return the PCA artifacts directory inside the package."""
    return get_sem_map_root() / "artifacts" / "pca"


# --------------------------- Transform helpers ---------------------------

def create_image_transform(img_size: int | tuple[int, int] | list[int], mean: list[float] | None = None, std: list[float] | None = None):
    """Create image transform for training datasets.

    Args:
        img_size: Target image size. If int, creates square resize+crop.
                  If tuple/list (H, W), preserves aspect ratio with resize only (no crop).
        mean: Normalization mean (defaults to IMAGENET_MEAN)
        std: Normalization std (defaults to IMAGENET_STD)

    Returns:
        torchvision.transforms.Compose transform
    """
    from torchvision.transforms.functional import InterpolationMode

    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    # Support both int (square) and tuple/list (aspect ratio preserving) for img_size
    if isinstance(img_size, (list, tuple)):
        # Tuple/List: Use as (H, W) for aspect-ratio preserving resize, no crop
        img_h, img_w = tuple(img_size)
        return T.Compose([
            T.Resize((img_h, img_w), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        # Single int: Keep current square behavior (resize + center crop)
        return T.Compose([
            T.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


class LabelTransform:
    """Label transform for segmentation masks (pickleable for multiprocessing)."""

    def __init__(self, img_size: int | tuple[int, int] | list[int]):
        self.img_size = img_size

    def __call__(self, label: Image.Image) -> torch.Tensor:
        """Convert PIL label to tensor (H, W) with integer labels."""
        import numpy as np
        # Support both int (square) and tuple/list (aspect ratio preserving) for img_size
        if isinstance(self.img_size, (list, tuple)):
            # Tuple/List: Use as (H, W) for aspect-ratio preserving resize, no crop
            # IMPORTANT: PIL.Image.resize expects (width, height), so swap the order!
            img_h, img_w = tuple(self.img_size)
            label = label.resize((img_w, img_h), resample=Image.NEAREST)
        else:
            # Single int: Keep current square behavior (resize + center crop)
            label = label.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        label_array = np.array(label, dtype=np.int64)
        return torch.from_numpy(label_array).long()


def create_label_transform(img_size: int | tuple[int, int] | list[int]):
    """Create label transform for segmentation masks.

    Args:
        img_size: Target image size. If int, creates square resize.
                  If tuple/list (H, W), preserves aspect ratio with resize only (no crop).

    Returns:
        Callable that converts PIL Image to (H, W) tensor with integer labels
    """
    return LabelTransform(img_size)


def create_segmentation_transforms(img_size: int | tuple[int, int] | list[int], mean: list[float] | None = None, std: list[float] | None = None):
    """Create both image and label transforms for segmentation training.

    Args:
        img_size: Target image size. If int, creates square resize+crop.
                  If tuple/list (H, W), preserves aspect ratio with resize only (no crop).
        mean: Normalization mean (defaults to IMAGENET_MEAN)
        std: Normalization std (defaults to IMAGENET_STD)

    Returns:
        Tuple of (image_transform, label_transform)
    """
    return create_image_transform(img_size, mean, std), create_label_transform(img_size)
