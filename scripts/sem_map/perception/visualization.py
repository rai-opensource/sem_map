# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#
# Portions of this file are copied from JAFAR (https://github.com/PaulCouairon/JAFAR),
# licensed under the Apache License, Version 2.0. See the NOTICE file for the
# list, and licenses/Apache-2.0.txt for the licence text.
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.decomposition import PCA



# ----------------------- Segmentation visualization helpers -----------------------

def unnormalize_image(image_tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Unnormalize an image tensor from ImageNet normalization.

    Args:
        image_tensor: Tensor of shape (C, H, W) or (B, C, H, W)
        mean: Mean values for normalization (e.g., IMAGENET_MEAN)
        std: Std values for normalization (e.g., IMAGENET_STD)

    Returns:
        Unnormalized image tensor (values in [0, 1])
    """
    img = image_tensor.clone()

    if img.dim() == 4:
        # Batch of images
        for i in range(img.shape[0]):
            for c in range(3):
                img[i, c] = img[i, c] * std[c] + mean[c]
    else:
        # Single image
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]

    return img.clamp(0, 1)


def label_to_color(label: np.ndarray, categories: List[Dict]) -> np.ndarray:
    """
    Convert label mask to RGB color image.

    Args:
        label: Label mask (H, W) with class indices
        categories: List of category dicts with 'id' and 'color' keys

    Returns:
        RGB image (H, W, 3) with values in [0, 255]
    """
    H, W = label.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for cat in categories:
        mask = label == cat["id"]
        rgb[mask] = cat["color"]

    return rgb


# ----------------------- Normalization helpers -----------------------

# Copied unmodified from JAFAR, Apache-2.0. See NOTICE.
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        if len(image2.shape) == 4:
            # batched
            image2 = image2.permute(1, 0, 2, 3)
        for t, m, s in zip(image2, self.mean, self.std):
            t = t * s + m  # .mul_(s).add_(m)
        return image2.permute(1, 0, 2, 3)


# Import constants from common
from .common import IMAGENET_MEAN, IMAGENET_STD

norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
unnorm = UnNormalize(IMAGENET_MEAN, IMAGENET_STD)

# Import unnormalize from common
from .common import unnormalize


# ----------------------- PCA helpers -----------------------

# TorchPCA and pca() below are copied unmodified from JAFAR, Apache-2.0.
# See NOTICE.
class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="area")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca


# --------------------------- PCA cache helpers ---------------------------
def save_torch_pca_fit(pca_fit, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean_": pca_fit.mean_.cpu(),
        "components_": pca_fit.components_.cpu(),
        "singular_values_": pca_fit.singular_values_.cpu(),
        "n_components": pca_fit.n_components,
    }
    torch.save(payload, str(path))


def load_torch_pca_fit(path: Path):
    data = torch.load(str(path), map_location="cpu")
    fit = TorchPCA(n_components=int(data["n_components"]))
    fit.mean_ = data["mean_"]
    fit.components_ = data["components_"]
    fit.singular_values_ = data["singular_values_"]
    return fit
