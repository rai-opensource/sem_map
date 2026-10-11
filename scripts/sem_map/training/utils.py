# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Shared training utilities for segmentation training scripts.

Provides:
- EarlyStopping: Stop training when validation metric stops improving
- create_val_figure: Create 4-panel validation visualization figure
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""

    def __init__(self, patience=5, min_delta=0.0, metric_name="iou", mode="max"):
        """
        Args:
            patience: Number of epochs with no improvement after which to stop
            min_delta: Minimum change to qualify as improvement
            metric_name: Name of metric to monitor
            mode: "max" for metrics to maximize (iou, accuracy), "min" for metrics to minimize (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        """Check if training should stop based on current metric score."""
        if self.best_score is None:
            self.best_score = current_score
            return False

        # Check if there's improvement
        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def create_val_figure(
    raw_img: np.ndarray,
    gt_colored: np.ndarray,
    pred_colored: np.ndarray,
    confidence: np.ndarray,
    confidence_title: str = "Confidence",
) -> plt.Figure:
    """Create a 4-panel validation visualization figure.

    Args:
        raw_img: Unnormalized RGB image, shape (H, W, 3), uint8
        gt_colored: Colorized ground truth labels, shape (H, W, 3), uint8
        pred_colored: Colorized predicted labels, shape (H, W, 3), uint8
        confidence: Confidence map, shape (H, W), values in [0, 1]
        confidence_title: Title for the confidence panel

    Returns:
        matplotlib Figure with 4 panels: Input, Ground Truth, Prediction, Confidence
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(raw_img)
    axes[0].set_title("Input")
    axes[1].imshow(gt_colored)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_colored)
    axes[2].set_title("Prediction")
    axes[3].imshow(confidence, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title(confidence_title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
