"""
Prediction visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional

import torch

from ..data.dataset import LABEL_COLORS, CLASS_NAMES, class_id_to_rgb


def visualize_prediction(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "",
):
    """Visualize image, ground truth, and prediction side by side.
    
    Args:
        image: (H, W, 3) RGB image (uint8 or float [0,1])
        gt_mask: (H, W) ground truth class IDs
        pred_mask: (H, W) predicted class IDs
        save_path: optional path to save figure
        title: figure title
    """
    gt_rgb = class_id_to_rgb(gt_mask)
    pred_rgb = class_id_to_rgb(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image if image.max() > 1 else (image * 255).astype(np.uint8))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=np.array(color) / 255, label=name)
        for name, color in zip(CLASS_NAMES, LABEL_COLORS)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()
