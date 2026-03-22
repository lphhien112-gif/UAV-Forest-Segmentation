"""
Segmentation metrics: mIoU, pixel accuracy.
"""

import torch
import numpy as np
from typing import Dict, List


class SegmentationMetrics:
    """Accumulates predictions to compute mIoU and pixel accuracy.
    
    Usage:
        metrics = SegmentationMetrics(num_classes=11)
        for batch in dataloader:
            preds = model(images).argmax(dim=1)
            metrics.update(preds, labels)
        results = metrics.compute()
    """

    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        """Reset accumulated metrics."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update confusion matrix with a batch of predictions.
        
        Args:
            preds: (B, H, W) predicted class indices
            targets: (B, H, W) ground truth class indices
        """
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Filter valid labels
        valid = (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]

        # Update confusion matrix
        indices = targets * self.num_classes + preds
        cm = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated confusion matrix."""
        cm = self.confusion_matrix

        # Per-class IoU
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = np.where(union > 0, intersection / union, 0.0)

        # Per-class accuracy (recall)
        class_total = cm.sum(axis=1)
        class_acc = np.where(class_total > 0, intersection / class_total, 0.0)

        # Overall pixel accuracy
        pixel_acc = intersection.sum() / cm.sum() if cm.sum() > 0 else 0.0

        # Mean IoU (excluding classes with no GT pixels)
        valid_classes = union > 0
        miou = iou[valid_classes].mean() if valid_classes.any() else 0.0

        result = {
            "miou": float(miou),
            "pixel_accuracy": float(pixel_acc),
            "mean_class_accuracy": float(class_acc[valid_classes].mean()) if valid_classes.any() else 0.0,
        }

        # Per-class IoU
        for i, name in enumerate(self.class_names):
            result[f"iou_{name}"] = float(iou[i])

        return result

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the raw confusion matrix."""
        return self.confusion_matrix.copy()
