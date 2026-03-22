"""
Evaluation pipeline for Forest Semantic Segmentation.

Runs inference on test set, computes metrics, and generates reports.

Usage:
    evaluator = Evaluator(model, test_loader, device="cuda")
    results = evaluator.evaluate()
    evaluator.save_report("outputs/eval_report.json")
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

from ..training.metrics import SegmentationMetrics
from ..data.dataset import CLASS_NAMES, NUM_CLASSES, class_id_to_rgb
from .visualize import visualize_prediction


class Evaluator:
    """Evaluation pipeline for segmentation models.

    Args:
        model: Trained segmentation model
        test_loader: Test DataLoader
        device: torch device
        num_classes: Number of semantic classes
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda",
        num_classes: int = NUM_CLASSES,
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.metrics = SegmentationMetrics(num_classes, CLASS_NAMES)
        self.results: Dict = {}

    @torch.no_grad()
    def evaluate(self, use_amp: bool = True) -> Dict[str, float]:
        """Run full evaluation on test set.

        Returns:
            Dictionary with mIoU, pixel accuracy, per-class IoU, etc.
        """
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = 0
        inference_times = []

        pbar = tqdm(self.test_loader, desc="Evaluating")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Inference with timing
            start_t = time.time()
            with autocast(device_type="cuda", enabled=use_amp):
                outputs = self.model(images)
            torch.cuda.synchronize() if self.device == "cuda" else None
            inference_times.append(time.time() - start_t)

            preds = outputs.argmax(dim=1)
            self.metrics.update(preds, masks)
            num_batches += 1

        # Compute metrics
        self.results = self.metrics.compute()

        # Add timing info
        self.results["avg_inference_time_ms"] = np.mean(inference_times) * 1000
        self.results["total_samples"] = len(self.test_loader.dataset)
        self.results["fps"] = len(self.test_loader.dataset) / sum(inference_times)

        return self.results

    def evaluate_per_sequence(
        self,
        dataset_root: str,
        sequences: List[str],
        transform,
        batch_size: int = 8,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model separately on each sequence (for altitude/pitch analysis).

        Returns:
            Dictionary mapping sequence name to its metrics
        """
        from ..data.dataset import ForestDataset

        per_seq_results = {}

        for seq_name in sequences:
            dataset = ForestDataset(
                root=dataset_root,
                sequences=[seq_name],
                transform=transform,
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            self.metrics.reset()
            self.model.eval()

            with torch.no_grad():
                for images, masks in loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    outputs = self.model(images)
                    preds = outputs.argmax(dim=1)
                    self.metrics.update(preds, masks)

            per_seq_results[seq_name] = self.metrics.compute()
            miou = per_seq_results[seq_name]["miou"]
            print(f"  {seq_name}: mIoU = {miou:.4f}")

        return per_seq_results

    def save_predictions(
        self,
        output_dir: str = "outputs/predictions",
        num_samples: int = 20,
        denormalize_mean: tuple = (0.485, 0.456, 0.406),
        denormalize_std: tuple = (0.229, 0.224, 0.225),
    ):
        """Save visual predictions for qualitative analysis."""
        self.model.eval()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        count = 0
        with torch.no_grad():
            for images, masks in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                masks_np = masks.numpy()

                for i in range(images.shape[0]):
                    if count >= num_samples:
                        return

                    # Denormalize image for visualization
                    img = images[i].cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
                    mean = np.array(denormalize_mean)
                    std = np.array(denormalize_std)
                    img = (img * std + mean) * 255
                    img = img.clip(0, 255).astype(np.uint8)

                    visualize_prediction(
                        image=img,
                        gt_mask=masks_np[i],
                        pred_mask=preds[i],
                        save_path=str(out_path / f"pred_{count:04d}.png"),
                        title=f"Sample {count}",
                    )
                    count += 1

        print(f"Saved {count} predictions to {output_dir}")

    def save_report(self, path: str = "outputs/eval_report.json"):
        """Save evaluation results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Report saved to {path}")

    def print_results(self):
        """Print formatted evaluation results."""
        if not self.results:
            print("No results yet. Run evaluate() first.")
            return

        print("\n" + "=" * 55)
        print("  EVALUATION RESULTS")
        print("=" * 55)
        print(f"  mIoU:             {self.results.get('miou', 0):.4f}")
        print(f"  Pixel Accuracy:   {self.results.get('pixel_accuracy', 0):.4f}")
        print(f"  Mean Class Acc:   {self.results.get('mean_class_accuracy', 0):.4f}")
        print(f"  Avg Inference:    {self.results.get('avg_inference_time_ms', 0):.1f} ms")
        print(f"  FPS:              {self.results.get('fps', 0):.1f}")
        print("-" * 55)
        print("  Per-Class IoU:")
        for name in CLASS_NAMES:
            iou = self.results.get(f"iou_{name}", 0)
            bar = "█" * int(iou * 30)
            print(f"    {name:20s} {iou:.4f} {bar}")
        print("=" * 55)
