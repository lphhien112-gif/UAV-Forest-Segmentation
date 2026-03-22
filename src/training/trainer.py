"""
Training loop for semantic segmentation.
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import SegmentationMetrics
from ..data.dataset import CLASS_NAMES


class Trainer:
    """Semantic segmentation trainer.

    Args:
        model: Segmentation model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: LR scheduler (optional)
        config: Training configuration dict
        device: torch device
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        config: dict = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.device = device

        # AMP
        self.use_amp = self.config.get("amp", True)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.accumulation_steps = self.config.get("accumulation_steps", 1)

        # Logging
        log_dir = self.config.get("log_dir", "outputs/logs")
        self.writer = SummaryWriter(log_dir=log_dir)

        # Checkpointing
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "outputs/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = self.config.get("save_top_k", 3)

        # Early stopping
        es = self.config.get("early_stopping", {})
        self.patience = es.get("patience", 15)
        self.best_metric = 0.0
        self.epochs_without_improvement = 0

        # Metrics
        num_classes = self.config.get("num_classes", 11)
        self.metrics = SegmentationMetrics(num_classes, CLASS_NAMES)

        # History
        self.global_step = 0
        self.best_checkpoints = []  # (metric, path) sorted ascending

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        self.optimizer.zero_grad()

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            with autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            self.global_step += 1

            # Log
            pbar.set_postfix({"loss": f"{loss.item() * self.accumulation_steps:.4f}"})
            if self.global_step % self.config.get("log_every_n_steps", 10) == 0:
                self.writer.add_scalar("train/loss", loss.item() * self.accumulation_steps, self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate and compute metrics. Returns metrics dict."""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            with autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            total_loss += loss.item()
            num_batches += 1

            preds = outputs.argmax(dim=1)
            self.metrics.update(preds, masks)

        results = self.metrics.compute()
        results["val_loss"] = total_loss / max(num_batches, 1)

        # Log to TensorBoard
        self.writer.add_scalar("val/loss", results["val_loss"], epoch)
        self.writer.add_scalar("val/miou", results["miou"], epoch)
        self.writer.add_scalar("val/pixel_accuracy", results["pixel_accuracy"], epoch)

        for name in CLASS_NAMES:
            self.writer.add_scalar(f"val_iou/{name}", results.get(f"iou_{name}", 0), epoch)

        return results

    def save_checkpoint(self, epoch: int, metric: float):
        """Save model checkpoint, keeping only top-k."""
        path = self.checkpoint_dir / f"epoch_{epoch:03d}_miou_{metric:.4f}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric": metric,
            "config": self.config,
        }, path)

        self.best_checkpoints.append((metric, path))
        self.best_checkpoints.sort(key=lambda x: x[0])

        # Remove old checkpoints
        while len(self.best_checkpoints) > self.save_top_k:
            _, old_path = self.best_checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()

    def fit(self, num_epochs: int) -> dict:
        """Full training loop.
        
        Returns:
            Best validation metrics dict
        """
        print(f"\n{'='*50}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")
        print(f"{'='*50}\n")

        best_results = {}

        for epoch in range(1, num_epochs + 1):
            start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_results = self.validate(epoch)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start
            miou = val_results["miou"]

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_results['val_loss']:.4f} | "
                f"mIoU: {miou:.4f} | "
                f"Pixel Acc: {val_results['pixel_accuracy']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpointing
            if miou > self.best_metric:
                self.best_metric = miou
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, miou)
                best_results = val_results
                print(f"  ★ New best mIoU: {miou:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={self.patience})")
                break

        self.writer.close()
        print(f"\nTraining complete! Best mIoU: {self.best_metric:.4f}")
        return best_results
