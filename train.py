"""
Forest Inspection - Semantic Segmentation Training CLI

Usage:
    python train.py --config configs/train_unet.yaml
    python train.py --config configs/train_unet.yaml --epochs 10 --batch-size 4
    python train.py --config configs/train_unet.yaml --data-root /kaggle/input/forest-sunny
"""

import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.dataset import ForestDataset, NUM_CLASSES
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.splits import get_split, print_split_info
from src.models import build_model
from src.training.losses import CEDiceLoss
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Forest Segmentation Model")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root (single path)")
    parser.add_argument("--data-roots", type=str, nargs="+", default=None, help="Override dataset roots (multiple paths for Kaggle)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    cfg = load_config("configs/dataset.yaml", args.config)

    # CLI overrides
    if args.data_roots:
        data_roots = args.data_roots
    elif args.data_root:
        data_roots = [args.data_root]
    else:
        data_roots = list(cfg.dataset.get("data_roots", [cfg.dataset.root]))
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.lr:
        cfg.training.optimizer.lr = args.lr

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Logger
    logger = setup_logger()

    # Data split
    split_cfg = cfg.dataset.split
    split = get_split(
        strategy=split_cfg.strategy,
        train_sequences=list(split_cfg.get("train_sequences", [])),
        val_sequences=list(split_cfg.get("val_sequences", [])),
        test_sequences=list(split_cfg.get("test_sequences", [])),
    )
    print_split_info(split)

    # Transforms
    img_size = tuple(cfg.augmentation.train.resize)
    train_transforms = get_train_transforms(img_size=img_size)
    val_transforms = get_val_transforms(img_size=img_size)

    # Datasets
    train_dataset = ForestDataset(
        root=data_roots,
        sequences=split["train"],
        transform=train_transforms,
    )
    val_dataset = ForestDataset(
        root=data_roots,
        sequences=split["val"],
        transform=val_transforms,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    print(f"\nTrain: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # Model
    model = build_model(
        name=cfg.model.name,
        encoder=cfg.model.encoder,
        encoder_weights=cfg.model.encoder_weights,
        num_classes=cfg.model.num_classes,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {cfg.model.name} ({cfg.model.encoder})")
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss
    criterion = CEDiceLoss(
        num_classes=NUM_CLASSES,
        ce_weight=cfg.training.loss.ce_weight,
        dice_weight=cfg.training.loss.dice_weight,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.scheduler.T_max,
        eta_min=cfg.training.scheduler.eta_min,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # Train
    trainer_config = {
        "amp": cfg.training.amp,
        "accumulation_steps": cfg.training.accumulation_steps,
        "log_dir": cfg.output.log_dir,
        "checkpoint_dir": cfg.output.checkpoint_dir,
        "save_top_k": cfg.output.save_top_k,
        "early_stopping": dict(cfg.training.early_stopping),
        "num_classes": NUM_CLASSES,
        "log_every_n_steps": cfg.logging.log_every_n_steps,
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_config,
        device=device,
    )

    best_results = trainer.fit(num_epochs=cfg.training.epochs)

    # Print final results
    print(f"\n{'='*50}")
    print("Final Results:")
    print(f"  mIoU: {best_results.get('miou', 0):.4f}")
    print(f"  Pixel Accuracy: {best_results.get('pixel_accuracy', 0):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
