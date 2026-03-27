"""
Data augmentation pipelines using Albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    img_size: tuple = (512, 512),
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Training augmentations with geometric and color transforms."""
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3, border_mode=0),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(
    img_size: tuple = (512, 512),
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Validation/test transforms (resize + normalize only)."""
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
