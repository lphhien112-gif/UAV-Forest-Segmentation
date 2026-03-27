"""
Forest Inspection PyTorch Dataset.

Loads RGB images and semantic label maps from the forest inspection dataset.
Handles color-to-class-id conversion for labels.
Supports multi-root data loading (e.g., multiple Kaggle dataset mounts).
"""

import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

# RGB -> class_id mapping
LABEL_COLORS = [
    (0, 255, 255),    # 0: Sky
    (0, 127, 0),      # 1: Deciduous_trees
    (19, 132, 69),    # 2: Coniferous_trees
    (0, 53, 65),      # 3: Fallen_trees
    (130, 76, 0),     # 4: Dirt_ground
    (152, 251, 152),  # 5: Ground_vegetation
    (151, 126, 171),  # 6: Rocks
    (250, 150, 0),    # 7: Building
    (115, 176, 195),  # 8: Fence
    (123, 123, 123),  # 9: Car
    (0, 0, 0),        # 10: Empty
]

CLASS_NAMES = [
    "Sky", "Deciduous_trees", "Coniferous_trees", "Fallen_trees",
    "Dirt_ground", "Ground_vegetation", "Rocks", "Building",
    "Fence", "Car", "Empty",
]

NUM_CLASSES = len(CLASS_NAMES)


def rgb_to_class_id(label_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB label image to class ID map.

    Args:
        label_rgb: (H, W, 3) uint8 array

    Returns:
        class_map: (H, W) int64 array with class IDs 0-10
    """
    h, w, _ = label_rgb.shape
    class_map = np.full((h, w), NUM_CLASSES - 1, dtype=np.int64)  # default: Empty

    for class_id, color in enumerate(LABEL_COLORS):
        mask = np.all(label_rgb == np.array(color, dtype=np.uint8), axis=-1)
        class_map[mask] = class_id

    return class_map


def class_id_to_rgb(class_map: np.ndarray) -> np.ndarray:
    """Convert class ID map back to RGB for visualization.

    Args:
        class_map: (H, W) array with class IDs

    Returns:
        rgb: (H, W, 3) uint8 array
    """
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(LABEL_COLORS):
        mask = class_map == class_id
        rgb[mask] = color

    return rgb


def _find_seq_dir(roots: List[Path], seq: str) -> Optional[Path]:
    """Find a sequence's color/labels directory across multiple roots.

    Handles both flat layout (root/seq/color/) and nested layout
    (root/seq/seq/color/) as seen in Kaggle dataset uploads.

    Returns:
        Path to the directory containing color/ and labels/, or None.
    """
    for root in roots:
        # Flat layout: root/seq1/color/
        flat = root / seq
        if (flat / "color").exists():
            return flat

        # Nested layout: root/seq1/seq1/color/ (Kaggle double-nesting)
        nested = root / seq / seq
        if (nested / "color").exists():
            return nested

    return None


class ForestDataset(Dataset):
    """PyTorch Dataset for Forest Inspection segmentation.

    Supports single or multiple data roots for loading sequences from
    different locations (e.g., multiple Kaggle dataset mounts).

    Args:
        root: Path(s) to dataset root(s). Can be:
            - str: single path (e.g., "data/forest_sunny")
            - List[str]: multiple paths (e.g., ["/kaggle/input/part1", "/kaggle/input/part2"])
        sequences: List of sequence names (e.g., ["seq1", "seq2"])
        transform: Optional albumentations transform pipeline
    """

    def __init__(
        self,
        root: Union[str, List[str]],
        sequences: List[str],
        transform: Optional[Callable] = None,
    ):
        if isinstance(root, (str, Path)):
            self.roots = [Path(root)]
        else:
            self.roots = [Path(r) for r in root]

        self.transform = transform
        self.samples: List[Tuple[Path, Path]] = []

        for seq in sequences:
            seq_dir = _find_seq_dir(self.roots, seq)

            if seq_dir is None:
                print(f"Warning: {seq} not found in any root, skipping.")
                continue

            color_dir = seq_dir / "color"
            label_dir = seq_dir / "labels"

            color_files = sorted(color_dir.glob("*.png"))
            for cf in color_files:
                lf = label_dir / cf.name
                if lf.exists():
                    self.samples.append((cf, lf))

        print(f"ForestDataset: loaded {len(self.samples)} samples from {sequences}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        color_path, label_path = self.samples[idx]

        # Load images
        image = np.array(Image.open(color_path).convert("RGB"))
        label_rgb = np.array(Image.open(label_path).convert("RGB"))

        # Convert label to class IDs
        mask = rgb_to_class_id(label_rgb)

        # Apply augmentations (albumentations)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensors
        if isinstance(image, np.ndarray):
            # HWC -> CHW, normalize to [0, 1]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        return image, mask.long()

    def get_sample_path(self, idx: int) -> Tuple[str, str]:
        """Get file paths for a sample (useful for debugging)."""
        color_path, label_path = self.samples[idx]
        return str(color_path), str(label_path)
