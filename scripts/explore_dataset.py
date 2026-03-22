"""
Dataset Explorer - Visualize and analyze the Forest Inspection Dataset.

Usage:
    python scripts/explore_dataset.py --data data/forest_sunny
    python scripts/explore_dataset.py --data data/forest_sunny --seq seq1 --show-samples 5
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Label color mapping (RGB)
LABEL_COLORS = {
    "Sky": (0, 255, 255),
    "Deciduous_trees": (0, 127, 0),
    "Coniferous_trees": (19, 132, 69),
    "Fallen_trees": (0, 53, 65),
    "Dirt_ground": (130, 76, 0),
    "Ground_vegetation": (152, 251, 152),
    "Rocks": (151, 126, 171),
    "Building": (250, 150, 0),
    "Fence": (115, 176, 195),
    "Car": (123, 123, 123),
    "Empty": (0, 0, 0),
}


def count_images(data_root: Path) -> dict:
    """Count images per sequence."""
    counts = {}
    for seq_dir in sorted(data_root.iterdir()):
        if seq_dir.is_dir() and seq_dir.name.startswith("seq"):
            color_dir = seq_dir / "color"
            label_dir = seq_dir / "labels"
            n_color = len(list(color_dir.glob("*.png"))) if color_dir.exists() else 0
            n_label = len(list(label_dir.glob("*.png"))) if label_dir.exists() else 0
            counts[seq_dir.name] = {"color": n_color, "labels": n_label}
    return counts


def analyze_class_distribution(data_root: Path, seq_name: str, sample_count: int = 50):
    """Analyze pixel-level class distribution from sampled label images."""
    label_dir = data_root / seq_name / "labels"
    if not label_dir.exists():
        print(f"  Label directory not found: {label_dir}")
        return {}

    label_files = sorted(label_dir.glob("*.png"))
    if not label_files:
        return {}

    # Sample evenly
    step = max(1, len(label_files) // sample_count)
    sampled = label_files[::step][:sample_count]

    # Build color→class lookup
    color_to_class = {}
    for name, rgb in LABEL_COLORS.items():
        color_to_class[rgb] = name

    pixel_counts = Counter()
    total_pixels = 0

    for lf in sampled:
        img = np.array(Image.open(lf).convert("RGB"))
        h, w, _ = img.shape
        total_pixels += h * w

        for name, rgb in LABEL_COLORS.items():
            mask = np.all(img == np.array(rgb), axis=-1)
            pixel_counts[name] += mask.sum()

    # Convert to percentages
    distribution = {
        name: (count / total_pixels * 100) for name, count in pixel_counts.items()
    }
    return distribution


def show_samples(data_root: Path, seq_name: str, n_samples: int = 3):
    """Display sample images with their labels side by side."""
    color_dir = data_root / seq_name / "color"
    label_dir = data_root / seq_name / "labels"

    color_files = sorted(color_dir.glob("*.png"))
    step = max(1, len(color_files) // n_samples)
    sampled = color_files[::step][:n_samples]

    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, cf in enumerate(sampled):
        lf = label_dir / cf.name

        img = Image.open(cf)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"RGB: {cf.name}", fontsize=10)
        axes[i, 0].axis("off")

        if lf.exists():
            label = Image.open(lf)
            axes[i, 1].imshow(label)
            axes[i, 1].set_title(f"Label: {lf.name}", fontsize=10)
            axes[i, 1].axis("off")

    # Add legend
    patches = [
        mpatches.Patch(color=np.array(rgb) / 255, label=name)
        for name, rgb in LABEL_COLORS.items()
    ]
    fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"outputs/samples_{seq_name}.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: outputs/samples_{seq_name}.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Explore Forest Inspection Dataset")
    parser.add_argument(
        "--data", type=str, default="data/forest_sunny", help="Dataset root"
    )
    parser.add_argument(
        "--seq", type=str, default=None, help="Specific sequence to analyze"
    )
    parser.add_argument(
        "--show-samples", type=int, default=0, help="Show N sample image pairs"
    )
    args = parser.parse_args()

    data_root = Path(args.data)

    print("╔══════════════════════════════════════════╗")
    print("║  Forest Inspection Dataset Explorer      ║")
    print("╚══════════════════════════════════════════╝\n")

    # Count images
    print("📊 Image counts per sequence:")
    counts = count_images(data_root)
    if not counts:
        print(f"  No sequences found in {data_root.resolve()}")
        print("  Run `python scripts/download_zenodo.py` first.")
        return

    total = 0
    for seq, c in counts.items():
        print(f"  {seq}: {c['color']} RGB, {c['labels']} labels")
        total += c["color"]
    print(f"  Total: {total} image pairs\n")

    # Class distribution
    target_seqs = [args.seq] if args.seq else list(counts.keys())[:3]
    for seq in target_seqs:
        print(f"🏷️  Class distribution ({seq}, sampled):")
        dist = analyze_class_distribution(data_root, seq)
        for cls, pct in sorted(dist.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct / 2)
            print(f"    {cls:20s} {pct:6.2f}% {bar}")
        print()

    # Show samples
    if args.show_samples > 0:
        seq = args.seq or list(counts.keys())[0]
        Path("outputs").mkdir(exist_ok=True)
        print(f"🖼️  Showing {args.show_samples} samples from {seq}:")
        show_samples(data_root, seq, args.show_samples)


if __name__ == "__main__":
    main()
