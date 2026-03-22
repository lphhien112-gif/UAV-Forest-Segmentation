"""
Deforestation Analysis Module.

Estimates deforestation degree based on semantic segmentation predictions.
Uses the ratio of tree-related classes vs. non-tree classes to assess forest health.

Reference:
    The Forest Inspection paper (arXiv:2403.06621) proposes a framework 
    to assess the deforestation degree of an area using segmentation outputs.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..data.dataset import CLASS_NAMES, NUM_CLASSES


# Define class groups for deforestation analysis
TREE_CLASSES = {
    1: "Deciduous_trees",
    2: "Coniferous_trees",
}

FALLEN_TREE_CLASSES = {
    3: "Fallen_trees",
}

VEGETATION_CLASSES = {
    5: "Ground_vegetation",
}

GROUND_CLASSES = {
    4: "Dirt_ground",
    6: "Rocks",
}

STRUCTURE_CLASSES = {
    7: "Building",
    8: "Fence",
    9: "Car",
}

IGNORE_CLASSES = {
    0: "Sky",
    10: "Empty",
}


def compute_class_percentages(pred_mask: np.ndarray) -> Dict[str, float]:
    """Compute percentage of each class in a prediction mask.
    
    Args:
        pred_mask: (H, W) predicted class IDs
        
    Returns:
        Dict mapping class name to percentage
    """
    total_pixels = pred_mask.size
    percentages = {}
    
    for class_id, name in enumerate(CLASS_NAMES):
        count = (pred_mask == class_id).sum()
        percentages[name] = float(count / total_pixels * 100)
    
    return percentages


def compute_deforestation_index(pred_mask: np.ndarray) -> Dict[str, float]:
    """Compute deforestation-related indices from a prediction mask.
    
    Indices:
        - canopy_cover: % of area covered by standing trees
        - fallen_tree_ratio: % of trees that are fallen
        - vegetation_index: overall greenness (trees + ground vegetation)
        - disturbance_index: inverse of canopy cover among vegetated area
        - deforestation_degree: 0 (fully forested) to 1 (fully deforested)
    
    Args:
        pred_mask: (H, W) predicted class IDs
        
    Returns:
        Dictionary of deforestation indices
    """
    total = pred_mask.size
    
    # Count pixels per group (excluding sky and empty)
    tree_pixels = sum((pred_mask == cid).sum() for cid in TREE_CLASSES)
    fallen_pixels = sum((pred_mask == cid).sum() for cid in FALLEN_TREE_CLASSES)
    veg_pixels = sum((pred_mask == cid).sum() for cid in VEGETATION_CLASSES)
    ground_pixels = sum((pred_mask == cid).sum() for cid in GROUND_CLASSES)
    structure_pixels = sum((pred_mask == cid).sum() for cid in STRUCTURE_CLASSES)
    ignore_pixels = sum((pred_mask == cid).sum() for cid in IGNORE_CLASSES)
    
    # Effective area (excluding sky and empty)
    effective_area = total - ignore_pixels
    if effective_area == 0:
        effective_area = 1  # avoid division by zero
    
    # Canopy cover: standing trees / effective area
    canopy_cover = float(tree_pixels / effective_area)
    
    # Fallen tree ratio: fallen / (standing + fallen)
    all_trees = tree_pixels + fallen_pixels
    fallen_ratio = float(fallen_pixels / all_trees) if all_trees > 0 else 0.0
    
    # Vegetation index: all green / effective area
    vegetation_index = float((tree_pixels + veg_pixels) / effective_area)
    
    # Deforestation degree: 1 - canopy_cover (simple metric)
    # 0 = fully forested, 1 = fully deforested
    deforestation_degree = 1.0 - canopy_cover
    
    # Disturbance index: considers fallen trees and bare ground
    disturbed = fallen_pixels + ground_pixels + structure_pixels
    disturbance_index = float(disturbed / effective_area)
    
    return {
        "canopy_cover": canopy_cover,
        "fallen_tree_ratio": fallen_ratio,
        "vegetation_index": vegetation_index,
        "deforestation_degree": deforestation_degree,
        "disturbance_index": disturbance_index,
        "tree_pixels": int(tree_pixels),
        "fallen_pixels": int(fallen_pixels),
        "ground_pixels": int(ground_pixels),
        "effective_area": int(effective_area),
    }


@torch.no_grad()
def analyze_sequence_deforestation(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    sequence_name: str = "",
) -> Dict[str, float]:
    """Analyze deforestation across an entire sequence.
    
    Args:
        model: Trained segmentation model
        dataloader: DataLoader for the sequence
        device: torch device
        sequence_name: Name for logging
        
    Returns:
        Aggregated deforestation indices for the sequence
    """
    model.eval()
    
    all_indices = []
    
    for images, _ in tqdm(dataloader, desc=f"Analyzing {sequence_name}"):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        for i in range(preds.shape[0]):
            indices = compute_deforestation_index(preds[i])
            all_indices.append(indices)
    
    # Aggregate: average across all frames
    aggregated = {}
    keys = all_indices[0].keys()
    for key in keys:
        values = [idx[key] for idx in all_indices]
        aggregated[f"mean_{key}"] = float(np.mean(values))
        aggregated[f"std_{key}"] = float(np.std(values))
    
    return aggregated


def plot_deforestation_comparison(
    per_sequence_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
):
    """Plot deforestation indices comparison across sequences.
    
    Args:
        per_sequence_results: Dict mapping sequence name to deforestation indices
        save_path: Optional path to save figure
    """
    seq_names = list(per_sequence_results.keys())
    metrics = ["mean_canopy_cover", "mean_deforestation_degree", "mean_disturbance_index"]
    labels = ["Canopy Cover", "Deforestation Degree", "Disturbance Index"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, metric, label in zip(axes, metrics, labels):
        values = [per_sequence_results[s].get(metric, 0) for s in seq_names]
        colors = plt.cm.RdYlGn_r(np.array(values))  # Red=bad, Green=good
        
        bars = ax.bar(seq_names, values, color=colors)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                fontsize=9,
            )
    
    plt.suptitle("Deforestation Analysis Across Sequences", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved deforestation plot to {save_path}")
    
    plt.show()
    plt.close()
