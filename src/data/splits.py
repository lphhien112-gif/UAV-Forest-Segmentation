"""
Dataset split strategies for Forest Inspection Dataset.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union


# Sequence metadata
SEQUENCE_INFO = {
    "seq1": {"pitch": 0, "altitude": 30},
    "seq2": {"pitch": -60, "altitude": 30},
    "seq3": {"pitch": -90, "altitude": 30},
    "seq4": {"pitch": 0, "altitude": 50},
    "seq5": {"pitch": -60, "altitude": 50},
    "seq6": {"pitch": -90, "altitude": 50},
    "seq7": {"pitch": 0, "altitude": 80},
    "seq8": {"pitch": -60, "altitude": 80},
    "seq9": {"pitch": -90, "altitude": 80},
}


def get_available_sequences(data_roots: Union[str, List[str]]) -> List[str]:
    """Scan one or more root directories for available sequences.

    Handles both flat (root/seq1/color/) and nested (root/seq1/seq1/color/)
    directory layouts.

    Args:
        data_roots: Single path or list of paths to scan.

    Returns:
        Sorted list of available sequence names.
    """
    if isinstance(data_roots, (str, Path)):
        data_roots = [data_roots]

    found = set()
    for root in data_roots:
        root = Path(root)
        if not root.exists():
            continue
        for d in root.iterdir():
            if not d.is_dir() or not d.name.startswith("seq"):
                continue
            # Flat layout: root/seq1/color/
            if (d / "color").exists():
                found.add(d.name)
            # Nested layout: root/seq1/seq1/color/
            elif (d / d.name / "color").exists():
                found.add(d.name)

    return sorted(found)


def get_split(
    strategy: str = "cross_sequence",
    train_sequences: List[str] = None,
    val_sequences: List[str] = None,
    test_sequences: List[str] = None,
) -> Dict[str, List[str]]:
    """Get train/val/test split based on strategy.

    Strategies:
        - cross_sequence: Use specific sequences for each split
        - cross_altitude: Train low+mid, test high
        - cross_pitch: Train by pitch angle
        - all_train: All sequences for training (cross-val separately)

    Returns:
        dict with 'train', 'val', 'test' keys, each containing list of seq names
    """
    all_seqs = list(SEQUENCE_INFO.keys())

    if strategy == "cross_sequence":
        return {
            "train": train_sequences or ["seq1", "seq2", "seq3", "seq4", "seq5", "seq7", "seq8"],
            "val": val_sequences or ["seq6"],
            "test": test_sequences or ["seq9"],
        }

    elif strategy == "cross_altitude":
        # Train: 30m + 50m, Test: 80m
        return {
            "train": ["seq1", "seq2", "seq3", "seq4", "seq5"],
            "val": ["seq6"],
            "test": ["seq7", "seq8", "seq9"],
        }

    elif strategy == "cross_pitch":
        # Train: 0 deg + -60 deg, Test: -90 deg (top-down)
        return {
            "train": ["seq1", "seq2", "seq4", "seq5", "seq7", "seq8"],
            "val": ["seq3"],
            "test": ["seq6", "seq9"],
        }

    elif strategy == "all_train":
        return {
            "train": all_seqs,
            "val": [],
            "test": [],
        }

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")


def print_split_info(split: Dict[str, List[str]]) -> None:
    """Print detailed info about a data split."""
    for subset, seqs in split.items():
        if not seqs:
            continue
        print(f"\n{subset.upper()}:")
        for seq in seqs:
            info = SEQUENCE_INFO.get(seq, {})
            print(f"  {seq}: pitch={info.get('pitch', '?')} deg, alt={info.get('altitude', '?')}m")
