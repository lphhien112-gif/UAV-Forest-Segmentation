"""
Configuration utility - loads and merges YAML configs.
"""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(*config_paths: str) -> DictConfig:
    """Load and merge multiple YAML config files.
    
    Later configs override earlier ones.
    
    Usage:
        cfg = load_config("configs/dataset.yaml", "configs/train_unet.yaml")
    """
    configs = []
    for path in config_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        configs.append(OmegaConf.load(p))
    
    merged = OmegaConf.merge(*configs)
    return merged


def save_config(cfg: DictConfig, path: str) -> None:
    """Save config to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def print_config(cfg: DictConfig) -> None:
    """Pretty print configuration."""
    print(OmegaConf.to_yaml(cfg))
