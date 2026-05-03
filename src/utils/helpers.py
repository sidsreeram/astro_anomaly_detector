"""
Utility: YAML config loader + common helpers
"""

import os
import yaml
import torch
import numpy as np
import random


def load_config(config_path: str) -> dict:
    """Load and return a YAML config file as a nested dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42):
    """Reproducibility: fix all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    if device.type == "cuda":
        print(f"         GPU  : {torch.cuda.get_device_name(0)}")
        print(f"         VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def save_checkpoint(model, optimizer, epoch, val_loss, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "val_loss":         val_loss,
    }, path)


def load_checkpoint(model, path: str, device: torch.device, optimizer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim_state_dict"])
    epoch    = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    print(f"[Checkpoint] Loaded from epoch {epoch}  |  val_loss = {val_loss:.6f}")
    return epoch, val_loss


def resolve_path(base_file: str, relative_path: str) -> str:
    """Resolve a path relative to a given base file's directory."""
    base_dir = os.path.dirname(os.path.abspath(base_file))
    return os.path.normpath(os.path.join(base_dir, relative_path))


def get_data_dirs(cfg: dict) -> tuple:
    """
    Resolve train and val directory paths from config.

    Reads cfg['data']['data_root'] and joins it with the
    train_subdir / val_subdir sub-paths.

    Raises a clear, actionable error if data_root is not configured
    or the folder does not exist.
    """
    data_root = cfg["data"].get("data_root", "UNCONFIGURED").strip()

    if data_root == "UNCONFIGURED" or data_root == "":
        print()
        print("=" * 60)
        print("  [SETUP ERROR] data_root is not configured!")
        print("=" * 60)
        print()
        print("  You need to tell the project where your dataset lives.")
        print("  Two ways to fix this:")
        print()
        print("  Option 1 - Run the setup script (easiest):")
        print("    > setup.bat")
        print()
        print("  Option 2 - Edit configs/config.yaml manually:")
        print("    Set  data_root  to the full path of your")
        print("    Hybrid-Astro-Deblender folder. Example:")
        print()
        print('    data_root: "E:/COLLAGE/ml/research/Hybrid-Astro-Deblender"')
        print()
        print("=" * 60)
        raise SystemExit(1)

    # Normalise slashes for cross-platform compatibility
    data_root = os.path.normpath(data_root)

    train_dir = os.path.join(data_root, cfg["data"]["train_subdir"])
    val_dir   = os.path.join(data_root, cfg["data"]["val_subdir"])

    # Validate directories exist
    for label, path in [("train_dir", train_dir), ("val_dir", val_dir)]:
        if not os.path.exists(path):
            print()
            print("=" * 60)
            print(f"  [PATH ERROR] {label} not found:")
            print(f"    {path}")
            print("=" * 60)
            print()
            print("  Check that data_root in configs/config.yaml points to")
            print("  the correct Hybrid-Astro-Deblender folder on your machine.")
            print()
            print(f"  Current data_root : {data_root}")
            print()
            raise SystemExit(1)

    return train_dir, val_dir

