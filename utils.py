"""
Shared utility helpers used across the project.
"""

import json
from pathlib import Path
from typing import List

from . import config


def list_class_names(source_dir: Path = None) -> List[str]:
    """
    Return a sorted list of class folder names from the given directory.
    Defaults to TRAIN_DIR; falls back to RAW_DATA_DIR if train doesn't exist yet.
    """
    if source_dir is None:
        source_dir = config.TRAIN_DIR if config.TRAIN_DIR.exists() else config.RAW_DATA_DIR
    if not source_dir.exists():
        raise FileNotFoundError(f"Class source directory not found: {source_dir}")
    return sorted(p.name for p in source_dir.iterdir() if p.is_dir())


def save_class_names(class_names: List[str], path: Path = None) -> Path:
    """Persist class names to a JSON file so all scripts share one source of truth."""
    if path is None:
        path = config.MODELS_DIR / "class_names.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    return path


def load_class_names(path: Path = None) -> List[str]:
    """Load class names previously saved by save_class_names()."""
    if path is None:
        path = config.MODELS_DIR / "class_names.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_images(folder: Path) -> int:
    """Recursively count image files in a folder."""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sum(1 for p in Path(folder).rglob("*") if p.suffix in exts)
