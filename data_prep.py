"""
Phase 1 - Image data preparation.

Splits the PlantVillage dataset in data/plantvillage_raw/ into
train/val/test folders using a 70/15/15 ratio (configurable in src/config.py).

Usage:
    venv/Scripts/python.exe -m src.data_prep
"""

import random
import shutil
from pathlib import Path

from tqdm import tqdm

from . import config
from .utils import ensure_dir, list_class_names, save_class_names

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def gather_images(class_dir: Path):
    return [p for p in class_dir.iterdir() if p.suffix in IMAGE_EXTS]


def split_indices(n: int, train_ratio: float, val_ratio: float):
    """Return three lists of indices: train, val, test (sum = n)."""
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def copy_subset(files, dest_class_dir: Path):
    ensure_dir(dest_class_dir)
    for src in files:
        shutil.copy2(src, dest_class_dir / src.name)


def main():
    random.seed(config.RANDOM_SEED)

    if not config.RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {config.RAW_DATA_DIR}. "
            f"Please copy the 38 PlantVillage class folders into it first."
        )

    class_names = list_class_names(config.RAW_DATA_DIR)
    if len(class_names) != config.NUM_CLASSES:
        print(
            f"WARNING: expected {config.NUM_CLASSES} classes, found {len(class_names)}"
        )

    print(f"Found {len(class_names)} classes in {config.RAW_DATA_DIR}")
    print(
        f"Split ratios -> train {config.TRAIN_RATIO}, "
        f"val {config.VAL_RATIO}, test {config.TEST_RATIO}"
    )

    # Wipe any previous split so reruns are reproducible
    for d in (config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR):
        if d.exists():
            shutil.rmtree(d)
        ensure_dir(d)

    totals = {"train": 0, "val": 0, "test": 0}

    for class_name in tqdm(class_names, desc="Classes"):
        src_dir = config.RAW_DATA_DIR / class_name
        files = gather_images(src_dir)
        if not files:
            print(f"  Skipping empty class: {class_name}")
            continue

        train_idx, val_idx, test_idx = split_indices(
            len(files), config.TRAIN_RATIO, config.VAL_RATIO
        )

        copy_subset([files[i] for i in train_idx], config.TRAIN_DIR / class_name)
        copy_subset([files[i] for i in val_idx], config.VAL_DIR / class_name)
        copy_subset([files[i] for i in test_idx], config.TEST_DIR / class_name)

        totals["train"] += len(train_idx)
        totals["val"] += len(val_idx)
        totals["test"] += len(test_idx)

    save_class_names(class_names)
    print("\nSplit complete.")
    print(f"  train: {totals['train']} images -> {config.TRAIN_DIR}")
    print(f"  val:   {totals['val']} images -> {config.VAL_DIR}")
    print(f"  test:  {totals['test']} images -> {config.TEST_DIR}")
    print(f"  class_names.json saved to {config.MODELS_DIR / 'class_names.json'}")


if __name__ == "__main__":
    main()
