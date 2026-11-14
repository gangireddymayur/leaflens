"""Dataset scanning and splitting. Pure Python and NumPy, so it is easy to unit-test."""
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_data_root(path):
    """PlantVillage ships color/, grayscale/ and segmented/. Use color/ when present."""
    path = Path(path)
    return path / "color" if (path / "color").is_dir() else path


def scan_folder(root):
    """Return (classes, items) where items is a list of (image_path, class_index)."""
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not classes:
        raise ValueError(f"No class folders found in {root}")
    items = []
    for idx, name in enumerate(classes):
        for p in sorted((root / name).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                items.append((str(p), idx))
    if not items:
        raise ValueError(f"No images found in {root}")
    return classes, items


def stratified_split(labels, fractions=(0.7, 0.15, 0.15), seed=0):
    """Split indices into train/val/test, keeping class proportions.

    Every class with at least 3 samples appears in all three splits.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, round(n * fractions[1])) if n >= 3 else 0
        n_test = max(1, round(n * fractions[2])) if n >= 3 else 0
        n_train = n - n_val - n_test
        train += idx[:n_train].tolist()
        val += idx[n_train:n_train + n_val].tolist()
        test += idx[n_train + n_val:].tolist()
    return sorted(train), sorted(val), sorted(test)
