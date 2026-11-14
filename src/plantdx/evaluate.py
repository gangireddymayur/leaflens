"""Evaluate a checkpoint on any folder of class-named subfolders.

Use this on real phone photos or another dataset (for example PlantDoc) to see how far
the lab-photo accuracy carries over. Classes the model does not know are reported, not hidden.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as D
from . import metrics as M
from .model import build_model, eval_transform
from .train import ImageListDataset, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None, help="optional path to write a JSON report")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    folder_classes, items = D.scan_folder(D.find_data_root(args.data_dir))
    known = {name: i for i, name in enumerate(classes)}
    unknown = sorted(c for c in folder_classes if c not in known)
    kept = [(p, known[folder_classes[y]]) for p, y in items if folder_classes[y] in known]
    if not kept:
        raise SystemExit("No folder names match the model's classes. Rename folders or map them first.")

    loader = DataLoader(ImageListDataset(kept, eval_transform()), batch_size=args.batch_size)
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += model(x.to(device)).argmax(1).cpu().tolist()
            ys += y.tolist()
    cm = M.confusion_matrix(ys, ps, len(classes))
    report = {
        "images_evaluated": len(kept),
        "classes_not_in_model": unknown,
        "accuracy": M.accuracy(ys, ps),
        "macro_f1": M.macro_f1(cm),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
