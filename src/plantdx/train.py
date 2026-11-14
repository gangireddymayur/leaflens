"""Train a plant-disease classifier.

Two modes, so the gain from fine-tuning is measured and not assumed:
  --mode linear    frozen pretrained backbone, train only the new head (baseline)
  --mode finetune  train the whole network
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import data as D
from . import metrics as M
from .model import build_model, eval_transform, train_transform


class ImageListDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), label


def pick_device():
    """CUDA on Kaggle, the Apple GPU (MPS) on a Mac, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model, loader, device, num_classes, criterion):
    model.eval()
    ys, ps, total_loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        ys += y.cpu().tolist()
        ps += logits.argmax(1).cpu().tolist()
    cm = M.confusion_matrix(ys, ps, num_classes)
    return {
        "loss": total_loss / max(1, len(ys)),
        "accuracy": M.accuracy(ys, ps),
        "macro_f1": M.macro_f1(cm),
    }, cm


def save_confusion_png(cm, classes, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(norm, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Normalised confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="PlantVillage folder (or its color/ subfolder)")
    ap.add_argument("--out-dir", default="runs/finetune")
    ap.add_argument("--mode", choices=["linear", "finetune"], default="finetune")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-3 linear, 3e-4 finetune")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-class", type=int, default=0, help="debug: cap images per class")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "a", buffering=1)

    def log(msg):
        """Print to the console and also save to runs/<name>/train.log."""
        print(msg, flush=True)
        log_file.write(msg + "\n")

    classes, items = D.scan_folder(D.find_data_root(args.data_dir))
    if args.limit_per_class:
        seen, kept = {}, []
        for p, y in items:
            if seen.get(y, 0) < args.limit_per_class:
                kept.append((p, y))
                seen[y] = seen.get(y, 0) + 1
        items = kept
    labels = [y for _, y in items]
    tr_idx, va_idx, te_idx = D.stratified_split(labels, seed=args.seed)
    (out / "split.json").write_text(json.dumps({
        "seed": args.seed,
        "train": [items[i][0] for i in tr_idx],
        "val": [items[i][0] for i in va_idx],
        "test": [items[i][0] for i in te_idx],
    }))
    (out / "labels.json").write_text(json.dumps(classes, indent=2))
    log(f"{len(classes)} classes | {len(tr_idx)} train / {len(va_idx)} val / {len(te_idx)} test images | device={device} | mode={args.mode} | epochs={args.epochs}")

    def loader(idx, tf, shuffle):
        ds = ImageListDataset([items[i] for i in idx], tf)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=device.type == "cuda")

    train_loader = loader(tr_idx, train_transform(), True)
    val_loader = loader(va_idx, eval_transform(), False)
    test_loader = loader(te_idx, eval_transform(), False)

    model = build_model(len(classes), pretrained=True).to(device)
    if args.mode == "linear":
        for p in model.features.parameters():
            p.requires_grad = False
    lr = args.lr or (1e-3 if args.mode == "linear" else 3e-4)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_f1, started = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.mode == "linear":
            model.features.eval()  # keep frozen batch-norm statistics fixed
        running, seen = 0.0, 0
        epoch_start = time.time()
        n_batches = len(train_loader)
        for b, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(y)
            seen += len(y)
            if b % 50 == 0 or b == n_batches:
                elapsed = time.time() - epoch_start
                eta = elapsed / b * (n_batches - b)
                log(f"  epoch {epoch}/{args.epochs}  batch {b}/{n_batches}  loss {running / seen:.4f}  "
                    f"elapsed {elapsed / 60:.1f} min  epoch ETA {eta / 60:.1f} min")
        scheduler.step()
        val, _ = run_eval(model, val_loader, device, len(classes), criterion)
        row = {"epoch": epoch, "train_loss": running / max(1, seen), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        log(f"EPOCH {epoch}/{args.epochs} done in {(time.time() - epoch_start) / 60:.1f} min | "
            f"train_loss {row['train_loss']:.4f} | val_accuracy {val['accuracy']:.4f} | val_macro_f1 {val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "classes": classes, "mode": args.mode}, out / "best.pt")

    # Final numbers come from the best-validation checkpoint, on the untouched test split.
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test, cm = run_eval(model, test_loader, device, len(classes), criterion)
    f1s = M.per_class_f1(cm)
    metrics = {
        "args": vars(args),
        "device": str(device),
        "train_seconds": round(time.time() - started, 1),
        "best_val_macro_f1": best_f1,
        "test": test,
        "per_class_f1": {c: float(f) for c, f in zip(classes, f1s)},
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(out / "confusion_matrix.npy", cm)
    save_confusion_png(cm, classes, out / "confusion_matrix.png")
    log("TEST: " + json.dumps(test))
    log(f"Saved to {out}/  (best.pt, metrics.json, confusion_matrix.png, train.log)")
    log_file.close()


if __name__ == "__main__":
    main()
