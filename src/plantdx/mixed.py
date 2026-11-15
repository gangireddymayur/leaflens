"""Training on a mix of lab photos (PlantVillage) and real-world photos (PlantDoc).

A model that only saw lab photos falls apart on real ones. Here the model sees both, with the
real-world images repeated and augmented hard. PlantDoc is split once into train / val / test.
Validation picks the best epoch; the test split is only used for the final report.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T

from . import data as D
from . import metrics as M
from .model import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, eval_transform
from .real_world import PLANTDOC_TO_OURS, collect
from .train import pick_device, set_seed


def strong_transform():
    return T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.25, 1.0), ratio=(0.7, 1.4)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(30)], p=0.5),
        T.ColorJitter(0.4, 0.4, 0.4, 0.05),
        T.RandomGrayscale(0.05),
        T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.3),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def squash_transform():
    """Whole image squeezed to 224x224: keeps the leaf in frame when the photo is not a tight crop."""
    return T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ListDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), y


class MixSampler(Sampler):
    """Each epoch: every real-world training photo `repeat` times, plus a fresh random lab subset."""

    def __init__(self, n_pd, n_pv, repeat, pv_per_epoch, seed):
        self.n_pd, self.n_pv, self.repeat, self.pv_per_epoch = n_pd, n_pv, repeat, pv_per_epoch
        self.rng = random.Random(seed)

    def __iter__(self):
        idx = list(range(self.n_pd)) * self.repeat
        idx += [self.n_pd + i for i in self.rng.sample(range(self.n_pv), min(self.pv_per_epoch, self.n_pv))]
        self.rng.shuffle(idx)
        return iter(idx)

    def __len__(self):
        return self.n_pd * self.repeat + min(self.pv_per_epoch, self.n_pv)


def build_model(name, num_classes, pretrained=True):
    if name == "mobilenet_v3_large":
        w = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.mobilenet_v3_large(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "efficientnet_b0":
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b2":
        w = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.efficientnet_b2(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise SystemExit(f"unknown backbone {name}")
    return m


@torch.no_grad()
def evaluate(model, loader, device, n_cls, topk=3):
    model.eval()
    ys, ps, top = [], [], []
    for x, y in loader:
        lg = model(x.to(device)).cpu()
        ys += y.tolist(); ps += lg.argmax(1).tolist()
        top += (lg.topk(topk, 1).indices == y[:, None]).any(1).tolist()
    cm = M.confusion_matrix(ys, ps, n_cls)
    present = cm.sum(1) > 0
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(n_cls), where=present)
    return {"top1": M.accuracy(ys, ps), "top3": float(np.mean(top)), "balanced": float(recall[present].mean()),
            "n": len(ys)}, cm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plantvillage", default="data/plantvillage")
    ap.add_argument("--plantdoc", default="data/raw/plantdoc")
    ap.add_argument("--labels", default="app/public/model/labels.json")
    ap.add_argument("--backbone", default="mobilenet_v3_large")
    ap.add_argument("--init", default="", help="optional checkpoint with a matching architecture")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--repeat", type=int, default=3, help="times each real-world training photo is shown per epoch")
    ap.add_argument("--pv-per-epoch", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--squash", action="store_true", help="evaluate with whole-image resize instead of centre crop")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    set_seed(a.seed)
    device = pick_device()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(a.labels)); idx = {n: i for i, n in enumerate(labels)}; n_cls = len(labels)

    # PlantDoc: one fixed stratified split (the same for every experiment)
    pd_items = [(p, idx[n]) for p, n in collect(a.plantdoc)]
    tr_i, va_i, te_i = D.stratified_split([y for _, y in pd_items], fractions=(0.6, 0.2, 0.2), seed=1234)
    pd_train, pd_val, pd_test = ([pd_items[i] for i in s] for s in (tr_i, va_i, te_i))
    # PlantVillage: all photos, used as extra training data and to watch for forgetting
    _, pv_all = D.scan_folder(D.find_data_root(a.plantvillage))
    pv_names = sorted(d.name for d in D.find_data_root(a.plantvillage).iterdir() if d.is_dir())
    pv_items = [(p, idx[pv_names[y]]) for p, y in pv_all]
    pv_tr, pv_va, pv_te = D.stratified_split([y for _, y in pv_items], seed=42)
    pv_train, pv_test = [pv_items[i] for i in pv_tr], [pv_items[i] for i in pv_te]
    print(f"PlantDoc train/val/test: {len(pd_train)}/{len(pd_val)}/{len(pd_test)} | PlantVillage train/test: {len(pv_train)}/{len(pv_test)} | device={device}")

    ev_tf = squash_transform() if a.squash else eval_transform()
    train_ds = ListDataset(pd_train + pv_train, strong_transform())
    sampler = MixSampler(len(pd_train), len(pv_train), a.repeat, a.pv_per_epoch, a.seed)
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, sampler=sampler, num_workers=a.workers, persistent_workers=a.workers > 0)
    mk = lambda items: DataLoader(ListDataset(items, ev_tf), batch_size=a.batch_size, num_workers=a.workers)
    pd_val_l, pd_test_l, pv_test_l = mk(pd_val), mk(pd_test), mk(pv_test[:2000])

    model = build_model(a.backbone, n_cls, pretrained=not a.init).to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device)["state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train_loader), pct_start=0.15)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    log = open(out / "train.log", "a", buffering=1)
    def say(m): print(m, flush=True); log.write(m + "\n")
    best, hist, t0 = -1.0, [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); run = seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y); loss.backward(); opt.step(); sched.step()
            run += loss.item() * len(y); seen += len(y)
        v, _ = evaluate(model, pd_val_l, device, n_cls)
        row = {"epoch": ep, "train_loss": run / seen, "plantdoc_val_top1": v["top1"], "plantdoc_val_top3": v["top3"], "plantdoc_val_balanced": v["balanced"]}
        hist.append(row)
        say(f"EPOCH {ep}/{a.epochs} | loss {row['train_loss']:.3f} | PlantDoc val top-1 {v['top1']*100:.1f}% top-3 {v['top3']*100:.1f}% balanced {v['balanced']*100:.1f}% | {(time.time()-t0)/60:.1f} min")
        if v["top1"] > best:
            best = v["top1"]; torch.save({"state_dict": model.state_dict(), "classes": labels, "backbone": a.backbone}, out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["state_dict"])
    te, cm = evaluate(model, pd_test_l, device, n_cls)
    pv, _ = evaluate(model, pv_test_l, device, n_cls)
    res = {"args": vars(a), "best_val_top1": best, "plantdoc_test": te, "plantvillage_test_2000": pv, "history": hist, "minutes": (time.time() - t0) / 60}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    json.dump({"train": [p for p, _ in pd_train], "val": [p for p, _ in pd_val], "test": [p for p, _ in pd_test]}, open(out / "plantdoc_split.json", "w"))
    say(f"FINAL held-out PlantDoc test: top-1 {te['top1']*100:.1f}% top-3 {te['top3']*100:.1f}% balanced {te['balanced']*100:.1f}% (n={te['n']}) | PlantVillage test top-1 {pv['top1']*100:.1f}%")


if __name__ == "__main__":
    main()
