"""Evaluate the shipped ONNX model on PlantDoc, a set of real-world web photos.

PlantVillage photos are taken on plain backgrounds, so scores on its test split are optimistic.
PlantDoc (Singh et al., 2020, CC BY 4.0) holds cropped photos scraped from the web, which look
much more like what a phone camera produces. The model is only tested on it, never trained on it.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .data import IMG_EXTS
from .model import eval_transform

# PlantDoc folder name -> our class name
PLANTDOC_TO_OURS = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}
NOT_SURE = 0.6  # the app shows "Not sure" below this confidence


def collect(root):
    items = []
    for split in ("train", "test"):  # we never train on these, so use every image
        for folder in sorted((Path(root) / split).iterdir()):
            if folder.is_dir() and folder.name in PLANTDOC_TO_OURS:
                items += [(str(p), PLANTDOC_TO_OURS[folder.name]) for p in sorted(folder.rglob("*"))
                          if p.suffix.lower() in IMG_EXTS]
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plantdoc-dir", required=True)
    ap.add_argument("--model", default="app/public/model/plant_model.onnx")
    ap.add_argument("--labels", default="app/public/model/labels.json")
    ap.add_argument("--split-file", default="", help="JSON with a \"test\" list of image paths: score only those held-out photos")
    ap.add_argument("--out", default="docs/real_world_results.json")
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    idx = {n: i for i, n in enumerate(labels)}
    items = collect(args.plantdoc_dir)
    if args.split_file:
        keep = set(json.load(open(args.split_file))["test"])
        items = [it for it in items if it[0] in keep]
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    tf = eval_transform()

    y, probs, skipped = [], [], 0
    for k in range(0, len(items), 64):
        chunk, xs = items[k:k + 64], []
        for path, name in chunk:
            try:
                with Image.open(path) as im:
                    xs.append(tf(im.convert("RGB")).numpy())
                y.append(idx[name])
            except Exception:
                skipped += 1
        logits = sess.run(None, {"input": np.stack(xs)})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs.append(e / e.sum(1, keepdims=True))
    p = np.concatenate(probs)
    y = np.array(y)
    pred, conf = p.argmax(1), p.max(1)
    top3 = np.argsort(-p, axis=1)[:, :3]
    healthy = np.array([l.endswith("___healthy") for l in labels])

    acc = float((pred == y).mean())
    top3_acc = float(np.mean([y[i] in top3[i] for i in range(len(y))]))
    classes = sorted(set(y.tolist()))
    recalls = {labels[c]: float((pred[y == c] == c).mean()) for c in classes}
    hy = healthy[y]
    sure = conf >= NOT_SURE
    report = {
        "images": int(len(y)), "skipped_unreadable": skipped, "classes": len(classes),
        "top1_accuracy": acc, "top3_accuracy": top3_acc,
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "healthy_images": int(hy.sum()), "diseased_images": int((~hy).sum()),
        "healthy_called_healthy": float(healthy[pred[hy]].mean()),
        "healthy_exact_class": float((pred[hy] == y[hy]).mean()),
        "diseased_called_healthy_(missed)": float(healthy[pred[~hy]].mean()),
        "not_sure_rate": float((~sure).mean()),
        "accuracy_when_confident": float((pred[sure] == y[sure]).mean()) if sure.any() else None,
        "accuracy_when_not_sure": float((pred[~sure] == y[~sure]).mean()) if (~sure).any() else None,
        "median_confidence_correct": float(np.median(conf[pred == y])),
        "median_confidence_wrong": float(np.median(conf[pred != y])),
        "per_class_recall": recalls,
        "top_confusions": [
            {"true": labels[t], "predicted": labels[q], "count": c}
            for (t, q), c in Counter(zip(y[pred != y].tolist(), pred[pred != y].tolist())).most_common(8)
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_class_recall", "top_confusions")}, indent=2))


if __name__ == "__main__":
    main()
