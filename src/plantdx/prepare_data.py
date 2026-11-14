"""Turn Hugging Face parquet files into the class-folder layout that train.py expects.

    data/plantvillage/color/<class_name>/<n>.jpg

The parquet files come split into train and test by the uploader. We merge them and let
train.py make its own stratified split from one seed, so every number is reproducible.
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def class_names_from_schema(schema):
    meta = schema.metadata or {}
    raw = meta.get(b"huggingface")
    if not raw:
        raise SystemExit("Parquet file has no Hugging Face metadata; cannot read class names.")
    features = json.loads(raw)["info"]["features"]
    return features["label"]["names"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", required=True, help="folder containing the *.parquet files")
    ap.add_argument("--out-dir", default="data/plantvillage")
    ap.add_argument("--limit-per-class", type=int, default=0, help="debug: stop after N images per class")
    args = ap.parse_args()

    files = sorted(Path(args.parquet_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet files in {args.parquet_dir}")
    out = Path(args.out_dir) / "color"
    names = class_names_from_schema(pq.read_schema(files[0]))
    for n in names:
        (out / n).mkdir(parents=True, exist_ok=True)

    counts = {n: 0 for n in names}
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
            for img, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
                name = names[label]
                if args.limit_per_class and counts[name] >= args.limit_per_class:
                    continue
                ext = ".png" if img["bytes"][:4] == b"\x89PNG" else ".jpg"
                (out / name / f"{counts[name]:05d}{ext}").write_bytes(img["bytes"])
                counts[name] += 1
        print(f"done {f.name}")

    total = sum(counts.values())
    print(f"wrote {total} images in {len(names)} classes to {out}")
    (Path(args.out_dir) / "class_counts.json").write_text(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
