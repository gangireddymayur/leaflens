# Plant Disease Detection, On Device

Train a plant-disease classifier with PyTorch, then run it **on the phone**: no server, no uploads, and it works offline.

> **Status: work in progress.** The training pipeline and export are written. Results, the model card and the app are filled in as each step is completed from my own runs. Nothing in the results table is invented.

## Why on-device?

A farmer with weak signal cannot wait for a server. The trained model is a single file (a few MB) bundled inside the app. The phone's own processor runs it, so a photo never leaves the device.

## Pipeline

```
Kaggle GPU:  data -> baseline -> fine-tune -> evaluate -> export ONNX -> parity check
App:         ONNX Runtime (browser / Android / iOS via Capacitor) -> prediction on the phone
```

| Step | What it does | Code |
|---|---|---|
| Data | Scan class folders, stratified train/val/test split with a fixed seed | `src/plantdx/data.py` |
| Baseline | Frozen ImageNet backbone, train only the new head | `train.py --mode linear` |
| Fine-tune | Train the whole network with augmentation, label smoothing and cosine LR | `train.py --mode finetune` |
| Evaluate | Accuracy and macro-F1 on a held-out test split, per-class F1, confusion matrix | `train.py`, `metrics.py` |
| Real-world check | Same metrics on real phone photos or another dataset | `evaluate.py` |
| Export | ONNX file, optional int8 copy, and a parity check against PyTorch | `export_onnx.py` |

The model is **MobileNetV3-Large** (ImageNet-pretrained), chosen because it is small and fast on phones.

## Run it on Kaggle

Open `notebooks/kaggle_train.ipynb` in a Kaggle notebook (GPU on, Internet on, add a PlantVillage dataset) and run the cells.

Or from a terminal:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m plantdx.train --data-dir /path/to/plantvillage --mode linear   --epochs 3 --out-dir runs/linear
python -m plantdx.train --data-dir /path/to/plantvillage --mode finetune --epochs 8 --out-dir runs/finetune
python -m plantdx.export_onnx --checkpoint runs/finetune/best.pt --out-dir model --quantize
```

## Run it on a Mac (Apple GPU)

PyTorch does not support Python 3.14 yet, so use Python 3.11 or 3.12. The code picks the best device automatically: CUDA on Kaggle, the Apple GPU (MPS) on a Mac, otherwise CPU.

```bash
pip install uv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
PYTHONPATH=src .venv/bin/python -m plantdx.train --data-dir /path/to/plantvillage --mode linear --epochs 3
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The data splitting and metrics code is unit-tested and runs in CI without a GPU.

## Results

| Model | Test accuracy | Test macro-F1 | Size | Notes |
|---|---|---|---|---|
| Linear baseline | _to be filled from my run_ | | | |
| Fine-tuned | _to be filled from my run_ | | | |
| Fine-tuned, int8 | _to be filled from my run_ | | | |
| Fine-tuned on real phone photos | _to be filled from my run_ | | | the number that matters most |

## Known limits (written before the results, on purpose)

- **PlantVillage photos are taken on plain lab backgrounds.** High accuracy on its test split will overstate real-field performance. That is why a real-world evaluation step exists.
- **The same leaf can appear in several images.** A random split can put near-duplicates in both train and test and inflate scores. I will check for this and report it.
- This is a learning and portfolio project. It is **not** an agronomy tool and should not guide real treatment decisions.

## Roadmap

- [x] Data scanning, stratified split, metrics (unit-tested)
- [x] Training script with baseline and fine-tune modes
- [x] ONNX export with parity check (single self-contained file)
- [x] Smoke-tested end to end on a tiny synthetic dataset (code path only; no real results yet)
- [ ] Run on Kaggle and fill in the results table
- [ ] Real-world evaluation and error analysis
- [ ] Model card
- [ ] Capacitor app (web first, then Android and iOS builds)
- [ ] Android APK in GitHub Releases

## Dataset

PlantVillage (about 54,000 leaf images, 38 classes). See the original project at https://github.com/spMohanty/PlantVillage-Dataset. Check the licence of the copy you use.

## The app: LeafLens (iOS and Android)

`app/` is a Capacitor app (React, Vite) that runs the exported ONNX model **on the phone** with ONNX Runtime Web. No server and no upload: the photo is analysed locally and the app works offline.

- **Scan:** camera or photo library, or one of 8 bundled sample leaves from the held-out test split.
- **Result:** top prediction with confidence, the next two possibilities, and the inference time measured on the device.
- **About this condition:** for each of the 26 diseases, the cause, how it spreads, what to look for and general management steps. It is general plant-health information, and every screen tells the user to confirm with a local agricultural officer.
- **Honest limits:** below 60% confidence the app says it is not sure and shows no advice. A note on every result says the model only knows 38 conditions across 14 crops and was trained on lab photos.

### Preprocessing parity

The app resizes, crops and normalises photos in JavaScript. I checked it against the Python pipeline on 8 held-out photos: same predictions and the same probabilities to three decimal places.

### Build

Requires Node 22+. Put `plant_model.onnx` (from `export_onnx.py`) in `app/public/model/` first.

```bash
cd app
npm install
npm run dev                 # run in a browser
npm run build && npx cap sync
npx cap open ios            # or: npx cap open android
```

### Notes

- The latency shown in the simulator comes from the Mac's processor, so it is not a real phone number. Measure on a real device before quoting a figure.
- The bundled model is the baseline for now; it will be replaced by the fine-tuned model.
- The training data is US-centric. It does not cover staples such as rice, wheat or cotton, so local expert review and local-crop data would be needed before real use.
