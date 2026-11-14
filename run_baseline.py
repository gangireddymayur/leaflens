"""Baseline: frozen pretrained backbone, train only the new head (3 epochs).

Right-click this file in PyCharm and choose Run. Progress prints in the Run window and is
also saved to runs/linear/train.log.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from plantdx.train import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [
        "train",
        "--data-dir", str(ROOT / "data" / "plantvillage"),
        "--mode", "linear",
        "--epochs", "3",
        "--workers", "4",
        "--out-dir", str(ROOT / "runs" / "linear"),
    ]
    main()
