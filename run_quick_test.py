"""Quick test: about 2 minutes, a few images per class. Checks that everything works.

Right-click this file in PyCharm and choose Run. Progress prints in the Run window and is
also saved to runs/quick_test/train.log.
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
        "--mode", "finetune",
        "--epochs", "1",
        "--workers", "4",
        "--out-dir", str(ROOT / "runs" / "quick_test"),
        '--limit-per-class',
        '50',
    ]
    main()
