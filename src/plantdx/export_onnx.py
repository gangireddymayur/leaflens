"""Export the trained model to ONNX and check it matches PyTorch.

The ONNX file is what the app loads (ONNX Runtime Web in the browser, ONNX Runtime Mobile
on a phone). Nothing is sent to a server: inference runs on the device.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .model import IMAGE_SIZE, build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="model")
    ap.add_argument("--quantize", action="store_true", help="also write an int8 dynamic-quantised copy")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    onnx_path = out / "plant_model.onnx"
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    (out / "labels.json").write_text(json.dumps(classes, indent=2))

    # Newer PyTorch writes the weights to a separate .data file. Embed them so the app
    # only has to load one self-contained file, and so the size we report is the real size.
    onnx.save_model(onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False)
    (out / "plant_model.onnx.data").unlink(missing_ok=True)

    # Parity check on random inputs: the exported graph must agree with PyTorch.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    batch = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        expected = model(batch).numpy()
    got = session.run(None, {"input": batch.numpy()})[0]
    max_diff = float(np.abs(expected - got).max())
    agree = float((expected.argmax(1) == got.argmax(1)).mean())
    report = {"onnx_mb": round(onnx_path.stat().st_size / 1e6, 2), "max_abs_diff": max_diff, "top1_agreement": agree}

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q_path = out / "plant_model.int8.onnx"
        quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QUInt8)
        q_session = ort.InferenceSession(str(q_path), providers=["CPUExecutionProvider"])
        q_got = q_session.run(None, {"input": batch.numpy()})[0]
        report["int8_mb"] = round(q_path.stat().st_size / 1e6, 2)
        report["int8_top1_agreement_with_fp32"] = float((expected.argmax(1) == q_got.argmax(1)).mean())
        report["note"] = "Agreement is measured on random inputs. Re-run evaluation on real images before trusting the int8 model."

    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if max_diff > 1e-3:
        raise SystemExit(f"Parity check failed: max abs diff {max_diff:.5f}")


if __name__ == "__main__":
    main()
