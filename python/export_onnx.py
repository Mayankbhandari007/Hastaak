"""Export a checkpoint to ONNX + metadata for the browser (and PyTorch weights for vision_server.py), and verify it.
Usage: python export_onnx.py runs/include_full/best.pt [--out ../web/public/models]"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from model import SignNet


def export(model, dummy, path: Path) -> str:
    """Try the new (dynamo) exporter, fall back to the legacy one. Always writes a single file."""
    for stale in (path, path.with_name(path.name + ".data")):
        stale.unlink(missing_ok=True)
    try:
        torch.onnx.export(model, (dummy,), str(path), input_names=["x"], output_names=["logits"],
                          opset_version=18, dynamo=True, external_data=False)
        return "new exporter (opset 18)"
    except Exception as e:
        print(f"New exporter failed ({type(e).__name__}: {e}).\nTrying the legacy exporter…")
        torch.onnx.export(model, (dummy,), str(path), input_names=["x"], output_names=["logits"],
                          opset_version=17, dynamo=False)
        return "legacy exporter (opset 17)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="../web/public/models", help="the web app's public/models folder")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    labels = ckpt["labels"]
    model = SignNet(len(labels))
    model.load_state_dict(ckpt["model"])
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "signnet.onnx"
    dummy = torch.randn(1, 32, 156)

    with torch.no_grad():
        method = export(model, dummy, onnx_path)
        expected = model(dummy).numpy()

    if onnx_path.with_name(onnx_path.name + ".data").exists():
        raise SystemExit("Export wrote a separate .data file; the browser needs a single .onnx file.")

    (out / "signnet.json").write_text(json.dumps(
        {"labels": labels, "featureSpecVersion": ckpt.get("feature_spec_version", 1)}, indent=1))
    # The same weights for PyTorch, which the native camera server (vision_server.py) runs
    torch.save({"model": model.state_dict(), "labels": labels,
                "feature_spec_version": ckpt.get("feature_spec_version", 1)}, out / "signnet.pt")

    sess = ort.InferenceSession(str(onnx_path))
    got = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]
    diff = float(np.abs(got - expected).max())
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"Exported {onnx_path} with the {method}: {len(labels)} signs, {size_mb:.1f} MB")
    print(f"Max difference vs PyTorch: {diff:.2e}  {'(OK)' if diff < 1e-3 else '(WARNING: larger than expected)'}")
    print(f"Wrote {out / 'signnet.json'} and {out / 'signnet.pt'}")


if __name__ == "__main__":
    main()