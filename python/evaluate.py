"""Test-set accuracy, weakest signs, and most common confusions.
Usage: python evaluate.py runs/include50/best.pt"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import NOTHING
from model import SignNet
from train import build_dataset, load_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--manifest", default="data/include_manifest.csv")
    ap.add_argument("--feats", default="data/include_feats")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    labels = ckpt["labels"]
    l2i = {l: i for i, l in enumerate(labels)}
    model = SignNet(len(labels))
    model.load_state_dict(ckpt["model"])
    model.eval()

    test = load_items(args.manifest, args.feats, set(labels))["test"]
    dl = DataLoader(build_dataset(test, l2i, train=False), batch_size=256)

    ys, top3 = [], []
    with torch.no_grad():
        for x, y in dl:
            top3.append(model(x).topk(min(3, len(labels)), dim=1).indices)
            ys.append(y)
    y = torch.cat(ys).numpy()
    t3 = torch.cat(top3).numpy()
    pred = t3[:, 0]

    top1_acc = (pred == y).mean()
    top3_acc = np.any(t3 == y[:, None], axis=1).mean()
    print(f"Test videos: {len(y)}   top-1 accuracy {top1_acc:.3f}   top-3 accuracy {top3_acc:.3f}")
    if NOTHING in l2i:
        n = l2i[NOTHING]
        real = y != n
        print(f"  real signs: {real.sum()}, accuracy {(pred[real] == y[real]).mean():.3f}, "
              f"wrongly called NOTHING {(pred[real] == n).sum()}")
        print(f"  NOTHING examples: {(~real).sum()}, correctly rejected {(pred[~real] == n).mean():.3f}")

    per_class = {labels[c]: (pred[y == c] == c).mean() for c in np.unique(y)}
    print("\nWeakest signs:")
    for name, acc in sorted(per_class.items(), key=lambda kv: kv[1])[:10]:
        print(f"  {name:25s} {acc:.2f}")

    confusions = Counter((labels[a], labels[b]) for a, b in zip(y, pred) if a != b)
    print("\nMost common confusions (true → predicted):")
    for (a, b), n in confusions.most_common(10):
        print(f"  {a} → {b}  ×{n}")

    out = Path(args.checkpoint).parent / "confusions.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true", "predicted", "count"])
        for (a, b), n in confusions.most_common():
            w.writerow([a, b, n])
    print(f"\nAll confusions saved to {out}")


if __name__ == "__main__":
    main()