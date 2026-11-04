"""Compare checkpoints on clean test clips and on webcam-style variants of the same test clips, per category.
Run from python/: python <this> runs/a/best.pt runs/b/best.pt"""
import csv
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from extract_include import feat_path  # noqa: E402
from model import SignNet  # noqa: E402
from train import build_dataset  # noqa: E402

FEATS = "data/include_data_feats"


def rows(manifest):
    with open(manifest, newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def score(ckpt_path, test_rows):
    ck = torch.load(ckpt_path, map_location="cpu")
    labels = ck["labels"]
    l2i = {l: i for i, l in enumerate(labels)}
    m = SignNet(len(labels))
    m.load_state_dict(ck["model"])
    m.eval()
    test_rows = [r for r in test_rows if r["label"] in l2i and feat_path(FEATS, r["video"]).exists()]
    items = [(str(feat_path(FEATS, r["video"])), r["label"]) for r in test_rows]
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(build_dataset(items, l2i, train=False), batch_size=256):
            preds += m(x).argmax(1).tolist()
    by = defaultdict(lambda: [0, 0])
    for r, p in zip(test_rows, preds):
        cat = r["category"].split("_")[0]
        ok = labels[p] == r["label"]
        for key in (cat, "all"):
            by[key][0] += ok
            by[key][1] += 1
    return by


def main():
    sets = {"clean test": rows("data/final_manifest.csv"), "webcam variants of test": rows("data/synth_test_manifest.csv")}
    for name, test_rows in sets.items():
        print(f"\n{name}")
        for ck in sys.argv[1:]:
            by = score(ck, test_rows)
            cells = "  ".join(f"{k} {v[0]}/{v[1]} ({v[0] / v[1]:.1%})" for k, v in sorted(by.items()))
            print(f"  {ck:38s} {cells}")


if __name__ == "__main__":
    main()
