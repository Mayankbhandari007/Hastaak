"""Train SignNet on INCLUDE features plus your own browser recordings (data/own/**/*.json).
Recordings labelled NOTHING (ordinary movement, no sign) add a "no sign" class; use --no-nothing to leave it out.
Examples:
  python train.py --max-classes 50 --epochs 60 --name include50
  python train.py --epochs 80 --name include_full
  python train.py --labels my_vocab.txt --name kiosk_vocab"""
import argparse
import csv
import json
import random
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import NOTHING, SignDataset
from extract_include import feat_path
from model import SignNet

OWN_DIR = Path("data/own")


def own_recordings():
    """(path, label, split) for browser recordings. The label comes from the file (set in the recorder)
    or its folder name; the split is fixed per file name (70/15/15)."""
    out = []
    for p in sorted(OWN_DIR.rglob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("featureSpecVersion") != 1 or not rec.get("frames"):
            continue
        label = (rec.get("label") or p.parent.name).strip().upper().replace(" ", "_")
        if not label or p.parent == OWN_DIR and not rec.get("label"):
            continue
        h = zlib.crc32(p.name.encode()) % 100
        out.append((str(p), label, "train" if h < 70 else "val" if h < 85 else "test"))
    return out


def build_dataset(items, l2i, train: bool, seed: int = 0):
    return SignDataset(items, l2i, train=train, seed=seed)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_items(manifest, feats, labels_filter, use_own=True):
    items = {"train": [], "val": [], "test": []}
    with open(manifest) as f:
        for r in csv.DictReader(f):
            if labels_filter and r["label"] not in labels_filter:
                continue
            p = feat_path(feats, r["video"])
            if p.exists() and np.load(p, mmap_mode="r").shape[0] > 0:
                items[r["split"]].append((str(p), r["label"]))
    for path, label, split in (own_recordings() if use_own else []):
        if not labels_filter or label in labels_filter:
            items[split].append((path, label))
    return items


def choose_labels(args):
    with open(args.manifest) as f:
        all_labels = sorted({r["label"] for r in csv.DictReader(f)} |
                            {l for _, l, _ in (own_recordings() if not args.no_own else []) if l != NOTHING})
    if args.labels:
        wanted = {l.strip().upper() for l in Path(args.labels).read_text().splitlines() if l.strip()}
        missing = wanted - set(all_labels)
        if missing:
            print(f"Warning: not in INCLUDE: {sorted(missing)}")
        return sorted(wanted & set(all_labels))
    if args.max_classes:
        return sorted(random.Random(args.seed).sample(all_labels, min(args.max_classes, len(all_labels))))
    return all_labels


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/include_manifest.csv")
    ap.add_argument("--feats", default="data/include_feats")
    ap.add_argument("--labels", help="text file, one sign per line (subset of INCLUDE labels)")
    ap.add_argument("--max-classes", type=int, default=0, help="random subset of N signs (0 = all)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="run")
    ap.add_argument("--no-nothing", action="store_true", help="ignore NOTHING (no sign) recordings")
    ap.add_argument("--no-own", action="store_true", help="train only on the manifest's videos (skip data/own)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    labels = set(choose_labels(args)) | (set() if args.no_nothing else {NOTHING})
    items = load_items(args.manifest, args.feats, labels, use_own=not args.no_own)
    labels = sorted({l for split in items.values() for _, l in split})    # only labels with data
    l2i = {l: i for i, l in enumerate(labels)}
    own = sum(p.endswith(".json") for split in items.values() for p, _ in split)
    print(f"{len(labels)} classes · train {len(items['train'])} · val {len(items['val'])} · "
          f"test {len(items['test'])} (incl. {own} own recordings)")

    train_dl = DataLoader(build_dataset(items["train"], l2i, train=True, seed=args.seed),
                          batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(build_dataset(items["val"], l2i, train=False), batch_size=256)

    device = pick_device()
    model = SignNet(len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    out_dir = Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(y)
        sched.step()
        val_acc = accuracy(model, val_dl, device)
        flag = ""
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "labels": labels, "feature_spec_version": 1},
                       out_dir / "best.pt")
            flag = "  ← best"
        print(f"epoch {epoch:3d}  loss {total_loss / len(train_dl.dataset):.3f}  val acc {val_acc:.3f}{flag}")

    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"Best validation accuracy {best:.3f}. Checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()