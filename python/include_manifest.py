"""Scan INCLUDE videos and write a manifest with labels and a stratified split.
Usage: python include_manifest.py [--root data/include] [--out data/include_manifest.csv]"""
import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

EXTS = {".mov", ".mp4", ".avi", ".mkv"}


def label_from(folder: str) -> str:
    """'48. Hello' -> 'HELLO', '12. thank you' -> 'THANK_YOU'."""
    return re.sub(r"^\d+\.\s*", "", folder).strip().upper().replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/include")
    ap.add_argument("--out", default="data/include_manifest.csv")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    by_label = defaultdict(list)
    for v in root.rglob("*"):
        if v.is_file() and v.suffix.lower() in EXTS:
            rel = v.relative_to(root)
            by_label[label_from(v.parent.name)].append((rel.as_posix(), rel.parts[0]))

    if not by_label:
        raise SystemExit(f"No videos found under {root}. Did you unzip INCLUDE there?")

    rng = random.Random(args.seed)
    rows, small = [], []
    for label, vids in sorted(by_label.items()):
        vids.sort()
        rng.shuffle(vids)
        n = len(vids)
        if n < 3:
            small.append(label)
        n_test = max(1, round(n * args.test)) if n >= 3 else 0
        n_val = max(1, round(n * args.val)) if n >= 3 else 0
        for i, (rel, category) in enumerate(vids):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            rows.append({"video": rel, "label": label, "category": category, "split": split})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "label", "category", "split"])
        w.writeheader()
        w.writerows(rows)

    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print(f"{len(rows)} videos, {len(by_label)} signs → {out}")
    print(f"  train {counts['train']} · val {counts['val']} · test {counts['test']}")
    if small:
        print(f"  {len(small)} signs have fewer than 3 videos (train only): {', '.join(small[:10])}…")


if __name__ == "__main__":
    main()