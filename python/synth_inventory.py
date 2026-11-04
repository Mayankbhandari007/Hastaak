"""Inventory clips for synthetic-variant planning: frames, fps, size per video (header reads only).

Usage (from python/):
  python synth_inventory.py --manifest data/final_manifest.csv \
      --videos E:/12/Final_Dataset --filter Greetings Pronouns \
      --out data/synth_inventory.csv

Writes one row per manifest row: video,label,category,split,frames,fps,width,height,ok.
Prints per-category totals plus the SAM 3.1 cost estimate at the project's
measured rate (~$0.15 per ~770 frames).
"""
import argparse
import csv
from pathlib import Path

import cv2

COST_PER_FRAME = 0.15 / 770  # project's measured SAM 3.1 person_masks rate


def probe(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                float(cap.get(cv2.CAP_PROP_FPS)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/final_manifest.csv")
    ap.add_argument("--videos", default="E:/12/Final_Dataset")
    ap.add_argument("--filter", nargs="*", default=[],
                    help="only rows whose video rel starts with one of these (e.g. Greetings Pronouns)")
    ap.add_argument("--out", default="data/synth_inventory.csv")
    args = ap.parse_args()

    with open(args.manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if not args.filter or any(r["video"].startswith(p) for p in args.filter)]

    out_rows, missing = [], []
    for r in rows:
        p = Path(args.videos) / r["video"]
        if not p.exists():
            missing.append(r["video"])
            out_rows.append({**r, "frames": "", "fps": "", "width": "", "height": "", "ok": "missing"})
            continue
        got = probe(p)
        if got is None:
            missing.append(r["video"])
            out_rows.append({**r, "frames": "", "fps": "", "width": "", "height": "", "ok": "unreadable"})
            continue
        frames, fps, w, h = got
        out_rows.append({**r, "frames": frames, "fps": round(fps, 2), "width": w, "height": h,
                         "ok": "ok" if frames > 0 else "zero-frames-header"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "label", "category", "split",
                                          "frames", "fps", "width", "height", "ok"])
        w.writeheader()
        w.writerows(out_rows)

    ok_rows = [r for r in out_rows if isinstance(r["frames"], int) and r["frames"] > 0]
    total_frames = sum(r["frames"] for r in ok_rows)
    print(f"Rows: {len(out_rows)} ({len(ok_rows)} readable, {len(missing)} missing/unreadable)")
    by_cat = {}
    for r in ok_rows:
        by_cat.setdefault(r["category"], [0, 0])
        by_cat[r["category"]][0] += 1
        by_cat[r["category"]][1] += r["frames"]
    for cat, (n, fr) in sorted(by_cat.items()):
        print(f"  {cat}: {n} clips, {fr} frames → SAM est. ${fr * COST_PER_FRAME:.2f}")
    print(f"Total {total_frames} frames → SAM est. ${total_frames * COST_PER_FRAME:.2f}")
    if missing:
        print(f"  ({len(missing)} problem files, e.g. {missing[0]})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
