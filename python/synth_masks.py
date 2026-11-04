"""Step 1 of synthetic variants: SAM 3.1 person masks per clip (cached, resumable).

Usage (from python/):
  python synth_masks.py --inventory data/synth_inventory.csv \
      --videos E:/12/Final_Dataset --out data/synth_masks [--limit 1] [--timeout 300]

Reads the Step-0 inventory, calls meta.person_masks for each readable clip, and
writes data/synth_masks/<video rel>.npz with parallel arrays:
  frame (int32), obj (object/str), top/left (int32 box origin),
  mask (object/uint8 h×w local mask, 1 = person). Load with np.load(..., allow_pickle=True).
Skips clips whose .npz already exists; failures append to _failures.csv.
Cost ≈ $0.15/770 frames; 358 G&P clips ≈ 23.8k frames ≈ $4.64.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

COST_PER_FRAME = 0.15 / 770  # project's measured SAM 3.1 person_masks rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="data/synth_inventory.csv")
    ap.add_argument("--videos", default="E:/12/Final_Dataset")
    ap.add_argument("--out", default="data/synth_masks")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N clips (0 = all)")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
    import meta  # the one module that calls Meta
    meta.load_env()
    if not meta.has_key():
        raise SystemExit("MODEL_API_KEY is not set in server/.env — add it first (no credits spent).")

    with open(args.inventory, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if str(r.get("frames", "")).isdigit() and int(r["frames"]) > 0]
    if args.limit:
        rows = rows[:args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures, done_frames = [], 0
    for i, r in enumerate(rows):
        rel = r["video"]
        out = out_dir / Path(rel).with_suffix(".npz")
        if out.exists():
            print(f"[{i + 1}/{len(rows)}] skip (cached): {rel}")
            continue
        try:
            by_frame = meta.person_masks(str(Path(args.videos) / rel), timeout=args.timeout)
        except Exception as e:  # keep going; resume re-tries these (no .npz written)
            print(f"[{i + 1}/{len(rows)}] FAILED {rel}: {e}")
            failures.append((rel, str(e)[:200]))
            continue
        fr, oid, top, left, masks = [], [], [], [], []
        for fi, dets in by_frame.items():
            for o, m, (l, t) in dets:
                fr.append(fi)
                oid.append(o)
                top.append(t)
                left.append(l)
                masks.append(np.asarray(m, np.uint8))
        out.parent.mkdir(parents=True, exist_ok=True)
        # Box-local masks vary in shape frame to frame, so a plain
        # np.array(masks, dtype=object) tries to stack them and fails.
        # Assigning into a pre-allocated object array stores each as-is.
        mask_arr = np.empty(len(masks), dtype=object)
        mask_arr[:] = masks
        np.savez_compressed(out, frame=np.array(fr, np.int32),
                            obj=np.array(oid, dtype=object),
                            top=np.array(top, np.int32), left=np.array(left, np.int32),
                            mask=mask_arr)
        done_frames += int(r["frames"])
        print(f"[{i + 1}/{len(rows)}] {rel}: {len(fr)} detections in {len(by_frame)} frames "
              f"(~${int(r['frames']) * COST_PER_FRAME:.3f} this clip)")

    if failures:
        with open(out_dir / "_failures.csv", "a", newline="") as f:
            w = csv.writer(f)
            for rel, err in failures:
                w.writerow([rel, err])
        print(f"{len(failures)} failures logged to {out_dir / '_failures.csv'} (re-run to retry)")
    print(f"Session frames: {done_frames} (~${done_frames * COST_PER_FRAME:.2f})")


if __name__ == "__main__":
    main()
