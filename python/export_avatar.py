"""Export each sign's motion for the 2D signing avatar in the web app (web/src/avatar.ts).
Usage: python export_avatar.py [--out ../web/public/avatar]
Uses the same INCLUDE clip per sign as make_clips.py, cut to the sign ± a few frames. Short hand-tracking gaps are
filled by interpolation and the motion is smoothed. Coordinates are feature spec v1 units: shoulder widths from
the shoulder midpoint, y down, camera view (so the avatar faces the viewer like the real signer).
Writes <SIGN>.json ({"fps", "frames": [[pose 7×(x,y)], [left hand 21×(x,y)] or [], [right hand] or []]})
and index.json."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from extract_include import feat_path
from make_clips import PAD, pick_clip

POSE_XY = [128 + 3 * k + d for k in range(7) for d in (0, 1)]     # nose, shoulders, elbows, wrists: x, y


def hand_xy(seq, pres):
    """(frames × 42) x,y of one hand slot with gaps interpolated; None if the hand is never seen."""
    seen = seq[:, pres] > 0
    if not seen.any():
        return None
    cols = [pres + 1 + 3 * k + d for k in range(21) for d in (0, 1)]
    xy = seq[:, cols].copy()
    idx = np.flatnonzero(seen)
    for c in range(xy.shape[1]):                      # linear between seen frames, held at the ends
        xy[:, c] = np.interp(np.arange(len(seq)), idx, xy[idx, c])
    return xy


def smooth(a, k=3):
    """Centred moving average over time (k frames), edges padded."""
    pad = np.pad(a, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.stack([pad[i:i + len(a)] for i in range(k)]).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/include_manifest.csv")
    ap.add_argument("--feats", default="data/include_feats")
    ap.add_argument("--out", default="../web/public/avatar")
    args = ap.parse_args()

    by_label = defaultdict(list)
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            by_label[r["label"]].append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for label, rows in sorted(by_label.items()):
        pick = pick_clip(rows, args.feats)
        if not pick:
            continue
        _, video, (s, e), _ = pick
        seq = np.load(feat_path(args.feats, video))
        seq = seq[max(0, s - PAD): min(len(seq), e + PAD)]
        body = seq[:, 128:149].any(1)
        if not body.all():                               # a frame without a body: hold the last good one
            idx = np.flatnonzero(body)
            seq = seq[np.maximum.accumulate(np.where(body, np.arange(len(seq)), 0)).clip(idx[0])]
        parts = [smooth(seq[:, POSE_XY])] + [None if h is None else smooth(h) for h in (hand_xy(seq, 0), hand_xy(seq, 64))]
        frames = [[[round(float(v), 3) for v in p[i]] if p is not None else [] for p in parts] for i in range(len(seq))]
        (out / f"{label}.json").write_text(json.dumps({"fps": 25, "frames": frames}, separators=(",", ":")))
        index[label] = f"{label}.json"
        print(f"  {label}: {video} → {len(frames)} frames, hands: "
              f"{'left ' if parts[1] is not None else ''}{'right' if parts[2] is not None else ''}")
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"Wrote {len(index)} signs to {out}")


if __name__ == "__main__":
    main()
