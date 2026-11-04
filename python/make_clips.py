"""Build the sign clip dictionary for the web app: one short, cropped WebM per sign from INCLUDE.
Usage: python make_clips.py [--out ../web/public/clips]
Writes <SIGN>.webm plus clips.json ({sign: file}). INCLUDE is CC BY-NC-SA 4.0: credit AI4Bharat."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import create_landmarkers, to_mp_image
from extract_include import feat_path
from segmenter import hand_presence, sign_span

PAD = 6            # frames kept before and after the sign
OUT_H = 480


def pick_clip(rows, feats):
    """The clip whose sign length is closest to the median for that sign (avoids odd recordings)."""
    cands = []
    for r in rows:
        p = feat_path(feats, r["video"])
        if not p.exists():
            continue
        seq = np.load(p)
        span = sign_span(seq)
        if span and hand_presence(seq[span[0]:span[1]]) > 0.9:
            cands.append((span[1] - span[0], r["video"], span, len(seq)))
    if not cands:
        return None
    med = np.median([c[0] for c in cands])
    return min(cands, key=lambda c: (abs(c[0] - med), c[1]))


def crop_box(frame, pose, ts):
    """Upper-body box around the signer from MediaPipe pose (shoulders)."""
    h, w = frame.shape[:2]
    res = pose.detect_for_video(to_mp_image(frame), ts)     # VIDEO mode: ts must increase per call
    if not res.pose_landmarks:
        return 0, 0, w, h
    lm = res.pose_landmarks[0]
    sx, sy = (lm[11].x + lm[12].x) / 2 * w, (lm[11].y + lm[12].y) / 2 * h
    sw = abs(lm[11].x - lm[12].x) * w
    bw, bh = 3.6 * sw, 3.4 * sw                     # room for hands above the head and out to the sides
    x0, y0 = int(max(0, sx - bw / 2)), int(max(0, sy - 1.9 * sw))
    return x0, y0, int(min(w, x0 + bw)), int(min(h, y0 + bh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/include_manifest.csv")
    ap.add_argument("--videos", default="data/include")
    ap.add_argument("--feats", default="data/include_feats")
    ap.add_argument("--out", default="../web/public/clips")
    args = ap.parse_args()

    by_label = defaultdict(list)
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            by_label[r["label"]].append(r)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, _, pose = create_landmarkers(num_hands=1, num_poses=1)
    index = {}
    for k, (label, rows) in enumerate(sorted(by_label.items())):
        pick = pick_clip(rows, args.feats)
        if not pick:
            print(f"  {label}: no usable clip")
            continue
        _, video, (s, e), n = pick
        cap = cv2.VideoCapture(str(Path(args.videos) / video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = []
        ok, frame = cap.read()
        while ok:
            frames.append(frame)
            ok, frame = cap.read()
        cap.release()
        frames = frames[max(0, s - PAD): min(len(frames), e + PAD)]
        x0, y0, x1, y1 = crop_box(frames[0], pose, 1000 * (k + 1))
        size = (round((x1 - x0) * OUT_H / (y1 - y0)) // 2 * 2, OUT_H)
        path = out / f"{label}.webm"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
        for fr in frames:
            writer.write(cv2.resize(fr[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA))
        writer.release()
        index[label] = path.name
        print(f"  {label}: {video} frames {s}-{e} → {path.name} ({path.stat().st_size / 1e3:.0f} KB)")
    pose.close()
    (out / "clips.json").write_text(json.dumps(index, indent=1))
    print(f"Wrote {len(index)} clips and {out / 'clips.json'}")


if __name__ == "__main__":
    main()
