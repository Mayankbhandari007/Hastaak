"""Extract feature spec v1 from every video in the manifest (parallel, resumable).
Usage: python extract_include.py [--workers 4] [--max-width 1280]"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def feat_path(feat_root: str, video_rel: str) -> Path:
    return Path(feat_root) / Path(video_rel).with_suffix(".npy")


def extract_one(job):
    video_root, feat_root, rel, max_width = job
    out = feat_path(feat_root, rel)
    if out.exists():
        return rel, "skip", 0

    # Imported inside the worker so each process creates its own MediaPipe instances
    from common import create_landmarkers, to_mp_image
    from features import FEAT_DIM, assign_hands, featurize, pick_face_blend, shoulder_frame

    hands, face, pose = create_landmarkers(num_hands=4, num_poses=3)
    cap = cv2.VideoCapture(str(Path(video_root) / rel))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames, n = [], 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            if w > max_width:
                bgr = cv2.resize(bgr, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
                h, w = bgr.shape[:2]
            a = w / h
            ts = int(n * 1000 / fps)
            n += 1
            img = to_mp_image(bgr)
            p_res = pose.detect_for_video(img, ts)
            h_res = hands.detect_for_video(img, ts)
            f_res = face.detect_for_video(img, ts)
            if not p_res.pose_landmarks:
                frames.append(np.zeros(FEAT_DIM, np.float32))
                continue
            signer = max(p_res.pose_landmarks, key=lambda q: shoulder_frame(q, a)[2])
            hand_a, hand_b = assign_hands(h_res.hand_landmarks, signer, a)
            frames.append(featurize(signer, hand_a, hand_b, pick_face_blend(f_res, signer, a), a))
    finally:
        cap.release()
        for m in (hands, face, pose):
            m.close()

    arr = np.stack(frames) if frames else np.zeros((0, 156), np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file, then rename, so a stopped run never leaves a half-written .npy that a re-run would skip.
    tmp = out.with_suffix(".npy.part")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, out)
    return rel, "ok", len(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/include_manifest.csv")
    ap.add_argument("--videos", default="data/include")
    ap.add_argument("--feats", default="data/include_feats")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--max-width", type=int, default=1280)
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    jobs = [(args.videos, args.feats, r["video"], args.max_width) for r in rows]

    empty, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, j): j[2] for j in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            rel = futures[fut]
            try:
                _, status, n = fut.result()
                if status == "ok" and n == 0:
                    empty.append(rel)
            except Exception as e:
                failed.append((rel, str(e)))

    print(f"Done. {len(failed)} failed, {len(empty)} produced no frames.")
    for rel, err in failed[:10]:
        print(f"  FAILED {rel}: {err}")


if __name__ == "__main__":
    main()