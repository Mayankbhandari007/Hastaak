"""Turn a video into a (frames × 156) feature file using spec v1.
Usage: python extract.py path/to/video.mp4 [output.npy]"""
import sys
from pathlib import Path

import cv2
import numpy as np

from common import create_landmarkers, to_mp_image
from features import FEAT_DIM, assign_hands, featurize, pick_face_blend, shoulder_frame


def extract(path: Path) -> np.ndarray:
    hands, face, pose = create_landmarkers(num_hands=4, num_poses=3)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Couldn't open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, n = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
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
        signer = max(p_res.pose_landmarks, key=lambda p: shoulder_frame(p, a)[2])   # largest person
        hand_a, hand_b = assign_hands(h_res.hand_landmarks, signer, a)
        frames.append(featurize(signer, hand_a, hand_b, pick_face_blend(f_res, signer, a), a))
    cap.release()
    for m in (hands, face, pose):
        m.close()
    return np.stack(frames) if frames else np.zeros((0, FEAT_DIM), np.float32)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".npy")
    feats = extract(src)
    np.save(dst, feats)
    T = len(feats)
    if T:
        print(f"Saved {dst}  shape {feats.shape}")
        print(f"  signer found:      {np.mean(feats[:, 11:14].any(axis=1) | (feats[:, 128:149] != 0).any(axis=1)) * 100:.0f}% of frames")
        print(f"  slot A hand (left wrist):  {np.mean(feats[:, 0] > 0) * 100:.0f}% of frames")
        print(f"  slot B hand (right wrist): {np.mean(feats[:, 64] > 0) * 100:.0f}% of frames")
    else:
        print("No frames read.")