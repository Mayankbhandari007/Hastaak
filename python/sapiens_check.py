"""Compare Meta's Sapiens2 pose model with MediaPipe on the same frames, before deciding to switch trackers
(docs/meta-models.md). Reports speed and how well Sapiens2 finds the fingers MediaPipe finds, and saves
side-by-side pictures (MediaPipe left, Sapiens2 right) to judge by eye.

Setup (once):  pip install "transformers>=5.22" torchvision pillow     (torchvision must match your torch)
Run:           python sapiens_check.py "data/include/Greetings/48. Hello/MVI_0031.MOV" --frames 8
               python sapiens_check.py my_signing.mp4 --model facebook/sapiens2-pose-0.8b
Works on CPU (slow: expect seconds per frame and ~2 GB of RAM for the 0.4B model; close other apps) or on a
free GPU (Kaggle / Colab: copy this file plus common.py and models/*.task, then run the same command)."""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import HAND_CONNECTIONS, create_landmarkers, to_mp_image

TIPS = [4, 8, 12, 16, 20]          # MediaPipe fingertip indices


def person_box(pose_lms, W, H):
    """COCO box (x, y, w, h) around the MediaPipe body, with room for raised hands; whole frame if no body."""
    if not pose_lms:
        return [0.0, 0.0, float(W), float(H)]
    xs = [p.x * W for p in pose_lms[0]]
    ys = [p.y * H for p in pose_lms[0]]
    sw = abs(pose_lms[0][11].x - pose_lms[0][12].x) * W
    x0, y0 = max(0, min(xs) - sw), max(0, min(ys) - sw)
    x1, y1 = min(W, max(xs) + sw), min(H, max(ys) + sw)
    return [x0, y0, x1 - x0, y1 - y0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="a video of someone signing (INCLUDE clip or your own recording)")
    ap.add_argument("--model", default="facebook/sapiens2-pose-0.4b")
    ap.add_argument("--frames", type=int, default=8, help="frames to test, spread over the video")
    ap.add_argument("--min-score", type=float, default=0.3, help="Sapiens2 keypoints below this are ignored")
    ap.add_argument("--out", default="sapiens_out")
    args = ap.parse_args()

    from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} (first run downloads it)…")
    t0 = time.perf_counter()
    processor = Sapiens2ImageProcessor.from_pretrained(args.model)
    model = Sapiens2ForPoseEstimation.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    print(f"  loaded in {time.perf_counter() - t0:.0f} s")

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = np.linspace(total * 0.15, total * 0.85, args.frames).astype(int)
    hands_m, _, pose_m = create_landmarkers(num_hands=2, num_poses=1)
    out = Path(args.out) / Path(args.video).stem
    out.mkdir(parents=True, exist_ok=True)

    mp_ms, sp_ms, tip_err, tips_found, tips_total = [], [], [], 0, 0
    for k, idx in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        H, W = frame.shape[:2]
        img = to_mp_image(frame)
        t0 = time.perf_counter()
        pose = pose_m.detect_for_video(img, 1000 * (k + 1))
        hands = hands_m.detect_for_video(img, 1000 * (k + 1))
        mp_ms.append((time.perf_counter() - t0) * 1000)

        box = person_box(pose.pose_landmarks, W, H)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        inputs = processor(images=rgb, boxes=[[box]], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(pixel_values=inputs["pixel_values"].to(dtype))
        person = processor.post_process_pose_estimation(outputs, boxes=[[box]])[0][0]
        sp_ms.append((time.perf_counter() - t0) * 1000)
        kps = person["keypoints"].float().cpu().numpy()
        good = kps[person["scores"].float().cpu().numpy() > args.min_score]

        # How close is Sapiens2's nearest confident keypoint to each MediaPipe fingertip (in hand sizes)?
        for h in hands.hand_landmarks:
            pts = np.array([[p.x * W, p.y * H] for p in h])
            size = np.linalg.norm(pts[9] - pts[0]) or 1.0          # wrist → middle knuckle
            for t in TIPS:
                tips_total += 1
                if len(good):
                    d = np.linalg.norm(good - pts[t], axis=1).min() / size
                    tip_err.append(d)
                    tips_found += d < 0.25

        left, right = frame.copy(), frame.copy()
        for h in hands.hand_landmarks:
            for a, b in HAND_CONNECTIONS:
                cv2.line(left, (int(h[a].x * W), int(h[a].y * H)), (int(h[b].x * W), int(h[b].y * H)), (93, 202, 165), 2)
        for x, y in good:
            cv2.circle(right, (int(x), int(y)), 3, (60, 120, 255), -1)
        cv2.putText(left, "MediaPipe", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(right, f"Sapiens2 ({len(good)} keypoints > {args.min_score})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(str(out / f"frame_{k:02d}.jpg"), np.hstack([left, right]))
        print(f"  frame {idx}: MediaPipe {mp_ms[-1]:.0f} ms, {len(hands.hand_landmarks)} hands | "
              f"Sapiens2 {sp_ms[-1]:.0f} ms, {len(good)} confident keypoints")

    lines = [
        f"video: {args.video}  model: {args.model}  device: {device}  frames: {len(sp_ms)}",
        f"time per frame: MediaPipe {np.median(mp_ms):.0f} ms, Sapiens2 {np.median(sp_ms):.0f} ms (median)",
        f"MediaPipe fingertips with a Sapiens2 keypoint within 0.25 hand sizes: {tips_found}/{tips_total}",
        f"median fingertip distance: {np.median(tip_err):.2f} hand sizes" if tip_err else "no hands seen",
        "Pictures (MediaPipe left, Sapiens2 right): " + str(out),
    ]
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
