"""Live sign recognition from the webcam with a trained checkpoint.
Usage: python live_classify.py runs/include50/best.pt     (q quits)"""
import sys
import time

import cv2
import numpy as np
import torch

from common import HAND_CONNECTIONS, create_landmarkers, to_mp_image
from dataset import model_input
from features import assign_hands, featurize, pick_face_blend, shoulder_frame
from model import SignNet
from segmenter import MIN_HAND_PRESENCE, Segmenter, hand_presence


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    ckpt = torch.load(sys.argv[1], map_location="cpu")
    labels = ckpt["labels"]
    model = SignNet(len(labels))
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    hands, face, pose = create_landmarkers(num_hands=2, num_poses=1)
    seg = Segmenter()
    last_ts, result = 0, ["Raise your hands and sign"]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = max(last_ts + 1, int(time.monotonic() * 1000))
        last_ts = ts
        H, W = frame.shape[:2]
        a = W / H
        img = to_mp_image(frame)
        p_res, h_res, f_res = (pose.detect_for_video(img, ts), hands.detect_for_video(img, ts),
                               face.detect_for_video(img, ts))

        if p_res.pose_landmarks:
            signer = max(p_res.pose_landmarks, key=lambda q: shoulder_frame(q, a)[2])
            ha, hb = assign_hands(h_res.hand_landmarks, signer, a)
            fv = featurize(signer, ha, hb, pick_face_blend(f_res, signer, a), a)
            segment = seg.push(fv)
            if segment is not None and hand_presence(segment) < MIN_HAND_PRESENCE:
                result = ["Hand not visible enough: step back, add light"]
                print(result[0])
            elif segment is not None:
                x = torch.from_numpy(model_input(segment)).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)[0]
                top = probs.topk(min(3, len(labels)))
                result = [f"{labels[i]}  {p:.0%}" for p, i in zip(top.values.tolist(), top.indices.tolist())]
                print(" | ".join(result))

        for hl in h_res.hand_landmarks:
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, (int(hl[s].x * W), int(hl[s].y * H)),
                         (int(hl[e].x * W), int(hl[e].y * H)), (165, 202, 93), 3)

        view = cv2.flip(frame, 1)
        status = "SIGNING…" if seg.state == "ACTIVE" else "ready"
        lines = [f"State: {status}"] + result
        for i, text in enumerate(lines):
            y = 36 + i * 34
            cv2.putText(view, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(view, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Live sign recognition (q quits)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()