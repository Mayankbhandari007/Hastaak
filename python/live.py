"""Webcam viewer: landmarks + thumbs-up + raised eyebrows. Keys: q = quit, c = recalibrate."""
import time

import cv2

from common import (HAND_CONNECTIONS, UPPER_BODY, BrowCalibrator, brow_score,
                    create_landmarkers, finger_states, is_thumbs_up, to_mp_image)

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Couldn't open the webcam.")

    hands, face, pose = create_landmarkers()
    brow = BrowCalibrator()
    thumb_frames, last_ts, fps, t_prev = 0, 0, 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = max(last_ts + 1, int(time.monotonic() * 1000))   # must strictly increase
        last_ts = ts
        img = to_mp_image(frame)
        h_res = hands.detect_for_video(img, ts)
        f_res = face.detect_for_video(img, ts)
        p_res = pose.detect_for_video(img, ts)

        H, W = frame.shape[:2]
        px = lambda p: (int(p.x * W), int(p.y * H))

        # Draw on the unmirrored frame
        if p_res.pose_landmarks:
            pl = p_res.pose_landmarks[0]
            for a, b in UPPER_BODY:
                cv2.line(frame, px(pl[a]), px(pl[b]), (235, 183, 133), 3)
        for hl in h_res.hand_landmarks:
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, px(hl[a]), px(hl[b]), (165, 202, 93), 3)
            for p in hl:
                cv2.circle(frame, px(p), 4, (123, 153, 240), -1)

        # Actions
        any_thumb = any(is_thumbs_up(hl) for hl in h_res.hand_landmarks)
        thumb_frames = thumb_frames + 1 if any_thumb else 0
        lines = [f"Thumbs-up: {'YES' if thumb_frames >= 10 else ('hold…' if any_thumb else 'no')}"]

        if f_res.face_blendshapes:
            score = brow_score(f_res.face_blendshapes[0])
            if brow.baseline is None:
                brow.push(score)
                lines.append(f"Eyebrows: calibrating ({len(brow.samples)}/{brow.needed})")
            else:
                state = "RAISED (question)" if brow.is_raised(score) else "neutral"
                lines.append(f"Eyebrows: {state}  score {score:.2f}  base {brow.baseline:.2f}")
        else:
            lines.append("Eyebrows: no face")

        if h_res.hand_landmarks:
            up = finger_states(h_res.hand_landmarks[0])
            lines.append("Fingers up: " + (", ".join(n for n, u in zip(FINGERS, up) if u) or "none"))

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        lines.append(f"FPS: {fps:.0f}")

        # Mirror for display, then write text so it reads normally
        view = cv2.flip(frame, 1)
        for i, text in enumerate(lines):
            cv2.putText(view, text, (16, 32 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(view, text, (16, 32 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("MediaPipe playground (q quits, c recalibrates)", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            brow.reset()

    cap.release()
    cv2.destroyAllWindows()
    for m in (hands, face, pose):
        m.close()


if __name__ == "__main__":
    main()