"""Native camera mode for the Hastaak demo: reads the webcam with OpenCV, runs MediaPipe, the segmenter and
SignNet natively (much faster than in a busy browser), and streams the preview and recognized signs to the page.
The last RECORD_S seconds are kept so "Save recording" can write them to python/recordings/ for debugging.
Run:  python vision_server.py                 (port 8001; the web dev server proxies /vision here)
      python vision_server.py --camera 1      (another webcam)
      python vision_server.py --video clip.mp4 (a looping video file instead of the webcam)"""
import argparse
import asyncio
import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from common import HAND_CONNECTIONS, UPPER_BODY, create_landmarkers, is_thumbs_up, to_mp_image
from dataset import model_input
from features import assign_hands, featurize, pick_face_blend, shoulder_frame
from segmenter import Segmenter, hand_presence, low_fps_settings

MODEL_DIR = Path(__file__).parent.parent / "web" / "public" / "models"     # same model the browser uses
RECORDINGS = Path(__file__).parent / "recordings"
THUMB_S = 0.35          # a thumbs-up held this long sends the sentence
PREVIEW_W = 640         # preview JPEG width
RECORD_S = 20           # seconds kept for "Save recording"
BROW_RAISE = 0.15       # brow level above the neutral baseline that counts as raised (a question)

# Face mesh points drawn on the preview: eyebrows, eyes, outer lips
FACE_LINES = [[70, 63, 105, 66, 107], [336, 296, 334, 293, 300],
              [33, 160, 158, 133, 153, 144, 33], [362, 385, 387, 263, 373, 380, 362],
              [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91, 61]]


class Vision:
    """Camera + recognition loop in a background thread. Events go to a list the SSE endpoint reads."""
    def __init__(self, camera: int, video: str | None = None):
        self.camera, self.video, self.thread, self.running = camera, video, None, False
        self.events, self.lock, self.next_id = [], threading.Lock(), 1
        self.jpeg, self.jpeg_id = None, 0
        self.history = deque()           # (time, annotated JPEG, feature vector or None) for the last RECORD_S s
        meta = json.loads((MODEL_DIR / "signnet.json").read_text())
        self.labels = meta["labels"]
        self.session = ort.InferenceSession(str(MODEL_DIR / "signnet.onnx"))

    def emit(self, **event):
        with self.lock:
            event["id"], self.next_id = self.next_id, self.next_id + 1
            event["t"] = time.time()
            self.events.append(event)
            del self.events[:-500]                       # keep the tail; clients track their position by id

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def classify(self, segment):
        logits = self.session.run(None, {self.session.get_inputs()[0].name: model_input(segment)[None]})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top = np.argsort(-p)[:3]
        return [{"label": self.labels[i], "p": float(p[i])} for i in top]

    def loop(self):
        if self.video:                                   # a file instead of the webcam (testing, demo fallback)
            cap = cv2.VideoCapture(self.video)
        else:
            cap = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(self.camera)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # always the newest frame, never a growing backlog
        if not cap.isOpened():
            self.emit(type="error", text=f"Couldn't open {self.video or f'camera {self.camera}'} (is another app using it?)")
            self.running = False
            return
        file_dt = 1 / (cap.get(cv2.CAP_PROP_FPS) or 25) if self.video else 0
        next_t = time.perf_counter()
        hands_m, face_m, pose_m = create_landmarkers(num_hands=2, num_poses=1)
        seg = Segmenter()
        brow_samples, baseline = [], None
        last_pose = last_face = None
        last_ts, n, fps_n, fps_t0 = 0, 0, 0, time.perf_counter()
        times = {"pose": 0.0, "hands": 0.0, "face": 0.0}
        thumb_since, thumb_sent, last_state = 0.0, False, None
        self.emit(type="status", text="Keep a neutral face for a second…")

        def timed(key, fn):
            t0 = time.perf_counter()
            r = fn()
            dt = (time.perf_counter() - t0) * 1000
            times[key] = times[key] * 0.8 + dt * 0.2 if times[key] else dt
            return r

        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    if self.video:                       # loop the file
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                if file_dt:                              # play a file in real time, like a camera:
                    next_t += file_dt                    # drop frames when processing falls behind
                    while next_t < time.perf_counter() - file_dt and cap.grab():
                        next_t += file_dt
                    time.sleep(max(0.0, next_t - time.perf_counter()))
                ts = max(last_ts + 1, int(time.monotonic() * 1000))
                last_ts = ts
                H, W = frame.shape[:2]
                a = W / H
                img = to_mp_image(frame)
                # Hands every frame; pose every other, face every third (same schedule as the browser)
                if n % 2 == 0 or last_pose is None:
                    last_pose = timed("pose", lambda: pose_m.detect_for_video(img, ts))
                h_res = timed("hands", lambda: hands_m.detect_for_video(img, ts))
                if n % 3 == 0 or last_face is None:
                    last_face = timed("face", lambda: face_m.detect_for_video(img, ts))
                n += 1

                signer = max(last_pose.pose_landmarks, key=lambda q: shoulder_frame(q, a)[2]) if last_pose.pose_landmarks else None
                fv, brow = None, None
                if signer is not None:
                    ha, hb = assign_hands(h_res.hand_landmarks, signer, a)
                    fv = featurize(signer, ha, hb, pick_face_blend(last_face, signer, a), a)
                    brow_now = float(fv[149:152].mean())
                    if baseline is None and last_face.face_landmarks:
                        brow_samples.append(brow_now)
                        if len(brow_samples) >= 30:
                            baseline = float(np.median(brow_samples))
                            self.emit(type="status", text="Ready: sign, then lower your hands")
                    if baseline is not None and last_face.face_landmarks:
                        brow = brow_now - baseline

                    thumb = any(is_thumbs_up(h) for h in h_res.hand_landmarks)
                    thumb_since = (thumb_since or time.perf_counter()) if thumb else 0.0
                    if not thumb:
                        thumb_sent = False
                    elif not thumb_sent and time.perf_counter() - thumb_since >= THUMB_S:
                        thumb_sent = True
                        seg = Segmenter()
                        seg.cfg.update(low_fps_settings(self.fps))
                        self.emit(type="thumb")

                    segment = seg.push(fv)
                    if seg.state != last_state:
                        last_state = seg.state
                        self.emit(type="state", state=seg.state)
                    if segment is not None:
                        levels = np.sort(segment[:, 149:152].mean(1))
                        question = baseline is not None and levels[int(len(levels) * 0.8)] - baseline > BROW_RAISE
                        self.emit(type="sign", candidates=self.classify(segment), question=bool(question),
                                  presence=hand_presence(segment), frames=len(segment))
                else:
                    if last_state != "NOBODY":
                        last_state = "NOBODY"
                        self.emit(type="state", state="NOBODY")

                fps_n += 1
                if time.perf_counter() - fps_t0 >= 1:
                    self.fps = fps_n / (time.perf_counter() - fps_t0)
                    fps_n, fps_t0 = 0, time.perf_counter()
                    if seg.state == "REST":
                        seg.cfg.update(low_fps_settings(self.fps))
                    self.emit(type="fps", fps=round(self.fps), ms={k: round(v) for k, v in times.items()})

                faces = last_face.face_landmarks[:1] if last_face else []
                status = (f"{self.fps:.0f} fps | body {'yes' if signer else 'NO'} | hands {len(h_res.hand_landmarks)} | "
                          f"face {'yes' if faces else 'NO'} | brows "
                          + ("calibrating" if brow is None else "RAISED" if brow > BROW_RAISE else "neutral")
                          + f" | {seg.state if signer else '-'}")
                self.publish_preview(frame, signer, h_res.hand_landmarks, faces, status, fv, W, H)
        finally:
            cap.release()
            for m in (hands_m, face_m, pose_m):
                m.close()
            self.emit(type="status", text="Native camera stopped")

    fps = 25.0

    def publish_preview(self, frame, signer, hands, faces, status, fv, W, H):
        """Draw everything that is tracked (body, face, hands), mirror, stream, and keep it for 'Save recording'."""
        px = lambda p: (int(p.x * W), int(p.y * H))                            # noqa: E731
        if signer is not None:
            for s, e in UPPER_BODY:
                cv2.line(frame, px(signer[s]), px(signer[e]), (235, 183, 133), 4)
            for k in (0, 11, 12, 13, 14, 15, 16):
                cv2.circle(frame, px(signer[k]), 6, (221, 138, 55), -1)
        for f in faces:
            for line in FACE_LINES:
                cv2.polylines(frame, [np.array([px(f[i]) for i in line], np.int32)], False, (236, 169, 175), 2)
        for h in hands:
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, px(h[s]), px(h[e]), (165, 202, 93), 3)
            for p in h:
                cv2.circle(frame, px(p), 3, (123, 153, 240), -1)
        small = cv2.resize(cv2.flip(frame, 1), (PREVIEW_W, round(PREVIEW_W * H / W)), interpolation=cv2.INTER_AREA)
        cv2.rectangle(small, (0, 0), (PREVIEW_W, 24), (0, 0, 0), -1)
        cv2.putText(small, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        self.jpeg, self.jpeg_id = buf.tobytes(), self.jpeg_id + 1
        now = time.time()
        with self.lock:
            self.history.append((now, self.jpeg, None if fv is None else fv.copy()))
            while self.history and self.history[0][0] < now - RECORD_S:
                self.history.popleft()

    def save(self) -> dict:
        """Write the last RECORD_S seconds: preview.webm (what the camera and tracker saw), features.json
        (the exact numbers the model got, recorder format) and events.json (what was recognized)."""
        with self.lock:
            hist, events = list(self.history), list(self.events)
        if len(hist) < 2:
            raise RuntimeError("Nothing to save yet: start the camera first")
        t0, t1 = hist[0][0], hist[-1][0]
        fps = (len(hist) - 1) / max(t1 - t0, 1e-3)
        out = RECORDINGS / datetime.now().strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        first = cv2.imdecode(np.frombuffer(hist[0][1], np.uint8), cv2.IMREAD_COLOR)
        writer = cv2.VideoWriter(str(out / "preview.webm"), cv2.VideoWriter_fourcc(*"VP80"), fps,
                                 (first.shape[1], first.shape[0]))
        for _, jpg, _ in hist:
            writer.write(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
        writer.release()
        zero = [0.0] * 156
        (out / "features.json").write_text(json.dumps({
            "featureSpecVersion": 1, "label": None, "fps": round(fps, 2), "frameCount": len(hist),
            "times": [round(t - t0, 3) for t, _, _ in hist],
            "frames": [zero if fv is None else [round(float(v), 5) for v in fv] for _, _, fv in hist]}))
        (out / "events.json").write_text(json.dumps(
            [{**e, "t": round(e["t"] - t0, 3)} for e in events if e["t"] >= t0 and e["type"] != "fps"], indent=1))
        return {"folder": str(out), "seconds": round(t1 - t0, 1), "frames": len(hist), "fps": round(fps, 1)}


app = FastAPI(title="Hastaak native vision")
vision: Vision


@app.get("/vision/health")
def health():
    return {"running": vision.running, "labels": vision.labels}


@app.post("/vision/start")
def start():
    vision.start()
    return {"running": True}


@app.post("/vision/stop")
def stop():
    vision.stop()
    return {"running": False}


@app.post("/vision/save")
def save():
    try:
        return vision.save()
    except RuntimeError as e:
        return {"error": str(e)}


@app.get("/vision/stream.mjpg")
async def stream():
    async def frames():
        last = -1
        while True:
            if vision.jpeg is not None and vision.jpeg_id != last:
                last = vision.jpeg_id
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + vision.jpeg + b"\r\n"
            await asyncio.sleep(0.03)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/vision/events")
async def events():
    async def gen():
        with vision.lock:
            last = vision.events[-1]["id"] if vision.events else 0
        while True:
            with vision.lock:
                new = [e for e in vision.events if e["id"] > last]
            for e in new:
                last = e["id"]
                yield f"data: {json.dumps(e)}\n\n"
            await asyncio.sleep(0.03)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="read this video file (looped) instead of the webcam")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    vision = Vision(args.camera, args.video)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
