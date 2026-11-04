"""Shared MediaPipe setup and the two actions (Python versions of web/src/actions.ts)."""
import math
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODELS = Path(__file__).parent / "models"

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
UPPER_BODY = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


def _model(name: str) -> str:
    path = MODELS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run `npm run models` in the web folder first.")
    return str(path)


def create_landmarkers(num_hands: int = 2, num_poses: int = 1):
    mode = vision.RunningMode.VIDEO
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=_model("hand_landmarker.task")),
        running_mode=mode, num_hands=num_hands))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=_model("face_landmarker.task")),
        running_mode=mode, num_faces=1, output_face_blendshapes=True))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=_model("pose_landmarker_lite.task")),
        running_mode=mode, num_poses=num_poses))
    return hands, face, pose


def to_mp_image(bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_thumbs_up(h) -> bool:
    thumb_up = h[4].y < h[3].y < h[2].y and h[4].y < min(p.y for p in h[5:])
    curled = all(_dist(h[tip], h[0]) < _dist(h[pip], h[0])
                 for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
    return thumb_up and curled


def finger_states(h) -> list[bool]:
    return [
        _dist(h[4], h[9]) > _dist(h[3], h[9]),
        _dist(h[8], h[0]) > _dist(h[6], h[0]),
        _dist(h[12], h[0]) > _dist(h[10], h[0]),
        _dist(h[16], h[0]) > _dist(h[14], h[0]),
        _dist(h[20], h[0]) > _dist(h[18], h[0]),
    ]


def brow_score(categories) -> float:
    s = {c.category_name: c.score for c in categories}
    return (s.get("browInnerUp", 0) + s.get("browOuterUpLeft", 0) + s.get("browOuterUpRight", 0)) / 3


class BrowCalibrator:
    def __init__(self, needed: int = 30):
        self.needed, self.samples, self.baseline = needed, [], None

    def reset(self):
        self.samples, self.baseline = [], None

    def push(self, score: float):
        if self.baseline is not None:
            return
        self.samples.append(score)
        if len(self.samples) >= self.needed:
            self.baseline = sorted(self.samples)[len(self.samples) // 2]

    def is_raised(self, score: float, threshold: float = 0.15) -> bool:
        return self.baseline is not None and score - self.baseline > threshold