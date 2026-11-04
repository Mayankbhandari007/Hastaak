"""Feature spec v1 (156 numbers per frame). Must match web/src/features.ts exactly."""
import math
import numpy as np

FEATURE_SPEC_VERSION = 1
FEAT_DIM = 156
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
BROW_KEYS = ["browInnerUp", "browOuterUpLeft", "browOuterUpRight",
             "browDownLeft", "browDownRight", "jawOpen", "mouthPucker"]


def shoulder_frame(pose, a):
    lx, ly, rx, ry = pose[11].x * a, pose[11].y, pose[12].x * a, pose[12].y
    return (lx + rx) / 2, (ly + ry) / 2, (math.hypot(lx - rx, ly - ry) or 1.0)


def assign_hands(hands, pose, a):
    """Slot A = hand nearest pose 15 (person's left wrist), slot B = nearest pose 16."""
    _, _, s = shoulder_frame(pose, a)
    targets = [pose[15], pose[16]]
    pairs = sorted((math.hypot((h[0].x - t.x) * a, h[0].y - t.y), hi, slot)
                   for hi, h in enumerate(hands) for slot, t in enumerate(targets))
    out, used = [None, None], set()
    for d, hi, slot in pairs:
        if d > 0.8 * s or out[slot] is not None or hi in used:
            continue
        out[slot] = hands[hi]
        used.add(hi)
    return out


def pick_face_blend(face_result, pose, a) -> dict:
    if not face_result.face_landmarks:
        return {}
    best = min(range(len(face_result.face_landmarks)), key=lambda i: math.hypot(
        (face_result.face_landmarks[i][1].x - pose[0].x) * a,
        face_result.face_landmarks[i][1].y - pose[0].y))
    return {c.category_name: c.score for c in face_result.face_blendshapes[best]}


def featurize(pose, hand_a, hand_b, blend: dict, a: float) -> np.ndarray:
    out = np.zeros(FEAT_DIM, np.float32)
    if pose is None:
        return out
    cx, cy, s = shoulder_frame(pose, a)
    i = 0

    def put(p):
        nonlocal i
        out[i:i + 3] = ((p.x * a - cx) / s, (p.y - cy) / s, (p.z * a) / s)
        i += 3

    for h in (hand_a, hand_b):
        if h is not None:
            out[i] = 1.0
            i += 1
            for p in h:
                put(p)
        else:
            i += 64
    for k in POSE_IDX:
        put(pose[k])
    for k in BROW_KEYS:
        out[i] = blend.get(k, 0.0)
        i += 1
    return out