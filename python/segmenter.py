"""Sign segmenter shared by live recognition and training. Twin of web/src/segmenter.ts."""
import numpy as np

from features import FEAT_DIM

# Real INCLUDE signs have a tracked hand in ≥81% of their frames. Below this, the hand was lost
# (out of frame, poor light) and the segment is not classified. Same value as web/src/segmenter.ts.
MIN_HAND_PRESENCE = 0.6
TRAINING_FPS = 25          # INCLUDE videos; the default thresholds are tuned for this rate


def low_fps_settings(fps: float) -> dict:
    """Segmenter.cfg updates for a slower camera loop: wait the same time (not the same number of frames)
    before ending a sign, and accept the same minimum sign duration. On INCLUDE test clips at 8 fps this took
    accuracy from 30/33 to 33/33. Motion thresholds stay unscaled (scaling them lost signs).
    Twin of segmenterForFps in web/src/segmenter.ts."""
    k = min(4.0, max(1.0, TRAINING_FPS / max(fps, 1.0)))
    # at least 5 still frames: 3 split a real user's HELLO at its pause by the forehead (no cost on INCLUDE)
    return {"stop_frames": max(5, round(8 / k)), "hold": max(1, round(4 / k)), "min_len": max(3, round(8 / k))}


def hand_presence(seq: np.ndarray) -> float:
    """Fraction of frames in which at least one hand slot is filled."""
    return float(((seq[:, 0] > 0) | (seq[:, 64] > 0)).mean()) if len(seq) else 0.0


class Segmenter:
    """Cuts the feature stream into single signs: starts when hands are up and moving,
    ends after a few still (or lowered) frames."""
    def __init__(self, rest_y=1.8, start_motion=0.05, stop_motion=0.015,
                 stop_frames=8, min_len=8, max_len=90, hold_frames=4):
        self.cfg = dict(rest_y=rest_y, start=start_motion, stop=stop_motion,
                        stop_frames=stop_frames, min_len=min_len, max_len=max_len, hold=hold_frames)
        self.state, self.buf, self.prev, self.still = "REST", [], None, 0
        self.last_hand, self.missing = [None, None], [0, 0]

    def _fill(self, f):
        """Bridge brief tracking dropouts: reuse each hand slot's last values for up to `hold` frames, but only
        for a hand last seen raised (a lowered hand leaving the frame means the sign is over).
        Returns (frame, bridged_only): bridged_only means no hand was really tracked but one was filled in."""
        tracked, filled = f[0] > 0 or f[64] > 0, False
        for k, off in ((0, 0), (1, 64)):
            if f[off] > 0:
                self.last_hand[k], self.missing[k] = f[off:off + 64].copy(), 0
            elif (self.last_hand[k] is not None and self.missing[k] < self.cfg["hold"]
                  and self.last_hand[k][2] < self.cfg["rest_y"]):           # [2] = wrist y
                self.missing[k] += 1
                if not filled:
                    f, filled = f.copy(), True
                f[off:off + 64] = self.last_hand[k]
            else:
                self.last_hand[k] = None
        return f, filled and not tracked

    def _motion(self, f):
        d = 0.0
        if self.prev is not None:
            for pres, x in ((0, 1), (64, 65)):
                if f[pres] > 0 and self.prev[pres] > 0:
                    d += np.hypot(f[x] - self.prev[x], f[x + 1] - self.prev[x + 1])
        self.prev = f
        return d

    def push(self, f):
        c = self.cfg
        f, bridged_only = self._fill(f)
        m = self._motion(f)
        up = (f[0] > 0 and f[2] < c["rest_y"]) or (f[64] > 0 and f[66] < c["rest_y"])
        if self.state == "REST":
            if up and m > c["start"]:
                self.state, self.buf, self.still = "ACTIVE", [f], 0
            return None
        self.buf.append(f)
        if not bridged_only:               # a bridged frame has no real motion: don't let it end the sign
            self.still = self.still + 1 if (m < c["stop"] or not up) else 0
        if self.still >= c["stop_frames"] or len(self.buf) >= c["max_len"]:
            seg = self.buf[: len(self.buf) - self.still]
            self.state, self.buf = "REST", []
            if len(seg) >= c["min_len"]:
                return np.stack(seg)
        return None


def sign_span(seq: np.ndarray, **cfg):
    """(start, end) frame indices of the longest sign the live segmenter finds in a recorded clip, or None.
    Rest frames are appended so a clip that ends mid-sign still closes, as when the signer lowers their hands."""
    seg, best, start = Segmenter(**cfg), None, 0
    tail = np.zeros((seg.cfg["stop_frames"] + seg.cfg["hold"] + 2, FEAT_DIM), np.float32)
    for t, f in enumerate(np.concatenate([seq, tail])):
        was_rest = seg.state == "REST"
        s = seg.push(f)
        if was_rest and seg.state == "ACTIVE":
            start = t
        if s is not None and (best is None or len(s) > best[1] - best[0]):
            best = (start, start + len(s))
    return best


def cut_sign(seq: np.ndarray, **cfg):
    """The longest sign in a recorded clip, cut exactly as the live segmenter would, or None."""
    span = sign_span(seq, **cfg)
    return None if span is None else seq[span[0]:span[1]]
