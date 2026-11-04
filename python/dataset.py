"""Sequence preparation and augmentation for feature spec v1."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from segmenter import cut_sign

FEAT_DIM, SEQ_LEN = 156, 32
NOTHING = "NOTHING"      # the "no sign" class

# x indices of every (x, y, z) triplet: slot A hand, slot B hand, 7 pose points
X_IDX = np.array([1 + 3 * k for k in range(21)] + [65 + 3 * k for k in range(21)]
                 + [128 + 3 * k for k in range(7)])
Y_IDX = X_IDX + 1
POSE_SWAPS = [(1, 2), (3, 4), (5, 6)]      # shoulders, elbows, wrists within the pose block


def resample(seq: np.ndarray, n: int = SEQ_LEN) -> np.ndarray:
    """Linear interpolation to exactly n frames (same as the browser)."""
    t = np.linspace(0, len(seq) - 1, n)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (t - lo)[:, None]
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(np.float32)


FACE_DIMS = slice(149, 156)   # the 7 face blendshapes


def model_input(seq: np.ndarray) -> np.ndarray:
    """What SignNet gets for one sign: resampled to SEQ_LEN frames, face values zeroed. INCLUDE signers stand
    too far away for face tracking (face data in 1 of 14,170 frames), so the model never learned faces, and a
    close-up live face only adds unfamiliar input. Faces are still used for question detection.
    Twin of the face zeroing in web/src/classifier.ts."""
    x = resample(seq)
    x[:, FACE_DIMS] = 0
    return x


def motion(seq: np.ndarray) -> np.ndarray:
    """Per-frame wrist movement, counting only hands present in both frames."""
    m = np.zeros(len(seq), np.float32)
    for pres, x in ((0, 1), (64, 65)):
        both = (seq[1:, pres] > 0) & (seq[:-1, pres] > 0)
        d = np.hypot(seq[1:, x] - seq[:-1, x], seq[1:, x + 1] - seq[:-1, x + 1])
        m[1:] += np.where(both, d, 0)
    return m


def prepare(seq):
    """Cut a clip exactly as the live segmenter would; fall back to trim_active if it finds no sign."""
    sign = cut_sign(seq)
    return sign if sign is not None else trim_active(seq)


def trim_active(seq, rest_y=1.8, stop_motion=0.015, min_len=8):
    """Keep the span where hands are up and moving (approximation of the live segmenter)."""
    if len(seq) == 0:
        return seq
    up = ((seq[:, 0] > 0) & (seq[:, 2] < rest_y)) | ((seq[:, 64] > 0) & (seq[:, 66] < rest_y))
    idx = np.where(up & (motion(seq) > stop_motion))[0]
    return seq if len(idx) < min_len else seq[idx[0]: idx[-1] + 1]


def mirror(seq):
    """Simulate a left-handed signer: negate x, swap hand slots, swap left/right body and brows."""
    s = seq.copy()
    s[:, X_IDX] *= -1
    s[:, 0:64], s[:, 64:128] = seq_copy_b(s), seq_copy_a(s)
    for l, r in POSE_SWAPS:
        L = slice(128 + 3 * l, 131 + 3 * l)
        R = slice(128 + 3 * r, 131 + 3 * r)
        s[:, L], s[:, R] = s[:, R].copy(), s[:, L].copy()
    s[:, [150, 151]] = s[:, [151, 150]]
    s[:, [152, 153]] = s[:, [153, 152]]
    return s


def seq_copy_a(s): return s[:, 0:64].copy()
def seq_copy_b(s): return s[:, 64:128].copy()


def augment(seq, rng):
    if rng.random() < 0.5:
        seq = mirror(seq)
    n = len(seq)
    if n > 12:                                            # random crop up to 10% at each end
        a = int(n * rng.uniform(0, 0.1))
        b = n - int(n * rng.uniform(0, 0.1))
        seq = seq[a:b]
    seq = seq.copy()
    th, k = np.deg2rad(rng.uniform(-10, 10)), rng.uniform(0.9, 1.1)
    x, y = seq[:, X_IDX].copy(), seq[:, Y_IDX].copy()
    seq[:, X_IDX] = k * (np.cos(th) * x - np.sin(th) * y)  # rotate + scale around shoulders
    seq[:, Y_IDX] = k * (np.sin(th) * x + np.cos(th) * y)
    seq[:, np.r_[X_IDX, Y_IDX]] += rng.normal(0, 0.01, (len(seq), 2 * len(X_IDX)))
    for pres in (0, 64):
        drop = rng.random(len(seq)) < 0.05                 # simulate missed hand detections
        seq[drop, pres] = 0
        seq[seq[:, pres] == 0, pres:pres + 64] = 0          # absent hands stay exactly zero
    return seq


def load_seq(path) -> np.ndarray:
    """Frames × 156 from an extracted .npy or a browser recording (.json from 'Record 3 s')."""
    if str(path).endswith(".json"):
        return np.asarray(json.loads(Path(path).read_text())["frames"], np.float32)
    return np.load(path)


class SignDataset(Dataset):
    """items are (path, label); NOTHING examples only come from your own recordings (data/own).
    (Synthetic 'no sign' examples made from real signs made the model reject real, flickery signs.)"""
    def __init__(self, items, label_to_idx, train: bool, seed: int = 0):
        self.items, self.l2i, self.train = items, label_to_idx, train
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        seq = self.cache.get(path)
        if seq is None:
            seq = prepare(load_seq(path))
            self.cache[path] = seq
        if self.train:
            seq = augment(seq, self.rng)
        if len(seq) < 2:
            seq = np.zeros((2, FEAT_DIM), np.float32)
        return torch.from_numpy(model_input(seq)), self.l2i[label]