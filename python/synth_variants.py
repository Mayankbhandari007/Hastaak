"""Synthetic variants of INCLUDE clips that look like a laptop webcam: lower resolution, compression, sensor
noise, dim / uneven / coloured light, blur, closer framing and slight tilt. Whole-frame edits only: no masks,
no API calls, no credits.

Why pixels and not features: train.py (dataset.augment) already mirrors, rotates, scales, trims and drops hand
frames on the 156 numbers, and the features are relative to the shoulders, so a plain zoom or shift of the
picture cancels out. What only pixels can teach is how MediaPipe itself behaves on a worse picture (jitter, a
hand lost in dim light or blur, a hand cut by the frame edge). The variants are re-tracked by
extract_include.py and cut by segmenter.cut_sign like every other clip (rule 3).

Leak-proof: variants inherit their source clip's split. Train clips -> extra training data; test clips ->
a separate robustness test (data/synth_test_manifest.csv), never trained on. Validation clips get none, so
model selection stays on clean clips.

Usage (from python/):
  1. python synth_variants.py --filter Greetings Pronouns --per-clip 4          # writes videos (resumable)
  2. python extract_include.py --manifest data/synth_manifest.csv --videos E:/12 --feats data/include_data_feats
  3. python synth_variants.py --finish                                         # QC + training manifests
  4. python train.py --manifest data/final_plus_synth_manifest.csv --feats data/include_data_feats --no-own --name <run>
"""
import argparse
import csv
import os
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

FIELDS = ["video", "label", "category", "split", "source", "recipe"]


# ---------------------------------------------------------------- recipe (fixed per variant, seeded)
def make_recipe(rng: np.random.Generator) -> dict:
    r = {}
    g = rng.choice(["none", "close", "tilt"], p=[0.25, 0.35, 0.4])
    if g == "close":      # seated at a laptop: shoulders fill a third or more of the frame and sit in its lower
        # half, head near the top, hands can leave the frame. INCLUDE signers stand far back (shoulders ~15%).
        # The head is ~1.3 shoulder widths tall above the shoulders, so these ranges keep it (mostly) in frame.
        r["close"] = dict(width=rng.uniform(0.26, 0.36), x=0.5 + rng.uniform(-0.08, 0.08),
                          y=rng.uniform(0.62, 0.78), rot=rng.uniform(-4, 4))
    elif g == "tilt":     # camera a little off: slight rotation and shift
        r["geo"] = dict(zoom=rng.uniform(0.92, 1.08), cx=rng.uniform(-0.06, 0.06), cy=rng.uniform(-0.06, 0.06),
                        rot=rng.uniform(-7, 7))
    light = rng.choice(["dim", "bright", "side", "normal"], p=[0.4, 0.15, 0.25, 0.2])
    r["light"] = str(light)
    r["gamma"] = {"dim": rng.uniform(1.3, 2.1), "bright": rng.uniform(0.6, 0.85)}.get(light, rng.uniform(0.9, 1.15))
    r["gain"] = {"dim": rng.uniform(0.55, 0.85), "bright": rng.uniform(1.05, 1.25)}.get(light, rng.uniform(0.9, 1.1))
    r["contrast"] = rng.uniform(0.65, 1.15)
    if light == "side":   # one side lit, the other in shadow
        r["side"] = dict(angle=rng.uniform(0, 2 * np.pi), low=rng.uniform(0.45, 0.75))
    if rng.random() < 0.6:  # warm (tungsten) or cool (daylight / screen) cast
        warm = rng.random() < 0.6
        r["wb"] = [rng.uniform(0.75, 0.92), 1.0, rng.uniform(1.06, 1.25)] if warm else \
                  [rng.uniform(1.06, 1.2), 1.0, rng.uniform(0.8, 0.94)]          # B, G, R gains
    r["sat"] = rng.uniform(0.55, 1.2)
    b = rng.choice(["none", "defocus", "motion"], p=[0.45, 0.3, 0.25])
    if b == "defocus":
        r["blur"] = dict(kind="defocus", sigma=rng.uniform(0.8, 1.8))
    elif b == "motion":
        r["blur"] = dict(kind="motion", length=int(rng.integers(5, 12)), angle=rng.uniform(0, 180))
    r["size"] = [640, 360] if rng.random() < 0.6 else [960, 540]     # laptop webcam resolutions (16:9)
    r["noise"] = rng.uniform(2, 6) * (1.6 if light == "dim" else 1.0)  # dim rooms are noisier
    r["jpeg"] = int(rng.integers(35, 75))
    return r


def describe(r: dict) -> str:
    parts = [f"{r['size'][1]}p", r["light"]]
    if "close" in r:
        parts.append("close")
    if "geo" in r:
        parts.append("tilt")
    if "wb" in r:
        parts.append("warm" if r["wb"][2] > 1 else "cool")
    if "blur" in r:
        parts.append(r["blur"]["kind"])
    return "+".join(parts)


# ---------------------------------------------------------------- applying a recipe
_POSE = None
_TS = 0


def shoulders(path: str):
    """(midpoint x, y, width) of the signer's shoulders as fractions of the frame, from one frame a third of
    the way in (the signer is in place by then). None if nobody is found."""
    global _POSE, _TS
    from common import create_landmarkers, to_mp_image
    from features import shoulder_frame
    if _POSE is None:                                  # one pose model per worker process
        hands, face, _POSE = create_landmarkers(num_hands=1, num_poses=3)
        hands.close()
        face.close()
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 3)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return None
    _TS += 1000
    res = _POSE.detect_for_video(to_mp_image(bgr), _TS)
    if not res.pose_landmarks:
        return None
    h, w = bgr.shape[:2]
    p = max(res.pose_landmarks, key=lambda q: shoulder_frame(q, w / h)[2])   # same signer choice as extraction
    l, r = p[11], p[12]
    return (l.x + r.x) / 2, (l.y + r.y) / 2, abs(l.x - r.x)


def build_ops(r: dict, w: int, h: int, src_wh=None, sh=None):
    """Precompute per-clip arrays (warp matrix, light map, blur kernel) at the output size."""
    ops = {}
    if "close" in r:
        W, H = src_wh
        c = r["close"]
        mx, my, sw = sh if sh else (0.5, 0.42, 0.15)   # no person found: INCLUDE's typical framing
        z = min(c["width"] / max(sw, 0.05), 3.5)       # zoom in output-frame units
        m = cv2.getRotationMatrix2D((mx * W, my * H), c["rot"], z * w / W)
        m[:, 2] += (c["x"] * w - mx * W, c["y"] * h - my * H)
        ops["warp_src"] = m                            # straight from the full-size frame, so the crop stays sharp
    if "geo" in r:
        g = r["geo"]
        c = (w / 2 + g["cx"] * w, h / 2 + g["cy"] * h)
        m = cv2.getRotationMatrix2D(c, g["rot"], g["zoom"])
        m[:, 2] += (w / 2 - c[0], h / 2 - c[1])       # keep the zoom centre in the middle of the frame
        ops["warp"] = m
    # White balance, contrast, gamma and overall gain are per-channel point operations: one lookup table each.
    v = np.arange(256, dtype=np.float32) / 255.0
    luts = []
    for gain_c in r.get("wb", [1.0, 1.0, 1.0]):                    # B, G, R
        y = (np.clip(v * gain_c, 0, 1) - 0.5) * r["contrast"] + 0.5
        y = np.clip(y, 0, 1) ** r["gamma"] * r["gain"]
        luts.append(np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8))
    ops["lut"] = np.stack(luts, axis=1)[None]                     # 1 × 256 × 3 for cv2.LUT
    if "side" in r:                                                # one side lit, the other in shadow
        s = r["side"]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = ((xx / w - 0.5) * np.cos(s["angle"]) + (yy / h - 0.5) * np.sin(s["angle"])) + 0.5
        lm = s["low"] + (1 - s["low"]) * np.clip(t, 0, 1)
        ops["light"] = cv2.merge([(lm * 255).astype(np.uint8)] * 3)
    ops["noise"] = np.empty((h, w, 3), np.int16)                   # reused buffer for cv2.randn
    if r.get("blur", {}).get("kind") == "motion":
        n = r["blur"]["length"]
        k = np.zeros((n, n), np.float32)
        k[n // 2, :] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), r["blur"]["angle"], 1), (n, n))
        ops["kernel"] = k / max(k.sum(), 1e-6)
    return ops


def apply(bgr: np.ndarray, r: dict, ops: dict, rng: np.random.Generator) -> np.ndarray:
    w, h = r["size"]
    if "warp_src" in ops:
        img = cv2.warpAffine(bgr, ops["warp_src"], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    else:
        img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    if "warp" in ops:
        img = cv2.warpAffine(img, ops["warp"], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if "blur" in r:
        img = cv2.GaussianBlur(img, (0, 0), r["blur"]["sigma"]) if r["blur"]["kind"] == "defocus" \
            else cv2.filter2D(img, -1, ops["kernel"])
    if r["sat"] != 1.0:
        grey = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, r["sat"], grey, 1 - r["sat"], 0)
    img = cv2.LUT(img, ops["lut"])
    if "light" in ops:
        img = cv2.multiply(img, ops["light"], scale=1 / 255)
    n = ops["noise"]
    cv2.randn(n, 0, r["noise"])                                    # sensor noise, new every frame
    img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, r["jpeg"]])   # webcam-style compression
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def make_one(job):
    """All variants of one source clip in a single pass, so the 1080p source is decoded once."""
    src, variants = job
    todo = [(Path(d), seed, rec) for d, seed, rec in variants if not Path(d).exists()]
    if not todo:
        return src, "skip"
    cv2.setRNGSeed(todo[0][1] % 2**31)                   # cv2.randn noise is reproducible per source clip
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    src_wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sh = shoulders(src) if any("close" in rec for _, _, rec in todo) else None
    outs = []
    for dst, seed, rec in todo:
        w, h = rec["size"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".part.mp4")
        outs.append((dst, tmp, rec, build_ops(rec, w, h, src_wh, sh), np.random.default_rng(seed),
                     cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))))
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for _, _, rec, ops, rng, wr in outs:
                wr.write(apply(frame, rec, ops, rng))
            n += 1
    finally:
        cap.release()
        for o in outs:
            o[5].release()
    for dst, tmp, *_ in outs:
        if n:
            os.replace(tmp, dst)
        else:
            tmp.unlink(missing_ok=True)
    return src, "ok" if n else "empty"


# ---------------------------------------------------------------- step 1: write variants
def make(args):
    with open(args.manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if any(r["video"].startswith(p) for p in args.filter) and r["split"] in ("train", "test")]
    out_root = Path(args.out)
    rel_root = out_root.parent
    train_rows, test_rows, jobs = [], [], []
    for r in rows:
        variants = []
        jobs.append((str(Path(args.videos) / r["video"]), variants))
        for k in range(1, args.per_clip + 1):
            seed = zlib.crc32(f"{r['video']}#{k}".encode())
            recipe = make_recipe(np.random.default_rng(seed))
            dst = out_root / Path(r["video"]).with_suffix("").as_posix()
            dst = dst.with_name(f"{dst.name}__v{k}.mp4")
            row = {"video": dst.relative_to(rel_root).as_posix(), "label": r["label"], "category": r["category"],
                   "split": r["split"], "source": r["video"], "recipe": describe(recipe)}
            (train_rows if r["split"] == "train" else test_rows).append(row)
            variants.append((str(dst), seed + 1, recipe))

    for path, rws in ((args.synth_manifest, train_rows + test_rows),):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rws)
    print(f"{len(rows)} source clips -> {len(train_rows) + len(test_rows)} variants ({len(train_rows)} train, "
          f"{len(test_rows)} test) -> {out_root}\nManifest: {args.synth_manifest}")
    if args.limit:
        jobs = jobs[:args.limit]

    from tqdm import tqdm
    bad = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(make_one, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Source clips"):
            try:
                path, status = fut.result()
                if status == "empty":
                    bad.append(path)
            except Exception as e:
                bad.append(str(e))
    print(f"Done. {len(bad)} problems." + (f" e.g. {bad[0]}" if bad else ""))


# ---------------------------------------------------------------- step 3: QC and manifests
def finish(args):
    from extract_include import feat_path
    from segmenter import MIN_HAND_PRESENCE, cut_sign, hand_presence
    with open(args.synth_manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    kept, dropped, missing = [], [], 0
    for r in rows:
        p = feat_path(args.feats, r["video"])
        if not p.exists():
            missing += 1
            continue
        seq = np.load(p)
        seg = cut_sign(seq) if len(seq) else seq
        # Same test the live app applies before trusting a sign: a variant whose hands were mostly lost
        # would be rejected live too, so it teaches the model nothing but noise.
        if seg is not None and len(seg) and hand_presence(seg) >= MIN_HAND_PRESENCE:
            kept.append(r)
        else:
            dropped.append(r)
    with open(args.manifest, newline="") as f:
        base = list(csv.DictReader(f))
    with open(args.combined, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "label", "category", "split"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in base)                       # every original clip
        w.writerows({k: r[k] for k in w.fieldnames} for r in kept if r["split"] == "train")
    with open(args.synth_test, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "label", "category", "split"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in kept if r["split"] == "test")
    by = {}
    for r in dropped:
        for part in r["recipe"].split("+"):
            by[part] = by.get(part, 0) + 1
    print(f"{len(kept)} variants kept, {len(dropped)} dropped (hands mostly lost), {missing} not extracted yet")
    if by:
        print("  dropped by condition: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])))
    print(f"Training manifest: {args.combined} (originals + {sum(r['split'] == 'train' for r in kept)} variants)")
    print(f"Robustness test:   {args.synth_test} ({sum(r['split'] == 'test' for r in kept)} variants of test clips)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/final_manifest.csv")
    ap.add_argument("--videos", default="E:/12/Final_Dataset", help="where the manifest's videos are")
    ap.add_argument("--out", default="E:/12/Synthetic_GP", help="variant videos (paths in the manifest are "
                                                              "relative to this folder's parent)")
    ap.add_argument("--filter", nargs="*", default=["Greetings", "Pronouns"])
    ap.add_argument("--per-clip", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N source clips (for a quick look)")
    ap.add_argument("--synth-manifest", default="data/synth_manifest.csv")
    ap.add_argument("--finish", action="store_true", help="after extraction: QC and write training manifests")
    ap.add_argument("--feats", default="data/include_data_feats")
    ap.add_argument("--combined", default="data/final_plus_synth_manifest.csv")
    ap.add_argument("--synth-test", default="data/synth_test_manifest.csv")
    args = ap.parse_args()
    finish(args) if args.finish else make(args)


if __name__ == "__main__":
    main()
