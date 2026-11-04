import type { NormalizedLandmark as Lm, FaceLandmarkerResult } from "@mediapipe/tasks-vision";

/** Feature spec v1: 156 numbers per frame. Must match python/features.py exactly. */
export const FEATURE_SPEC_VERSION = 1;
export const FEAT_DIM = 156;
export const POSE_IDX = [0, 11, 12, 13, 14, 15, 16] as const;
export const BROW_KEYS = ["browInnerUp", "browOuterUpLeft", "browOuterUpRight",
                          "browDownLeft", "browDownRight", "jawOpen", "mouthPucker"] as const;

export function shoulderFrame(pose: Lm[], a: number) {
  const lx = pose[11].x * a, ly = pose[11].y, rx = pose[12].x * a, ry = pose[12].y;
  return { cx: (lx + rx) / 2, cy: (ly + ry) / 2, s: Math.hypot(lx - rx, ly - ry) || 1 };
}

/** Slot A = hand nearest pose landmark 15 (person's left wrist), slot B = nearest 16. */
export function assignHands(hands: Lm[][], pose: Lm[], a: number): [Lm[] | null, Lm[] | null] {
  const { s } = shoulderFrame(pose, a);
  const targets = [pose[15], pose[16]];
  const pairs: { h: number; slot: number; d: number }[] = [];
  hands.forEach((h, hi) => targets.forEach((t, slot) =>
    pairs.push({ h: hi, slot, d: Math.hypot((h[0].x - t.x) * a, h[0].y - t.y) })));
  pairs.sort((p, q) => p.d - q.d);

  const out: [Lm[] | null, Lm[] | null] = [null, null];
  const used = new Set<number>();
  for (const p of pairs) {
    if (p.d > 0.8 * s || out[p.slot] || used.has(p.h)) continue;
    out[p.slot] = hands[p.h];
    used.add(p.h);
  }
  return out;
}

/** Blendshapes of the face whose nose tip is nearest the signer's pose nose. */
export function pickFaceBlend(face: FaceLandmarkerResult, pose: Lm[], a: number): Record<string, number> {
  if (!face.faceLandmarks.length) return {};
  let best = 0, bestD = Infinity;
  face.faceLandmarks.forEach((f, i) => {
    const d = Math.hypot((f[1].x - pose[0].x) * a, f[1].y - pose[0].y);
    if (d < bestD) { bestD = d; best = i; }
  });
  const out: Record<string, number> = {};
  face.faceBlendshapes?.[best]?.categories.forEach((c) => { out[c.categoryName] = c.score; });
  return out;
}

export function featurize(pose: Lm[] | null, handA: Lm[] | null, handB: Lm[] | null,
                          blend: Record<string, number>, a: number): Float32Array {
  const out = new Float32Array(FEAT_DIM);
  if (!pose) return out;
  const { cx, cy, s } = shoulderFrame(pose, a);
  let i = 0;
  const put = (p: Lm) => {
    out[i++] = (p.x * a - cx) / s;
    out[i++] = (p.y - cy) / s;
    out[i++] = (p.z * a) / s;
  };
  for (const h of [handA, handB]) {
    if (h) { out[i++] = 1; h.forEach(put); } else { i += 64; }
  }
  POSE_IDX.forEach((k) => put(pose[k]));
  BROW_KEYS.forEach((k) => { out[i++] = blend[k] ?? 0; });
  return out;
}