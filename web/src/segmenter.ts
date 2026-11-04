import { FEAT_DIM } from "./features";

export const SEQ_LEN = 32;
export const NOTHING = "NOTHING";   // the model's "no sign" class

// Real INCLUDE signs have a tracked hand in ≥81% of their frames. Below this the hand was lost
// (out of frame, poor light) and the segment is not classified. Same value as python/segmenter.py.
export const MIN_HAND_PRESENCE = 0.6;

/** Fraction of frames in which at least one hand slot is filled. */
export function handPresence(seq: Float32Array[]): number {
  return seq.length ? seq.filter((f) => f[0] > 0 || f[64] > 0).length / seq.length : 0;
}

export type SegmenterConfig = {
  restY: number;        // hands below this (shoulder widths under the shoulders) are at rest
  startMotion: number;  // wrist movement per frame needed to start a sign
  stopMotion: number;   // below this counts as still
  stopFrames: number;   // still frames needed to end a sign
  minLen: number;       // shorter segments are discarded
  maxLen: number;       // longer segments are cut
  holdFrames: number;   // a hand lost for up to this many frames keeps its last position
};

export const DEFAULT_SEGMENTER: SegmenterConfig = {
  restY: 1.8, startMotion: 0.05, stopMotion: 0.015, stopFrames: 8, minLen: 8, maxLen: 90, holdFrames: 4,
};
export const TRAINING_FPS = 25;   // INCLUDE videos; DEFAULT_SEGMENTER is tuned for this rate

/** On a slower camera loop, wait the same time (not the same number of frames) before ending a sign, and
 *  accept the same minimum sign duration. On INCLUDE test clips at 8 fps this took accuracy from 30/33 to
 *  33/33. Motion thresholds stay unscaled (scaling them lost signs). Twin of python/segmenter.low_fps_settings. */
export function segmenterForFps(fps: number): SegmenterConfig {
  const k = Math.min(4, Math.max(1, TRAINING_FPS / Math.max(fps, 1)));   // frames get k× longer
  const d = DEFAULT_SEGMENTER;
  const frames = (n: number, min: number) => Math.max(min, Math.round(n / k));
  // at least 5 still frames: 3 split a real user's HELLO at its pause by the forehead (no cost on INCLUDE)
  return { ...d, stopFrames: frames(d.stopFrames, 5), holdFrames: frames(d.holdFrames, 1), minLen: frames(d.minLen, 3) };
}

/** Cuts the feature stream into single signs. Same logic as python/segmenter.Segmenter. */
export class Segmenter {
  state: "REST" | "ACTIVE" = "REST";
  private buf: Float32Array[] = [];
  private prev: Float32Array | null = null;
  private still = 0;
  private lastHand: (Float32Array | null)[] = [null, null];
  private missing = [0, 0];

  constructor(public cfg: SegmenterConfig = { ...DEFAULT_SEGMENTER }) {}

  reset() {
    this.state = "REST";
    this.buf = [];
    this.prev = null;
    this.still = 0;
    this.lastHand = [null, null];
    this.missing = [0, 0];
  }

  /** Bridge brief tracking dropouts: reuse each hand slot's last values for up to holdFrames frames, but only
   *  for a hand last seen raised (a lowered hand leaving the frame means the sign is over).
   *  bridgedOnly means no hand was really tracked in this frame but one was filled in. */
  private fill(f: Float32Array): { f: Float32Array; bridgedOnly: boolean } {
    const tracked = f[0] > 0 || f[64] > 0;
    let filled = false;
    for (const [k, off] of [[0, 0], [1, 64]] as const) {
      const last = this.lastHand[k];
      if (f[off] > 0) {
        this.lastHand[k] = f.slice(off, off + 64);
        this.missing[k] = 0;
      } else if (last && this.missing[k] < this.cfg.holdFrames && last[2] < this.cfg.restY) {   // [2] = wrist y
        this.missing[k]++;
        if (!filled) { f = f.slice(); filled = true; }
        f.set(this.lastHand[k]!, off);
      } else {
        this.lastHand[k] = null;
      }
    }
    return { f, bridgedOnly: filled && !tracked };
  }

  /** Sum of wrist movement for hand slots present in both frames (shoulder widths per frame). */
  private motion(f: Float32Array): number {
    let d = 0;
    if (this.prev) {
      for (const [pres, x] of [[0, 1], [64, 65]] as const) {
        if (f[pres] > 0 && this.prev[pres] > 0) {
          d += Math.hypot(f[x] - this.prev[x], f[x + 1] - this.prev[x + 1]);
        }
      }
    }
    this.prev = f;
    return d;
  }

  /** Feed one frame's features. Returns the frames of a finished sign, or null. */
  push(frame: Float32Array): Float32Array[] | null {
    const c = this.cfg;
    const { f, bridgedOnly } = this.fill(frame);
    const m = this.motion(f);
    const up = (f[0] > 0 && f[2] < c.restY) || (f[64] > 0 && f[66] < c.restY);

    if (this.state === "REST") {
      if (up && m > c.startMotion) {
        this.state = "ACTIVE";
        this.buf = [f];
        this.still = 0;
      }
      return null;
    }

    this.buf.push(f);
    // a bridged frame has no real motion: don't let it end the sign
    if (!bridgedOnly) this.still = m < c.stopMotion || !up ? this.still + 1 : 0;
    if (this.still >= c.stopFrames || this.buf.length >= c.maxLen) {
      const seg = this.buf.slice(0, this.buf.length - this.still);
      this.state = "REST";
      this.buf = [];
      return seg.length >= c.minLen ? seg : null;
    }
    return null;
  }
}

/** Linear interpolation to exactly n frames, flattened to n × FEAT_DIM (matches dataset.resample). */
export function resample(seq: Float32Array[], n = SEQ_LEN): Float32Array {
  const out = new Float32Array(n * FEAT_DIM);
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0 : (i * (seq.length - 1)) / (n - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, seq.length - 1);
    const w = t - lo;
    for (let k = 0; k < FEAT_DIM; k++) {
      out[i * FEAT_DIM + k] = seq[lo][k] * (1 - w) + seq[hi][k] * w;
    }
  }
  return out;
}
