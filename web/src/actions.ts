import type { NormalizedLandmark as Lm } from "@mediapipe/tasks-vision";

const dist = (a: Lm, b: Lm) => Math.hypot(a.x - b.x, a.y - b.y);

/** Thumb points up (image y grows downward), thumb tip is the hand's highest point,
 *  and the four fingers are curled (tip closer to the wrist than the middle joint). */
export function isThumbsUp(h: Lm[]): boolean {
  const thumbUp = h[4].y < h[3].y && h[3].y < h[2].y &&
                  h[4].y < Math.min(...h.slice(5).map((p) => p.y));
  const curled = [[8, 6], [12, 10], [16, 14], [20, 18]]
    .every(([tip, pip]) => dist(h[tip], h[0]) < dist(h[pip], h[0]));
  return thumbUp && curled;
}

/** [thumb, index, middle, ring, pinky] extended? */
export function fingerStates(h: Lm[]): boolean[] {
  return [
    dist(h[4], h[9]) > dist(h[3], h[9]),
    dist(h[8], h[0]) > dist(h[6], h[0]),
    dist(h[12], h[0]) > dist(h[10], h[0]),
    dist(h[16], h[0]) > dist(h[14], h[0]),
    dist(h[20], h[0]) > dist(h[18], h[0]),
  ];
}

/** Average of the three brow-raise blendshapes. */
export function browScore(categories: { categoryName: string; score: number }[]): number {
  const get = (n: string) => categories.find((c) => c.categoryName === n)?.score ?? 0;
  return (get("browInnerUp") + get("browOuterUpLeft") + get("browOuterUpRight")) / 3;
}

/** Learns a person's neutral brow level from the first N frames (median). */
export class BrowCalibrator {
  private samples: number[] = [];
  baseline: number | null = null;
  constructor(private needed = 30) {}

  reset() { this.samples = []; this.baseline = null; }
  get progress() { return Math.min(1, this.samples.length / this.needed); }

  push(score: number) {
    if (this.baseline !== null) return;
    this.samples.push(score);
    if (this.samples.length >= this.needed) {
      const s = [...this.samples].sort((a, b) => a - b);
      this.baseline = s[Math.floor(s.length / 2)];
    }
  }

  isRaised(score: number, threshold: number) {
    return this.baseline !== null && score - this.baseline > threshold;
  }
}