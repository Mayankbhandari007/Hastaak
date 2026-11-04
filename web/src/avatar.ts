/** 2D signing avatar: replays the recorded motion of real signers (python/export_avatar.py → public/avatar/)
 *  as a simple figure, blending smoothly from one sign into the next. Coordinates are feature-spec units
 *  (shoulder widths from the shoulder midpoint, y down, camera view: the avatar faces the viewer). */
type Frame = [number[], number[], number[]];          // pose 7×(x,y); left hand 21×(x,y) or []; right hand or []
type Clip = { fps: number; frames: Frame[] };

const BLEND = 6;                                      // frames to move from one sign into the next
const HAND_LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12],
                    [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20]];
const PALM = [0, 1, 5, 9, 13, 17];
// Arms down: nose, L/R shoulder, L/R elbow, L/R wrist (the person's left is on the viewer's right)
const REST: Frame = [[0, -0.62, 0.5, 0, -0.5, 0, 0.62, 0.95, -0.62, 0.95, 0.55, 1.85, -0.55, 1.85], [], []];
// Namaste: elbows out, palms pressed together at mid-chest, fingers up. Hand-built (not recorded): in this
// front view the two palms meet edge-on, so each hand's points are stacked in a narrow upright strip.
const namasteHand = (side: 1 | -1): number[] =>
  [[0.07, 0.68], [0.1, 0.6], [0.1, 0.52], [0.08, 0.46], [0.05, 0.42],          // wrist, thumb
   [0.06, 0.49], [0.05, 0.38], [0.04, 0.31], [0.03, 0.25],                      // index
   [0.07, 0.48], [0.05, 0.35], [0.04, 0.27], [0.03, 0.2],                       // middle
   [0.08, 0.49], [0.06, 0.37], [0.05, 0.3], [0.04, 0.24],                       // ring
   [0.09, 0.51], [0.07, 0.42], [0.06, 0.36], [0.05, 0.31]]                      // little
    .flatMap(([x, y]) => [x * side, y]);
const NAMASTE: Frame = [[0, -0.62, 0.5, 0, -0.5, 0, 0.8, 0.8, -0.8, 0.8, 0.07, 0.68, -0.07, 0.68],
                        namasteHand(1), namasteHand(-1)];
const C = { bg: "#e9eef5", shirt: "#3c6fb0", skin: "#e8b48f", skinLine: "#b77e5c", hair: "#2b2321", face: "#3a2b25" };

type Step = { frame: Frame; sign: number; smile?: boolean };

export class Avatar {
  private clips: Record<string, Clip> = {};
  private ctx: CanvasRenderingContext2D;
  private run = 0;              // id of the animation in progress; a newer one stops the older one
  private signing = false;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    this.draw(REST);
  }

  async has(sign: string): Promise<boolean> {
    if (!this.clips[sign]) {
      const r = await fetch(`/avatar/${sign}.json`);
      if (!r.ok) return false;
      this.clips[sign] = await r.json();
    }
    return true;
  }

  /** Sign each word in order; onSign(i) is called as sign i starts (−1 at the end). */
  async play(signs: string[], onSign: (i: number) => void = () => {}): Promise<void> {
    const timeline: Step[] = [];
    let prev = REST;
    for (const [i, s] of signs.entries()) {
      if (!(await this.has(s))) continue;
      const frames = this.clips[s].frames;
      for (let k = 1; k <= BLEND; k++) timeline.push({ frame: mix(prev, frames[0], k / BLEND), sign: i });
      frames.forEach((f) => timeline.push({ frame: f, sign: i }));
      prev = frames[frames.length - 1];
    }
    for (let k = 1; k <= BLEND * 2; k++) timeline.push({ frame: mix(prev, REST, k / (BLEND * 2)), sign: -1 });
    this.signing = true;
    try { await this.animate(timeline, onSign); } finally { this.signing = false; }
  }

  /** Smile and hold namaste for holdMs, then lower the hands. onGreet(true) as the hands meet, (false) as they
   *  drop. Skipped while signing; a sign that starts during the greeting cuts it short (onGreet(false) first). */
  async greet(onGreet: (on: boolean) => void, holdMs = 4000): Promise<void> {
    if (this.signing) return;
    const IN = 10, OUT = 12;
    const timeline: Step[] = [];
    for (let k = 1; k <= IN; k++) timeline.push({ frame: mix(REST, NAMASTE, ease(k / IN)), sign: -1, smile: true });
    for (let k = 0; k < Math.round((holdMs / 1000) * 25); k++) timeline.push({ frame: NAMASTE, sign: 0, smile: true });
    for (let k = 1; k <= OUT; k++) timeline.push({ frame: mix(NAMASTE, REST, ease(k / OUT)), sign: -1, smile: k < OUT / 2 });
    const done = await this.animate(timeline, (i) => onGreet(i === 0));
    if (!done) onGreet(false);
  }

  /** Draw a timeline at 25 fps. Resolves true when it ran to the end, false when a newer animation took over. */
  private animate(timeline: Step[], onSign: (i: number) => void): Promise<boolean> {
    const id = ++this.run;
    const start = performance.now();
    let shown = -2;
    return new Promise<boolean>((done) => {
      const tick = () => {
        if (id !== this.run) return done(false);
        const n = Math.floor(((performance.now() - start) / 1000) * 25);
        if (n >= timeline.length) { onSign(-1); this.draw(REST); return done(true); }
        if (timeline[n].sign !== shown) { shown = timeline[n].sign; onSign(shown); }
        this.draw(timeline[n].frame, timeline[n].smile);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
  }

  private draw([pose, left, right]: Frame, smile = false) {
    const { ctx, canvas } = this;
    const s = canvas.height * 0.2117;                              // one shoulder width in pixels (101.6 at 480 high)
    const ox = canvas.width / 2, oy = canvas.height * 0.4;
    const P = (x: number, y: number): [number, number] => [ox + x * s, oy + y * s];
    const pt = (arr: number[], i: number) => P(arr[2 * i], arr[2 * i + 1]);
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const [nose, ls, rs] = [pt(pose, 0), pt(pose, 1), pt(pose, 2)];

    // torso and neck
    ctx.fillStyle = C.shirt;
    ctx.beginPath();
    ctx.moveTo(rs[0] - 0.08 * s, rs[1] - 0.05 * s);
    ctx.lineTo(ls[0] + 0.08 * s, ls[1] - 0.05 * s);
    ctx.lineTo(ls[0] - 0.05 * s, ls[1] + 2.6 * s);
    ctx.lineTo(rs[0] + 0.05 * s, rs[1] + 2.6 * s);
    ctx.closePath();
    ctx.fill();
    const neck: [number, number] = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2];
    this.line([neck, [nose[0], nose[1] + 0.2 * s]], 0.22 * s, C.skin);

    // head: face, hair, eyes, mouth around the nose
    const hx = nose[0], hy = nose[1] - 0.08 * s;
    ctx.fillStyle = C.skin;
    ctx.beginPath();
    ctx.ellipse(hx, hy, 0.34 * s, 0.42 * s, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = C.hair;
    ctx.beginPath();
    ctx.ellipse(hx, hy - 0.16 * s, 0.36 * s, 0.3 * s, 0, Math.PI, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = C.face;
    for (const dx of [-0.12, 0.12]) {
      ctx.beginPath();
      ctx.arc(hx + dx * s, hy - 0.02 * s, 0.035 * s, 0, Math.PI * 2);
      ctx.fill();
    }
    if (smile) {
      ctx.strokeStyle = C.face;
      ctx.lineWidth = 0.03 * s;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(hx - 0.11 * s, hy + 0.17 * s);
      ctx.quadraticCurveTo(hx, hy + 0.3 * s, hx + 0.11 * s, hy + 0.17 * s);
      ctx.stroke();
    } else {
      this.line([[hx - 0.09 * s, hy + 0.2 * s], [hx + 0.09 * s, hy + 0.2 * s]], 0.025 * s, C.face);
    }

    // arms (shoulder → elbow → wrist; the hand's own wrist point when the hand is tracked) and hands
    for (const [sh, el, wr, hand] of [[1, 3, 5, left], [2, 4, 6, right]] as const) {
      const wrist = hand.length ? P(hand[0], hand[1]) : pt(pose, wr);
      this.line([pt(pose, sh), pt(pose, el)], 0.24 * s, C.shirt);
      this.line([pt(pose, el), wrist], 0.17 * s, C.skin);
      if (hand.length) this.hand(hand, P, s);
      else { ctx.fillStyle = C.skin; ctx.beginPath(); ctx.arc(wrist[0], wrist[1], 0.11 * s, 0, Math.PI * 2); ctx.fill(); }
    }
  }

  private hand(h: number[], P: (x: number, y: number) => [number, number], s: number) {
    const { ctx } = this;
    const pts = Array.from({ length: 21 }, (_, i) => P(h[2 * i], h[2 * i + 1]));
    ctx.fillStyle = C.skin;
    ctx.strokeStyle = C.skinLine;
    ctx.lineWidth = 0.02 * s;
    ctx.beginPath();
    PALM.forEach((i, k) => (k ? ctx.lineTo(...pts[i]) : ctx.moveTo(...pts[i])));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    for (const [a, b] of HAND_LINKS) this.line([pts[a], pts[b]], 0.075 * s, C.skinLine);
    for (const [a, b] of HAND_LINKS) this.line([pts[a], pts[b]], 0.05 * s, C.skin);
  }

  private line(points: [number, number][], width: number, color: string) {
    const { ctx } = this;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.beginPath();
    points.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
    ctx.stroke();
  }
}

const ease = (t: number) => t * t * (3 - 2 * t);      // smoothstep: gentle start and stop

/** Linear blend between two frames; a hand present in only one of them appears/disappears at the midpoint. */
function mix(a: Frame, b: Frame, t: number): Frame {
  const lerp = (x: number[], y: number[]) => x.map((v, i) => v + (y[i] - v) * t);
  const hand = (x: number[], y: number[]) => (x.length && y.length ? lerp(x, y) : t < 0.5 ? x : y);
  return [lerp(a[0], b[0]), hand(a[1], b[1]), hand(a[2], b[2])];
}
