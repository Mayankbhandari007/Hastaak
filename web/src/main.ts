import "./style.css";
import { DrawingUtils, HandLandmarker, FaceLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import { createLandmarkers, type Landmarkers } from "./landmarkers";
import { isThumbsUp, fingerStates, browScore, BrowCalibrator } from "./actions";
import { assignHands, pickFaceBlend, featurize, FEATURE_SPEC_VERSION } from "./features";
import { Segmenter, resample, handPresence, MIN_HAND_PRESENCE, NOTHING } from "./segmenter";
import { loadClassifier, type SignClassifier, type Prediction } from "./classifier";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>("video");
const canvas = $<HTMLCanvasElement>("canvas");
const ctx = canvas.getContext("2d")!;
const draw = new DrawingUtils(ctx);

const showPose = $<HTMLInputElement>("showPose");
const showHands = $<HTMLInputElement>("showHands");
const showFace = $<HTMLInputElement>("showFace");
const thr = $<HTMLInputElement>("thr");

let lm: Landmarkers | null = null;
let lastVideoTime = -1;
let lastTs = 0;
let thumbFrames = 0;
const brow = new BrowCalibrator(30);

const RECORD_FRAMES = 90;
let recording: number[][] | null = null;

const MIN_CONFIDENCE = 0.35;   // weaker predictions are shown but not added to the history
const HISTORY_MAX = 12;
let clf: SignClassifier | null = null;
const segmenter = new Segmenter();
let lastSegState = "";
const history: string[] = [];

const times = { pose: 0, hands: 0, face: 0 };
let frameCount = 0;
let fpsStart = performance.now();

const setStatus = (t: string) => { $("status").textContent = t; };
const smooth = (prev: number, v: number) => (prev === 0 ? v : prev * 0.9 + v * 0.1);
const fmtMs = (v: number) => `${v.toFixed(1)} ms`;

async function start() {
  const btn = $<HTMLButtonElement>("start");
  btn.disabled = true;
  try {
    setStatus("Loading MediaPipe models and the sign classifier from local files…");
    const [landmarkers, classifier] = await Promise.all([
      createLandmarkers(),
      loadClassifier().catch((err: Error) => {
        console.warn(err);
        $("segState").textContent = "Sign model unavailable";
        return null;
      }),
    ]);
    lm = landmarkers;
    clf = classifier;
    if (clf) showVocabulary(clf.labels);
    setStatus("Starting camera…");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, frameRate: 30 }, audio: false,
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    setStatus(`Running at ${video.videoWidth}×${video.videoHeight}. ` +
              (clf ? `Model knows ${clf.labels.length} signs. ` : "Sign recognition is off (model not loaded). ") +
              "Keep a neutral face for a second.");
    requestAnimationFrame(tick);
  } catch (err) {
    console.error(err);
    btn.disabled = false;
    setStatus(`Couldn't start: ${(err as Error)?.message ?? err}. ` +
              "Check camera permission, and run `npm run models` if model files are missing.");
  }
}

function tick() {
  if (lm && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    processFrame(lm);
  }
  requestAnimationFrame(tick);
}

function timed<T>(key: keyof typeof times, fn: () => T): T {
  const t0 = performance.now();
  const r = fn();
  times[key] = smooth(times[key], performance.now() - t0);
  return r;
}

function processFrame(m: Landmarkers) {
  const ts = Math.max(lastTs + 1, Math.round(performance.now()));   // must strictly increase
  lastTs = ts;

  const pose = timed("pose", () => m.pose.detectForVideo(video, ts));
  const hands = timed("hands", () => m.hands.detectForVideo(video, ts));
  const face = timed("face", () => m.face.detectForVideo(video, ts));

  // ---------- Drawing ----------
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const signer = pose.landmarks[0] ?? null;

  if (showPose.checked && signer) {
    draw.drawConnectors(signer, PoseLandmarker.POSE_CONNECTIONS, { color: "#85B7EB", lineWidth: 3 });
    draw.drawLandmarks(signer, { color: "#378ADD", radius: 3 });
  }
  if (showHands.checked) {
    for (const h of hands.landmarks) {
      draw.drawConnectors(h, HandLandmarker.HAND_CONNECTIONS, { color: "#5DCAA5", lineWidth: 4 });
      draw.drawLandmarks(h, { color: "#F0997B", radius: 4 });
    }
  }
  if (showFace.checked && face.faceLandmarks[0]) {
    const f = face.faceLandmarks[0];
    draw.drawConnectors(f, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#888780", lineWidth: 1 });
    draw.drawConnectors(f, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#AFA9EC", lineWidth: 3 });
    draw.drawConnectors(f, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#AFA9EC", lineWidth: 3 });
  }

  // ---------- Action 1: thumbs-up ----------
  const anyThumb = hands.landmarks.some(isThumbsUp);
  thumbFrames = anyThumb ? thumbFrames + 1 : 0;
  const thumbOn = thumbFrames >= 10;
  $("thumb").textContent = thumbOn ? "Thumbs-up detected" : anyThumb ? "Hold it…" : "Not detected";
  $("thumb").classList.toggle("on", thumbOn);

  const states = hands.landmarks[0] ? fingerStates(hands.landmarks[0]) : [false, false, false, false, false];
  states.forEach((up, i) => $("f" + i).classList.toggle("up", up));

  // ---------- Action 2: raised eyebrows ----------
  const blend = face.faceBlendshapes?.[0]?.categories;
  if (blend) {
    const score = browScore(blend);
    $("browVal").textContent = score.toFixed(2);
    $("browFill").style.width = `${Math.min(100, score * 100)}%`;
    if (brow.baseline === null) {
      brow.push(score);
      $("brow").textContent = `Calibrating… ${Math.round(brow.progress * 100)}%`;
      const base = brow.baseline as number | null;   // push() may have just set it
      if (base !== null) {
        $("baseVal").textContent = base.toFixed(2);
        $("browMark").style.display = "block";
        $("browMark").style.left = `${Math.min(100, base * 100)}%`;
      }
    } else {
      const raised = brow.isRaised(score, Number(thr.value));
      $("brow").textContent = raised ? "Eyebrows raised (question)" : "Neutral";
      $("brow").classList.toggle("on", raised);
    }
  } else {
    $("brow").textContent = "Waiting for face…";
    $("brow").classList.remove("on");
  }

  // ---------- Feature vector (spec v1) ----------
  const a = video.videoWidth / video.videoHeight;
  const [hA, hB] = signer ? assignHands(hands.landmarks, signer, a) : [null, null];
  const fv = featurize(signer, hA, hB, signer ? pickFaceBlend(face, signer, a) : {}, a);

  $("ftPose").textContent = signer ? "yes" : "no";
  $("ftA").textContent = fv[0] > 0 ? "present" : "absent";
  $("ftB").textContent = fv[64] > 0 ? "present" : "absent";
  $("ftAy").textContent = fv[0] > 0 ? fv[2].toFixed(2) : "–";
  $("ftBy").textContent = fv[64] > 0 ? fv[66].toFixed(2) : "–";

  // ---------- Sign recognition ----------
  if (clf) {
    const segment = segmenter.push(fv);
    updateSegState();
    if (segment) void recognize(clf, segment);
  }

  if (recording) {
    recording.push(Array.from(fv, (v) => Math.round(v * 1e5) / 1e5));
    setStatus(`Recording… ${recording.length}/${RECORD_FRAMES} frames`);
    if (recording.length >= RECORD_FRAMES) finishRecording();
  }

  // ---------- Performance ----------
  frameCount++;
  const now = performance.now();
  if (now - fpsStart >= 1000) {
    $("fps").textContent = String(Math.round((frameCount * 1000) / (now - fpsStart)));
    frameCount = 0;
    fpsStart = now;
    $("tPose").textContent = fmtMs(times.pose);
    $("tHands").textContent = fmtMs(times.hands);
    $("tFace").textContent = fmtMs(times.face);
  }
}

// ---------- Recognition UI ----------
const pretty = (label: string) => (label === NOTHING ? "(no sign)" : label.replace(/_/g, " ").toLowerCase());

function showVocabulary(labels: string[]) {
  const signs = labels.filter((l) => l !== NOTHING);
  $("vocabSummary").textContent = `Known signs (${signs.length})`;
  $("vocab").textContent = signs.map(pretty).join(" · ");
  $("segState").textContent = "Ready: sign, then lower your hands";
  // Recording labels: every known sign plus NOTHING, so recordings can go straight into training
  const sel = $<HTMLSelectElement>("recLabel");
  sel.replaceChildren(new Option("unlabelled", ""), new Option("NOTHING (no sign)", NOTHING),
    ...signs.map((l) => new Option(pretty(l), l)));
}

function updateSegState() {
  const s = segmenter.state;
  if (s === lastSegState) return;
  lastSegState = s;
  const el = $("segState");
  el.textContent = s === "ACTIVE" ? "Signing…" : "Ready: sign, then lower your hands";
  el.classList.toggle("active", s === "ACTIVE");
}

async function recognize(c: SignClassifier, frames: Float32Array[]) {
  const presence = handPresence(frames);
  if (presence < MIN_HAND_PRESENCE) {
    $("preds").replaceChildren(Object.assign(document.createElement("li"), {
      textContent: `Hand lost in ${Math.round((1 - presence) * 100)}% of the sign: step back so your raised hands stay in view, add light in front`,
    }));
    return;
  }
  const t0 = performance.now();
  const preds = await c.classify(resample(frames));
  $("inferMs").textContent = `${(performance.now() - t0).toFixed(1)} ms`;
  renderPredictions(preds);
  if (preds[0].label !== NOTHING && preds[0].p >= MIN_CONFIDENCE) {
    history.push(`${pretty(preds[0].label)} ${Math.round(preds[0].p * 100)}%`);
    if (history.length > HISTORY_MAX) history.shift();
    renderHistory();
  }
}

function renderPredictions(preds: Prediction[]) {
  $("preds").replaceChildren(...preds.map(({ label, p }) => {
    const li = document.createElement("li");
    const fill = Object.assign(document.createElement("div"), { className: "fillbar" });
    fill.style.width = `${Math.round(p * 100)}%`;
    li.append(fill,
      Object.assign(document.createElement("span"), { textContent: pretty(label) }),
      Object.assign(document.createElement("b"), { textContent: `${Math.round(p * 100)}%` }));
    return li;
  }));
}

function renderHistory() {
  $("history").replaceChildren(...history.map((h) =>
    Object.assign(document.createElement("span"), { className: "chip", textContent: h })));
}

function finishRecording() {
  const frames = recording!;
  recording = null;
  const label = $<HTMLSelectElement>("recLabel").value;
  const payload = {
    featureSpecVersion: FEATURE_SPEC_VERSION,
    label: label || undefined,
    width: video.videoWidth, height: video.videoHeight,
    frameCount: frames.length,
    frames,
  };
  const name = `features-${label ? label + "-" : ""}${Date.now()}.json`;
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload)], { type: "application/json" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: name });
  link.click();
  URL.revokeObjectURL(url);
  setStatus(`Saved ${name}. Move labelled recordings into python/data/own/ to train on them.`);
}

$("start").addEventListener("click", start);
$("recal").addEventListener("click", () => {
  brow.reset();
  $("browMark").style.display = "none";
  $("baseVal").textContent = "–";
  setStatus("Recalibrating. Keep a neutral face for a second.");
});
$("record").addEventListener("click", () => {
  if (!lm) { setStatus("Start the camera first."); return; }
  recording = [];
});
$("clearHistory").addEventListener("click", () => {
  history.length = 0;
  renderHistory();
  $("preds").replaceChildren();
});
thr.addEventListener("input", () => { $("thrVal").textContent = Number(thr.value).toFixed(2); });