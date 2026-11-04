import "./style.css";
import "./demo.css";
import { DrawingUtils, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import { createLandmarkers, type Landmarkers } from "./landmarkers";
import { isThumbsUp, BrowCalibrator } from "./actions";
import { assignHands, pickFaceBlend, featurize } from "./features";
import { Segmenter, resample, handPresence, segmenterForFps, MIN_HAND_PRESENCE, NOTHING } from "./segmenter";
import { loadClassifier, type SignClassifier, type Prediction } from "./classifier";
import { toWav16k } from "./wav";
import { Avatar } from "./avatar";

type SignEvent = { candidates: Prediction[]; question: boolean; manual?: boolean };

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>("video");
const canvas = $<HTMLCanvasElement>("canvas");
const ctx = canvas.getContext("2d")!;
const draw = new DrawingUtils(ctx);
const clip = $<HTMLVideoElement>("clip");
const langSel = $<HTMLSelectElement>("lang");
const camMode = $<HTMLSelectElement>("camMode");

const MIN_CONFIDENCE = 0.35;      // weaker signs are dropped (and the user is told)
const BROW_RAISE = 0.15;          // brow level above the neutral baseline that marks a question
const THUMB_MS = 350;             // a thumbs-up held this long sends the sentence
const COOLDOWN_MS = 1500;         // ignore signs right after sending (the thumbs-up itself)
const MAX_RECORD_MS = 25000;      // keep each spoken turn short, so transcription stays quick
const DEBUG_FRAMES = 300;         // "Save recording" in browser mode keeps this many raw feature frames

let lm: Landmarkers | null = null;
let clf: SignClassifier | null = null;
const segmenter = new Segmenter();
const brow = new BrowCalibrator(30);
let lastVideoTime = -1, lastTs = 0, thumbSince = 0, thumbSent = false, cooldownUntil = 0, busy = false;
let frameNo = 0, fpsFrames = 0, fpsStart = performance.now(), fps = 0, lastAttempt = "";
let lastFace: ReturnType<Landmarkers["face"]["detectForVideo"]> | null = null;
let lastPose: ReturnType<Landmarkers["pose"]["detectForVideo"]> | null = null;
const times = { pose: 0, hands: 0, face: 0 };
function timed<T>(key: keyof typeof times, fn: () => T): T {
  const t0 = performance.now();
  const r = fn();
  const dt = performance.now() - t0;
  times[key] = times[key] ? times[key] * 0.8 + dt * 0.2 : dt;
  return r;
}
const recent: Float32Array[] = [];
const sentence: SignEvent[] = [];
let clipIndex: Record<string, string> = {};
let lastSigns: string[] = [];

const pretty = (l: string) => l.replace(/_/g, " ").toLowerCase();
const banner = (text: string, kind: "" | "active" | "ok" | "bad" = "") => {
  const b = $("banner");
  if (b.textContent === text && b.className === `banner ${kind}`) return;   // called every frame
  b.textContent = text;
  b.className = `banner ${kind}`;
};
const showDiag = (engine = lm?.delegate ?? "") => {
  const ms = (k: keyof typeof times) => `${k} ${Math.round(times[k])}`;
  $("diag").textContent = `${fps} fps (${engine} ms: ${ms("hands")}, ${ms("pose")}, ${ms("face")})` +
    (fps && fps < 8 ? " · slow: close other apps (Docker, extra browser tabs) to free memory and CPU" : "") +
    (lastAttempt ? ` · last attempt: ${lastAttempt}` : "");
};
/** Buttons hold an icon and a <span class="lbl">: change the words, keep the icon. */
const setLabel = (btn: HTMLElement, text: string) => { btn.querySelector(".lbl")!.textContent = text; };
const chip = (text: string, cls = "") => Object.assign(document.createElement("span"), { className: `chip ${cls}`, textContent: text });

// ---------------------------------------------------------------- Server
async function checkServer() {
  const pill = $("server");
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    langSel.replaceChildren(...Object.entries(h.languages as Record<string, string>).map(([code, name]) => new Option(name, code)));
    langSel.value = localStorage.getItem("lang") ?? "en-IN";
    pill.textContent = h.configured ? `Connected: ${h.provider}` : "No API key: offline fallbacks only";
    pill.className = `pill ${h.configured ? "ok" : "warn"}`;
  } catch {
    pill.textContent = "Server not running (start it in /server)";
    pill.className = "pill warn";
    langSel.replaceChildren(new Option("English", "en-IN"));
  }
  $<HTMLButtonElement>("mic").disabled = false;
}
langSel.addEventListener("change", () => localStorage.setItem("lang", langSel.value));

// ---------------------------------------------------------------- Camera + recognition
async function start() {
  const btn = $<HTMLButtonElement>("start");
  btn.disabled = true;
  if (camMode.value === "native") return startNative(btn);
  try {
    banner("Loading models…");
    [lm, clf] = await Promise.all([createLandmarkers(), loadClassifier()]);
    showVocabulary(clf.labels);
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, frameRate: 30 }, audio: false });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    setLabel(btn, "Camera on");
    banner("Keep a neutral face for a second…");
    requestAnimationFrame(tick);
  } catch (err) {
    console.error(err);
    btn.disabled = false;
    banner(`Couldn't start: ${(err as Error)?.message ?? err}`, "bad");
  }
}

// Native camera mode: python/vision_server.py reads the webcam and recognizes signs outside the browser
// (much faster on a busy laptop); the page shows its preview and reacts to its events.
async function startNative(btn: HTMLButtonElement) {
  try {
    banner("Starting native camera…");
    clf = clf ?? (await loadClassifier());
    showVocabulary(clf.labels);
    await fetch("/vision/start", { method: "POST" }).then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); });
    video.hidden = canvas.hidden = true;
    const img = $<HTMLImageElement>("nativeView");
    img.hidden = false;
    img.src = `/vision/stream.mjpg?t=${Date.now()}`;
    const es = new EventSource("/vision/events");
    es.onmessage = (e) => onNativeEvent(JSON.parse(e.data));
    es.onerror = () => banner("Lost the native camera (is vision_server.py running?)", "bad");
    setLabel(btn, "Camera on (native)");
  } catch (err) {
    btn.disabled = false;
    banner(`Native camera unavailable: ${(err as Error).message}. Start python/vision_server.py or pick "In browser".`, "bad");
  }
}

type NativeEvent =
  | { type: "sign"; candidates: Prediction[]; question: boolean; presence: number; frames: number }
  | { type: "state"; state: "ACTIVE" | "REST" | "NOBODY" }
  | { type: "fps"; fps: number; ms: Record<string, number> }
  | { type: "thumb" }
  | { type: "status" | "error"; text: string };

function onNativeEvent(ev: NativeEvent) {
  if (ev.type === "sign") {
    if (performance.now() > cooldownUntil) handleSign(ev.candidates, ev.presence, ev.frames, ev.question);
  } else if (ev.type === "state") {
    if (ev.state === "ACTIVE") banner("Signing…", "active");
    else if (ev.state === "NOBODY") banner("Step into view", "");
    else if ($("banner").classList.contains("active")) banner("Ready: sign, then lower your hands", "ok");
  } else if (ev.type === "fps") {
    fps = ev.fps;
    Object.assign(times, ev.ms);
    showDiag("native");
  } else if (ev.type === "thumb") {
    if (sentence.length && !busy) { cooldownUntil = performance.now() + COOLDOWN_MS; void speakSentence(); }
  } else {
    banner(ev.text, ev.type === "error" ? "bad" : "ok");
  }
}

function tick() {
  if (lm && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    processFrame(lm);
  }
  requestAnimationFrame(tick);
}

function processFrame(m: Landmarkers) {
  const ts = Math.max(lastTs + 1, Math.round(performance.now()));
  lastTs = ts;
  // Hands every frame (they carry the sign). The body moves slowly, so pose every other frame is enough,
  // and the face only feeds the eyebrow (question) signal: every third frame. Keeps slow laptops usable.
  const n = frameNo++;
  const pose = n % 2 === 0 || !lastPose ? (lastPose = timed("pose", () => m.pose.detectForVideo(video, ts))) : lastPose;
  const hands = timed("hands", () => m.hands.detectForVideo(video, ts));
  const face = n % 3 === 0 || !lastFace ? (lastFace = timed("face", () => m.face.detectForVideo(video, ts))) : lastFace;
  const signer = pose.landmarks[0] ?? null;

  fpsFrames++;
  if (performance.now() - fpsStart >= 1000) {
    fps = Math.round((fpsFrames * 1000) / (performance.now() - fpsStart));
    fpsFrames = 0;
    fpsStart = performance.now();
    if (segmenter.state === "REST") segmenter.cfg = segmenterForFps(fps);   // same timing at any frame rate
    showDiag();
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (signer) draw.drawConnectors(signer, PoseLandmarker.POSE_CONNECTIONS, { color: "#85B7EB", lineWidth: 2 });
  for (const h of hands.landmarks) {
    draw.drawConnectors(h, HandLandmarker.HAND_CONNECTIONS, { color: "#5DCAA5", lineWidth: 4 });
    draw.drawLandmarks(h, { color: "#F0997B", radius: 3 });
  }

  const a = video.videoWidth / video.videoHeight;
  const [hA, hB] = signer ? assignHands(hands.landmarks, signer, a) : [null, null];
  const fv = featurize(signer, hA, hB, signer ? pickFaceBlend(face, signer, a) : {}, a);
  recent.push(fv);
  if (recent.length > DEBUG_FRAMES) recent.shift();

  if (!signer) { banner("Step into view", ""); return; }
  // Neutral brow level, learned from the first second with a face (questions need it; signing doesn't)
  if (brow.baseline === null && face.faceLandmarks.length) {
    brow.push((fv[149] + fv[150] + fv[151]) / 3);
    if ((brow.baseline as number | null) !== null) banner("Ready: sign, then lower your hands", "ok");
  }

  // Thumbs-up (held) sends the sentence
  const thumb = hands.landmarks.some(isThumbsUp);
  thumbSince = thumb ? thumbSince || performance.now() : 0;
  if (!thumb) thumbSent = false;
  if (thumb && performance.now() - thumbSince >= THUMB_MS && !thumbSent && sentence.length && !busy) {
    thumbSent = true;
    segmenter.reset();
    cooldownUntil = performance.now() + COOLDOWN_MS;
    void speakSentence();
    return;
  }

  const segment = segmenter.push(fv);
  if (segmenter.state === "ACTIVE") banner("Signing…", "active");
  else if ($("banner").classList.contains("active")) banner("Ready: sign, then lower your hands", "ok");
  if (segment && performance.now() > cooldownUntil && clf) void recognize(clf, segment);
}

async function recognize(c: SignClassifier, frames: Float32Array[]) {
  const preds = await c.classify(resample(frames));
  // Question = brows clearly above baseline for a good part of the sign (80th percentile)
  const levels = frames.map((f) => (f[149] + f[150] + f[151]) / 3).sort((x, y) => x - y);
  const question = brow.baseline !== null && levels[Math.floor(levels.length * 0.8)] - brow.baseline > BROW_RAISE;
  handleSign(preds, handPresence(frames), frames.length, question);
}

/** Accept a recognized sign, or offer its best guesses (both camera modes end here). */
function handleSign(preds: Prediction[], presence: number, nFrames: number, question: boolean) {
  const top = preds[0];
  const signs = preds.filter((p) => p.label !== NOTHING);
  lastAttempt = `${nFrames} frames, hand seen ${Math.round(presence * 100)}% · ` +
    preds.map((p) => `${p.label === NOTHING ? "no sign" : pretty(p.label)} ${Math.round(p.p * 100)}%`).join(", ");
  showDiag(camMode.value === "native" ? "native" : undefined);
  // Rejected or unsure: offer the best guesses as one-tap buttons instead of making the user sign again
  if (presence < MIN_HAND_PRESENCE) return suggest("Hand was hard to see (step back, light in front).", signs);
  if (top.label === NOTHING) return suggest("Didn't look like a sign.", signs);
  if (top.p < MIN_CONFIDENCE) return suggest("Not sure.", signs);
  sentence.push({ candidates: signs, question });
  banner(`${pretty(top.label)}${question ? " ?" : ""} (${Math.round(top.p * 100)}%)`, "ok");
  suggest("", []);
  renderSentence();
}

function suggest(reason: string, candidates: Prediction[]) {
  if (reason) banner(reason, "bad");
  const box = $("suggest");
  box.replaceChildren();
  if (!candidates.length) return;
  box.append(Object.assign(document.createElement("span"), { className: "suggest-label", textContent: "Did you mean" }),
    ...candidates.slice(0, 2).map((p) => {
      const el = chip(`${pretty(p.label)}?`, "maybe");
      el.onclick = () => {
        sentence.push({ candidates: [p, ...candidates.filter((q) => q !== p)], question: false });
        box.replaceChildren();
        banner("Ready: sign, then lower your hands", "ok");
        renderSentence();
      };
      return el;
    }));
}

async function saveRecent() {
  if (camMode.value === "native") {        // the native server keeps video + tracking data; it writes the files
    const res = await fetch("/vision/save", { method: "POST" }).then((r) => r.json()).catch((e) => ({ error: String(e) }));
    if (res.error) return banner(`Couldn't save: ${res.error}`, "bad");
    return banner(`Saved ${res.seconds} s (${res.frames} frames) to ${res.folder}`, "ok");
  }
  const payload = { featureSpecVersion: 1, width: video.videoWidth, height: video.videoHeight, fps,
                    frameCount: recent.length, frames: recent.map((f) => Array.from(f, (v) => Math.round(v * 1e5) / 1e5)) };
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload)], { type: "application/json" }));
  Object.assign(document.createElement("a"), { href: url, download: `hastaak-last10s-${Date.now()}.json` }).click();
  URL.revokeObjectURL(url);
}

function renderSentence() {
  $("sentence").replaceChildren(...sentence.map((s, i) => {
    const el = chip(`${pretty(s.candidates[0].label)}${s.manual ? "" : ` ${Math.round(s.candidates[0].p * 100)}%`}`,
      `${s.question ? "q" : ""} ${s.manual ? "manual" : ""}`);
    el.title = "Remove";
    el.onclick = () => { sentence.splice(i, 1); renderSentence(); };
    return el;
  }));
  $<HTMLButtonElement>("speakBtn").disabled = !sentence.length || busy;
}

function showVocabulary(labels: string[]) {
  $("vocab").replaceChildren(...labels.filter((l) => l !== NOTHING).map((l) => {
    const el = chip(pretty(l));
    el.onclick = () => { sentence.push({ candidates: [{ label: l, p: 1 }], question: false, manual: true }); renderSentence(); };
    return el;
  }));
}

async function speakSentence() {
  if (!sentence.length || busy) return;
  busy = true;
  renderSentence();
  banner("Making a sentence…", "active");
  try {
    const res = await fetch("/api/sentence", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ signs: sentence.map(({ candidates, question }) => ({ candidates, question })), language: langSel.value }),
    }).then((r) => r.json());
    $("spoken").textContent = res.text || "–";
    addLog("deaf", res.text, sentence.map((s) => pretty(s.candidates[0].label)).join(" · ") + (res.source === "template" ? " · offline" : ""));
    if (res.error) console.warn(res.error);
    sentence.length = 0;
    if (res.audio) await playAudio(res.audio);
    banner("Ready: sign, then lower your hands", "ok");
  } catch (err) {
    banner(`Couldn't reach the server: ${(err as Error).message}`, "bad");
  } finally {
    busy = false;
    renderSentence();
  }
}

function playAudio(b64: string) {
  const audio = new Audio(`data:audio/wav;base64,${b64}`);
  return new Promise<void>((resolve) => {
    audio.onended = () => resolve();
    audio.onerror = () => resolve();
    audio.play().catch(() => resolve());
  });
}

// ---------------------------------------------------------------- Microphone → captions → signs
let recorder: MediaRecorder | null = null;
let stopTimer = 0;

async function toggleMic() {
  const btn = $<HTMLButtonElement>("mic");
  if (recorder?.state === "recording") { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
    recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks: Blob[] = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      clearTimeout(stopTimer);
      stream.getTracks().forEach((t) => t.stop());
      btn.classList.remove("rec");
      setLabel(btn, "Tap to talk");
      void sendSpeech(new Blob(chunks, { type: recorder?.mimeType || "audio/webm" }));
    };
    recorder.start();
    btn.classList.add("rec");
    setLabel(btn, "Listening… tap to stop");
    stopTimer = window.setTimeout(() => recorder?.stop(), MAX_RECORD_MS);
  } catch (err) {
    sayLine(`Microphone unavailable: ${(err as Error).message}`, "err");
  }
}

async function sendSpeech(blob: Blob) {
  const btn = $<HTMLButtonElement>("mic");
  btn.disabled = true;
  sayLine("Transcribing…", "pending");
  try {
    const form = new FormData();
    form.append("file", await toWav16k(blob), "speech.wav");    // Muse Voice Transcribe accepts only WAV
    form.append("language", "unknown");
    const res = await fetch("/api/listen", { method: "POST", body: form }).then((r) => r.json());
    if (res.error && !res.text) throw new Error(res.error);
    sayLine(res.text || "(nothing heard)", "", res.english && res.english !== res.text ? `English: ${res.english}` : "");
    addLog("hearing", res.text || "(nothing heard)", res.glosses?.length ? `ISL: ${res.glosses.join(" · ")}` : "no matching signs");
    lastSigns = res.signs ?? [];
    await playSigns(lastSigns);
  } catch (err) {
    sayLine(`Couldn't transcribe: ${(err as Error).message}`, "err");
  } finally {
    btn.disabled = false;
  }
}

/** "They said" keeps every turn; the newest line is highlighted and scrolled into view. */
function sayLine(text: string, cls = "", english = "") {
  const list = $("transcript");
  list.querySelector("li.pending")?.remove();
  list.querySelector("li.latest")?.classList.remove("latest");
  const li = Object.assign(document.createElement("li"), { className: `latest ${cls}`, textContent: text });
  if (english) li.append(Object.assign(document.createElement("span"), { className: "en", textContent: english }));
  list.append(li);
  list.scrollTop = list.scrollHeight;
}

async function playSigns(signs: string[]) {
  $<HTMLButtonElement>("replay").disabled = !signs.length;
  $("glosses").replaceChildren(...signs.map((s) => chip(pretty(s))));
  const chips = [...$("glosses").children];
  $("greet").classList.remove("show");
  if (showAvatar) {
    await avatar.play(signs, (i) => {
      chips.forEach((c, k) => c.classList.toggle("now", k === i));
      $("clipLabel").textContent = i >= 0 ? pretty(signs[i]) : "";
    });
    return;
  }
  for (const [i, s] of signs.entries()) {
    if (!clipIndex[s]) continue;
    chips.forEach((c, k) => c.classList.toggle("now", k === i));
    $("clipLabel").textContent = pretty(s);
    clip.src = `/clips/${clipIndex[s]}`;
    await new Promise<void>((resolve) => {
      clip.onended = () => resolve();
      clip.onerror = () => resolve();
      clip.play().catch(() => resolve());
    });
  }
  chips.forEach((c) => c.classList.remove("now"));
  $("clipLabel").textContent = "";
}

// ---------------------------------------------------------------- Conversation log
function addLog(who: "deaf" | "hearing", text: string, detail: string) {
  const el = Object.assign(document.createElement("div"), { className: `msg ${who}` });
  el.append(Object.assign(document.createElement("small"), { textContent: `${who === "deaf" ? "Deaf person" : "Hearing person"} · ${detail}` }), text);
  const log = $("log");
  log.append(el);                       // oldest first, like a chat
  log.scrollTop = log.scrollHeight;     // the log scrolls (demo.css): keep the newest message in view
}

// ---------------------------------------------------------------- Wiring
$("start").addEventListener("click", start);
$("speakBtn").addEventListener("click", () => void speakSentence());
$("undo").addEventListener("click", () => { sentence.pop(); renderSentence(); });
$("clear").addEventListener("click", () => { sentence.length = 0; renderSentence(); });
$("saveRecent").addEventListener("click", () => void saveRecent());

// "In ISL" shows the 2D avatar (default) or the real signer's videos
const avatar = new Avatar($<HTMLCanvasElement>("avatar"));
let showAvatar = true;
/** The avatar says hello: hands in namaste and a "Namaste, I am Hastaak" bubble by its head for a few seconds. */
function greet() {
  if (showAvatar) void avatar.greet((on) => $("greet").classList.toggle("show", on));
}
function setSignView(useAvatar: boolean) {
  const switchedToAvatar = useAvatar && !showAvatar;
  showAvatar = useAvatar;
  $("avatarBox").hidden = !useAvatar;
  if (!useAvatar) $("greet").classList.remove("show");
  if (switchedToAvatar) greet();
  clip.hidden = useAvatar;
  $("showAvatar").classList.toggle("on", useAvatar);
  $("showVideo").classList.toggle("on", !useAvatar);
  $("showAvatar").setAttribute("aria-pressed", String(useAvatar));
  $("showVideo").setAttribute("aria-pressed", String(!useAvatar));
  localStorage.setItem("signView", useAvatar ? "avatar" : "video");
}
$("showAvatar").addEventListener("click", () => setSignView(true));
$("showVideo").addEventListener("click", () => setSignView(false));
setSignView(localStorage.getItem("signView") !== "video");
greet();
$("mic").addEventListener("click", () => void toggleMic());
// Small screens stack the panels; "Flip" turns the hearing panel to face the person opposite
function setFlip(on: boolean) {
  document.body.classList.toggle("flipped", on);
  $("flip").setAttribute("aria-pressed", String(on));
  localStorage.setItem("flip", on ? "1" : "0");
}
$("flip").addEventListener("click", () => setFlip(!document.body.classList.contains("flipped")));
setFlip(localStorage.getItem("flip") === "1");
$("replay").addEventListener("click", () => void playSigns(lastSigns));
fetch("/clips/clips.json").then((r) => r.json()).then((j) => { clipIndex = j; }).catch(() => {});
loadClassifier().then((c) => showVocabulary(c.labels)).catch(() => {});   // manual chips work before the camera starts
void checkServer();
// Prefer the native camera when its server is running (much faster on a busy laptop)
fetch("/vision/health").then((r) => (r.ok ? r.json() : Promise.reject()))
  .then(() => { camMode.value = localStorage.getItem("camMode") ?? "native"; })
  .catch(() => { camMode.value = "browser"; });
camMode.addEventListener("change", () => localStorage.setItem("camMode", camMode.value));
