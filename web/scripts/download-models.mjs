import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const MODELS = {
  "hand_landmarker.task":
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  "face_landmarker.task":
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  "pose_landmarker_lite.task":
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
};
const TARGETS = ["public/models", "../python/models"];   // same files for browser and Python

for (const dir of TARGETS) mkdirSync(dir, { recursive: true });

for (const [name, url] of Object.entries(MODELS)) {
  if (TARGETS.every((d) => existsSync(join(d, name)))) {
    console.log(`✓ ${name} already present`);
    continue;
  }
  process.stdout.write(`Downloading ${name}… `);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download ${name}: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  for (const dir of TARGETS) writeFileSync(join(dir, name), buf);
  console.log(`${(buf.length / 1e6).toFixed(1)} MB`);
}
console.log("Models ready in web/public/models and python/models");