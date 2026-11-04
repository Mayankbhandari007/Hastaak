import { copyFileSync, cpSync, existsSync, mkdirSync } from "node:fs";

const src = "node_modules/@mediapipe/tasks-vision/wasm";
const dst = "public/wasm";

if (!existsSync(src)) {
  console.error("MediaPipe WASM folder not found. Did npm install finish?");
  process.exit(1);
}
mkdirSync(dst, { recursive: true });
cpSync(src, dst, { recursive: true });
console.log(`Copied MediaPipe WASM files to ${dst}`);

// onnxruntime-web/wasm (CPU build) only needs this one file at runtime
const ortSrc = "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm";
const ortDst = "public/ort";
if (existsSync(ortSrc)) {
  mkdirSync(ortDst, { recursive: true });
  copyFileSync(ortSrc, `${ortDst}/ort-wasm-simd-threaded.wasm`);
  console.log(`Copied ONNX Runtime WASM file to ${ortDst}`);
}
