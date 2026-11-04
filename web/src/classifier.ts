import * as ort from "onnxruntime-web/wasm";
import { FEAT_DIM, FEATURE_SPEC_VERSION } from "./features";
import { SEQ_LEN } from "./segmenter";

export type Prediction = { label: string; p: number };

const FACE_START = 149;   // features 149–155 are the face blendshapes (the last 7 of each frame)

export type SignClassifier = {
  labels: string[];
  classify: (x: Float32Array) => Promise<Prediction[]>;
};

/** Loads signnet.onnx + signnet.json written by python/export_onnx.py (no CDN needed). */
export async function loadClassifier(base = "/models"): Promise<SignClassifier> {
  const res = await fetch(`${base}/signnet.json`);
  if (!res.ok) throw new Error("signnet.json not found. Run export_onnx.py in the python folder.");
  const meta = (await res.json()) as { labels: string[]; featureSpecVersion: number };
  if (meta.featureSpecVersion !== FEATURE_SPEC_VERSION) {
    throw new Error(`Model uses feature spec ${meta.featureSpecVersion}, app uses ${FEATURE_SPEC_VERSION}.`);
  }

  // The WASM file is copied to public/ort by scripts/copy-wasm.mjs. Only the .wasm is given, so the
  // bundle keeps using its built-in loader script. Single-threaded: no cross-origin isolation headers needed.
  ort.env.wasm.wasmPaths = { wasm: "/ort/ort-wasm-simd-threaded.wasm" };
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(`${base}/signnet.onnx`, { executionProviders: ["wasm"] });
  const [inputName] = session.inputNames;
  const [outputName] = session.outputNames;

  return {
    labels: meta.labels,
    async classify(resampled: Float32Array) {
      // Zero the 7 face values: INCLUDE signers were too far away for face tracking, so the model never
      // learned faces (same as python/dataset.model_input). Faces still drive question detection.
      const x = resampled.slice();
      for (let t = 0; t < SEQ_LEN; t++) x.fill(0, t * FEAT_DIM + FACE_START, (t + 1) * FEAT_DIM);
      const out = await session.run({ [inputName]: new ort.Tensor("float32", x, [1, SEQ_LEN, FEAT_DIM]) });
      const logits = out[outputName].data as Float32Array;
      const max = Math.max(...logits);
      const exps = Array.from(logits, (v) => Math.exp(v - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      return meta.labels
        .map((label, i) => ({ label, p: exps[i] / sum }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 3);
    },
  };
}
