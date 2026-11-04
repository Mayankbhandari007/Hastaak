import { FilesetResolver, HandLandmarker, FaceLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

export type Delegate = "GPU" | "CPU";

/** Loads the three MediaPipe models from the local public/ folder (no CDN needed). */
export async function createLandmarkers(delegate: Delegate = "GPU") {
  const fileset = await FilesetResolver.forVisionTasks("/wasm");
  const base = (file: string) => ({ modelAssetPath: `/models/${file}`, delegate });

  const [hands, face, pose] = await Promise.all([
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: base("hand_landmarker.task"), runningMode: "VIDEO", numHands: 2,
    }),
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: base("face_landmarker.task"), runningMode: "VIDEO", numFaces: 1,
      outputFaceBlendshapes: true,
    }),
    PoseLandmarker.createFromOptions(fileset, {
      baseOptions: base("pose_landmarker_lite.task"), runningMode: "VIDEO", numPoses: 1,
    }),
  ]);
  return { hands, face, pose, delegate };
}

export type Landmarkers = Awaited<ReturnType<typeof createLandmarkers>>;

/** The WebGL renderer, e.g. "ANGLE (Intel, ...)" or "SwiftShader" (software: the GPU delegate will be very slow). */
export function glRenderer(): string {
  const gl = document.createElement("canvas").getContext("webgl2") ?? document.createElement("canvas").getContext("webgl");
  if (!gl) return "none";
  const ext = gl.getExtension("WEBGL_debug_renderer_info");
  return String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
}

export function closeLandmarkers(m: Landmarkers) {
  m.hands.close();
  m.face.close();
  m.pose.close();
}
