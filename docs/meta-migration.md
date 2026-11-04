# Moving to Meta's models

The prototype uses non-Meta parts so it could be tested without spending Meta credits (decisions.md D-001).
For the hackathon version, each part has a Meta replacement. **Check every model name and endpoint below against
the Meta Model API docs and the hackathon rules before relying on it.** They come from planning notes, not from
tested code.

## What replaces what
| Job | Now | Meta version | Effort |
| --- | --- | ------------ | ------ |
| Signs → sentence; speech → sign plan | Sarvam chat (`sarvam-105b`) | Muse Spark (Meta Model API) | small: `server/meta.py` |
| Speech → text | Sarvam Saaras | Muse Voice Transcribe, or open-weights SeamlessM4T v2 / MMS / Omnilingual ASR | small (API) or a GPU host |
| Text → voice | Sarvam Bulbul | MMS-TTS (open weights, Indian languages incl. Hindi) | needs somewhere to run it |
| Hand / body / face tracking | MediaPipe (Google) | Sapiens pose (keypoints incl. hands and face), or V-JEPA 2 on whole clips | **large** |
| Running SignNet | ONNX Runtime (Microsoft) | PyTorch on the server; ExecuTorch on a phone | medium |
| Signer lock in a crowd | not done (largest person) | SAM 3.1 | new feature |
| Sign videos | cropped INCLUDE clips | your own signer cut out by SAM 3.1 onto one background; SAM 3D Body for an avatar | offline job |
| Extra training backgrounds | none | SAM 3.1 masks + Muse Image scenes | offline job |
| Web app | Vite + TypeScript | React / React Native | optional |

## Step 1: language and speech (small)
The backend calls its provider through three functions (`server/sarvam.py`):
```python
async def chat_json(system: str, user: str) -> dict          # one JSON reply
async def transcribe(audio: bytes, filename: str, mime: str, language: str = "unknown") -> dict   # {"text", "language"}
async def speak(text: str, language: str) -> str              # base64 WAV
```
1. Write `server/meta.py` with the same three functions.
2. Add the Meta key and model names to `server/.env` (and their names to `.env.example`); set `PROVIDER=meta`.
3. Re-run the speech and language checks in testing.md and compare with the Sarvam results there.
4. Record the switch and the numbers in decisions.md.

Nothing else changes: prompts, rules, keyword checks, the page. Check that the chosen models support Hindi and the
other listed languages; drop any they don't from `LANGUAGES` in `server/main.py`.

## Step 2: sign recognition (large)
Replacing MediaPipe changes the model's input, so it is not a drop-in swap:
1. **Feature spec v2** from Sapiens keypoints: the two hand slots, the same 7 body points, face points if useful.
   Same normalization (shoulder midpoint, shoulder width, aspect-corrected x). Bump `FEATURE_SPEC_VERSION`.
2. **Re-extract** INCLUDE and your own recordings with Sapiens on a cloud GPU (Kaggle notebooks or Modal; this
   laptop has none). Recordings must be kept as video for this: the v1 feature files can't be converted.
3. **Retrain** SignNet: `train.py` works unchanged once v2 features exist. Evaluate and export as now.
4. **Serve recognition where Sapiens runs.** `vision_server.py` already does camera → tracking → segmenter → model
   → events outside the browser, so it becomes the GPU service. Send it frames or ~1 s chunks from the page or a
   phone, rather than reading a local webcam. The in-browser camera mode goes away.
5. Keep the segmenter, bridging, low-fps timing, hand-visibility check and "Did you mean" logic: they are
   independent of the tracker.

## What will be different
- **Video leaves the device** for recognition. Process frames in memory, never store them on the server, and say
  so to users (security.md).
- **Extra delay per sign** (upload + GPU). Batch frames per chunk; keep the server warm during the demo.
- **SAM 3.1 signer lock:** bystanders' hands are ignored, which the current version can't do.
- **Sign videos** from your own signer, or an avatar driven by SAM 3D Body.
- **Credits:** every Meta call costs. Keep the rules-first fallbacks, cache identical requests, and use the
  logged Sarvam inputs and outputs as a fixed test set, so Muse Spark can be compared in one run.
- **Accuracy:** the lessons in design.md still apply. Own recordings matter more than the tracking model.
