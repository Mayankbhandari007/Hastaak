# Architecture

Read alongside coding-standards.md and data-formats.md. Root CLAUDE.md is the index.

## Components
```
 Browser: web/demo.html (Vite dev server, port 5173)
   ├─ camera mode "native":  <img> preview  ◀── MJPEG ──┐   events ◀── SSE ──┐
   ├─ camera mode "browser": MediaPipe + segmenter + SignNet (ONNX/WASM) in JS │
   ├─ microphone (MediaRecorder) → 16 kHz WAV (wav.ts)                        │
   ├─ "In ISL": 2D avatar (avatar.ts, public/avatar/) or real-signer videos   │
   │        │ /api/*  (proxied)                     /vision/* (proxied)     │
   ▼        ▼                                              ▼                 │
 server/main.py (FastAPI, port 8000)           python/vision_server.py (FastAPI, port 8001)
   ├─ /api/sentence → meta.chat_json + speak         webcam (OpenCV) → MediaPipe → features
   ├─ /api/listen   → meta.transcribe + chat_json    → segmenter → SignNet (PyTorch) ──────┘
   └─ meta.py (all Meta calls)                      → rolling 20 s buffer → /vision/save
        ├─ Muse Spark 1.3, Voice Transcribe ─▶ api.meta.ai ◀── SAM 3.1 (signer lock, --signer-lock)
        └─ MMS-TTS voices, run locally
```
- **Web app** (`web/`): `demo.html` is Hastaak. `index.html` is the playground and recorder, for debugging
  tracking and recording labelled clips. Vite proxies `/api` → 8000 and `/vision` → 8001.
- **Backend** (`server/`): holds the API key and calls Meta's models. It never sees video.
- **Native vision server** (`python/vision_server.py`): reads the webcam, recognizes signs, streams the annotated
  preview and events, and saves debugging recordings. Video stays on the laptop, except single frames sent to
  SAM 3.1 when signer lock is on and 2+ people are in view.
- **Python tools** (`python/`): feature extraction, training, evaluation, export, sign videos, avatar motion,
  the Sapiens2 comparison.

The camera modes share the same model (`signnet.onnx` in the browser, `signnet.pt` in Python) and identical logic.
Only one of them may use the webcam at a time.

## Folder layout
```
CLAUDE.md                 # index (this project's rules + glossary)
README.md                 # short intro + quick start
docs/                     # this folder
server/
  main.py                 # FastAPI app: /api/health, /api/sentence, /api/listen; prompts; validation
  meta.py                 # chat_json (Muse Spark), transcribe (Voice Transcribe), speak (MMS-TTS),
                          #   plus find_people / person_masks (SAM 3.1); the only file that calls Meta
  rules.py                # offline fallbacks: KEYWORDS, keyword_signs, template_sentence
  .env.example            # names of settings (copy to .env)
web/
  demo.html, index.html   # Hastaak page; playground/recorder page
  src/demo.ts, demo.css   # Hastaak logic (both camera modes, mic, avatar/clips, log); one-screen layout (D-030)
  src/avatar.ts           # 2D signing avatar (replays public/avatar/<SIGN>.json)
  src/wav.ts              # microphone recording → 16 kHz mono WAV
  src/main.ts, style.css  # playground logic; style.css is shared
  src/features.ts         # feature spec v1 (twin of python/features.py)
  src/segmenter.ts        # segmenter + resample + low-fps settings (twin of python/segmenter.py)
  src/classifier.ts       # loads signnet.onnx/json, zeroes face values, returns top 3
  src/landmarkers.ts      # MediaPipe models (GPU/CPU delegate)
  src/actions.ts          # thumbs-up, finger states, eyebrow calibration
  scripts/                # copy-wasm.mjs (MediaPipe + ONNX WASM → public/), download-models.mjs
  public/models/          # MediaPipe .task files + signnet.onnx/.json (browser) + signnet.pt (vision_server)
  public/clips/           # one WebM per sign + clips.json
  public/avatar/          # one motion file per sign + index.json
  public/brand/           # Hastaak logo + icon (navy and -light variants), favicon.svg
python/
  features.py             # feature spec v1 (twin of web/src/features.ts)
  segmenter.py            # Segmenter, cut_sign, hand_presence, low_fps_settings (twin of segmenter.ts)
  dataset.py              # load_seq, model_input, augment, SignDataset
  model.py                # SignNet
  train.py, evaluate.py, export_onnx.py
  include_manifest.py, extract_include.py, extract.py   # INCLUDE → features
  synth_variants.py       # webcam-style video variants of Greetings/Pronouns clips (+ --finish: QC, manifests)
  compare_models.py       # checkpoints side by side: clean test vs webcam-variant test, per category
  make_clips.py           # sign videos for the web app (--sam: SAM 3.1 cut-outs)
  export_avatar.py        # sign motion for the 2D avatar
  vision_server.py        # native camera mode (--signer-lock: SAM 3.1)
  sapiens_check.py        # Sapiens2 vs MediaPipe comparison (speed, fingertips, pictures)
  live.py, live_classify.py, common.py                  # desktop tools; shared MediaPipe setup
  data/                   # include/ (videos), include_feats/ (.npy), own/ (your recordings) — not shared
  runs/<name>/            # best.pt, labels.json, confusions.csv
  recordings/<time>/      # debugging saves from native mode — contain video of people
```

## Key libraries
| Need | Library | Note |
| ---- | ------- | ---- |
| Landmarks | MediaPipe Tasks (`@mediapipe/tasks-vision`, `mediapipe`) | hands, pose lite, face + blendshapes |
| Classifier training + native runtime | PyTorch | CPU is enough |
| Classifier in the browser | ONNX Runtime (`onnxruntime-web/wasm`) | browser fallback mode only |
| Voices | Transformers + PyTorch (CPU), `facebook/mms-tts-*` | MMS-TTS, downloaded on first use |
| SAM 3.1 output | `meta-sam-parser` | decodes boxes and masks |
| Video I/O | OpenCV | webcam, clips (VP8 WebM), recordings |
| Backend | FastAPI + uvicorn + httpx | `python-dotenv` reads `server/.env` |
| Web | Vite + TypeScript (no framework) | `strict` TypeScript |

## Endpoints
| Endpoint | Body / reply | Notes |
| -------- | ------------ | ----- |
| `GET /api/health` | `provider` ("Meta"), `configured`, signs, showable signs, languages | the page's "Connected" pill |
| `POST /api/sentence` | `{signs: [{candidates, question}], language, speak}` → `{text, source, signs, audio?, error?}` | Deaf → hearing |
| `POST /api/listen` | multipart `file` (≤ 30 s audio), `language` → `{text, language, english, signs, glosses, source, error?}` | hearing → Deaf |
| `GET /vision/health` | `{running, labels}` | the page picks native mode when this answers |
| `POST /vision/start` · `/vision/stop` | `{running}` | opens / releases the webcam |
| `GET /vision/stream.mjpg` | MJPEG | annotated, mirrored preview (640 px) |
| `GET /vision/events` | server-sent events | `state`, `sign`, `thumb`, `fps`, `status`, `error` (data-formats.md) |
| `POST /vision/save` | `{folder, seconds, frames, fps}` | writes the last 20 s to `python/recordings/` |

CORS on the backend allows only `http://localhost:5173`. Both servers listen on localhost only.
