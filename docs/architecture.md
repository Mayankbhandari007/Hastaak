# Architecture

Read alongside coding-standards.md and data-formats.md. Root CLAUDE.md is the index.

## Components
```
 Browser: web/demo.html (Vite dev server, port 5173)
   ├─ camera mode "native":  <img> preview  ◀── MJPEG ──┐   events ◀── SSE ──┐
   ├─ camera mode "browser": MediaPipe + segmenter + SignNet (ONNX/WASM) in JS │
   ├─ microphone (MediaRecorder, WebM/Opus)                                  │
   │        │ /api/*  (proxied)                     /vision/* (proxied)     │
   ▼        ▼                                              ▼                 │
 server/main.py (FastAPI, port 8000)           python/vision_server.py (FastAPI, port 8001)
   ├─ /api/sentence → provider.chat_json + speak     webcam (OpenCV) → MediaPipe → features
   ├─ /api/listen   → provider.transcribe + chat     → segmenter → SignNet (ONNX Runtime) ──┘
   └─ provider = sarvam.py → api.sarvam.ai           → rolling 20 s buffer → /vision/save
```
- **Web app** (`web/`): `demo.html` is Hastaak. `index.html` is the playground and recorder, for debugging
  tracking and recording labelled clips. Vite proxies `/api` → 8000 and `/vision` → 8001.
- **Backend** (`server/`): holds the API key and calls the provider. It never sees video.
- **Native vision server** (`python/vision_server.py`): reads the webcam, recognizes signs, streams the annotated
  preview and events, and saves debugging recordings. Video stays on the laptop.
- **Python tools** (`python/`): feature extraction, training, evaluation, ONNX export, sign videos.

The two camera modes share one model file (`web/public/models/signnet.onnx`) and identical logic.
Only one of them may use the webcam at a time.

## Folder layout
```
CLAUDE.md                 # index (this project's rules + glossary)
README.md                 # short intro + quick start
docs/                     # this folder
server/
  main.py                 # FastAPI app: /api/health, /api/sentence, /api/listen; prompts; validation
  sarvam.py               # the provider: chat_json, transcribe, speak (only file that calls Sarvam)
  rules.py                # offline fallbacks: KEYWORDS, keyword_signs, template_sentence
  .env.example            # names of settings (copy to .env)
web/
  demo.html, index.html   # Hastaak page; playground/recorder page
  src/demo.ts, demo.css   # Hastaak logic (both camera modes, mic, clips, log)
  src/main.ts, style.css  # playground logic; style.css is shared
  src/features.ts         # feature spec v1 (twin of python/features.py)
  src/segmenter.ts        # segmenter + resample + low-fps settings (twin of python/segmenter.py)
  src/classifier.ts       # loads signnet.onnx/json, zeroes face values, returns top 3
  src/landmarkers.ts      # MediaPipe models (GPU/CPU delegate)
  src/actions.ts          # thumbs-up, finger states, eyebrow calibration
  scripts/                # copy-wasm.mjs (MediaPipe + ONNX WASM → public/), download-models.mjs
  public/models/          # MediaPipe .task files + signnet.onnx + signnet.json
  public/clips/           # one WebM per sign + clips.json
python/
  features.py             # feature spec v1 (twin of web/src/features.ts)
  segmenter.py            # Segmenter, cut_sign, hand_presence, low_fps_settings (twin of segmenter.ts)
  dataset.py              # load_seq, model_input, augment, SignDataset
  model.py                # SignNet
  train.py, evaluate.py, export_onnx.py
  include_manifest.py, extract_include.py, extract.py   # INCLUDE → features
  make_clips.py           # sign videos for the web app
  vision_server.py        # native camera mode
  live.py, live_classify.py, common.py                  # desktop tools; shared MediaPipe setup
  data/                   # include/ (videos), include_feats/ (.npy), own/ (your recordings) — not shared
  runs/<name>/            # best.pt, labels.json, confusions.csv
  recordings/<time>/      # debugging saves from native mode — contain video of people
```

## Key libraries
| Need | Library | Note |
| ---- | ------- | ---- |
| Landmarks | MediaPipe Tasks (`@mediapipe/tasks-vision`, `mediapipe`) | hands, pose lite, face + blendshapes |
| Classifier training | PyTorch | CPU is enough (minutes per run) |
| Classifier runtime | ONNX Runtime (`onnxruntime-web/wasm`, `onnxruntime`) | single-threaded WASM, served locally |
| Video I/O | OpenCV | webcam, clips (VP8 WebM), recordings |
| Backend | FastAPI + uvicorn + httpx | `python-dotenv` reads `server/.env` |
| Web | Vite + TypeScript (no framework) | `strict` TypeScript |

## Endpoints
| Endpoint | Body / reply | Notes |
| -------- | ------------ | ----- |
| `GET /api/health` | provider, `configured`, signs, showable signs, languages | the page's "Connected" pill |
| `POST /api/sentence` | `{signs: [{candidates, question}], language, speak}` → `{text, source, signs, audio?, error?}` | Deaf → hearing |
| `POST /api/listen` | multipart `file` (≤ 30 s audio), `language` → `{text, language, english, signs, glosses, source, error?}` | hearing → Deaf |
| `GET /vision/health` | `{running, labels}` | the page picks native mode when this answers |
| `POST /vision/start` · `/vision/stop` | `{running}` | opens / releases the webcam |
| `GET /vision/stream.mjpg` | MJPEG | annotated, mirrored preview (640 px) |
| `GET /vision/events` | server-sent events | `state`, `sign`, `thumb`, `fps`, `status`, `error` (data-formats.md) |
| `POST /vision/save` | `{folder, seconds, frames, fps}` | writes the last 20 s to `python/recordings/` |

CORS on the backend allows only `http://localhost:5173`. Both servers listen on localhost only.
