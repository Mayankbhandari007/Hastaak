# Local setup

## Prerequisites
- Windows 10/11 (also works on macOS/Linux with the usual path changes), a webcam and a microphone.
- Python 3.12 (`py -3.12`), Node.js 20+ and npm, Chrome (or another Chromium browser).
- A Meta Model API key (`MODEL_API_KEY`) for Muse Spark, Muse Voice Transcribe and SAM 3.1. Optional: without it
  the demo uses offline fallbacks, and the local Meta voice (MMS-TTS) still speaks.
- Disk space: the voices download ~150 MB per language on first use (cached in your user folder).
- Memory matters more than CPU. On an 8 GB laptop, close Docker/WSL, Discord and spare browser tabs before
  demoing (decisions.md D-014).

## First-time setup
```bash
# 1. Backend
cd server
py -3.12 -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu   # small CPU build, for the voices
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env            # then put your MODEL_API_KEY in .env

# 2. Web app
cd ..\web
npm install                       # also copies the MediaPipe + ONNX WASM files into public/
npm run models                    # downloads the 3 MediaPipe .task files (web/public/models + python/models)

# 3. Python tools + native camera
cd ..\python
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Run the demo (three terminals)
| Terminal | Folder | Command | Port |
| -------- | ------ | ------- | ---- |
| 1 Backend | `server` | `.venv\Scripts\python -m uvicorn main:app --reload --port 8000` | 8000 |
| 2 Web app | `web` | `npm run dev` | 5173 |
| 3 Native camera | `python` | `.venv\Scripts\python vision_server.py` | 8001 |

Open **http://localhost:5173/demo.html**. Check the top-right pill says **Connected: Meta**, and the
camera menu says **Native camera (faster)**. Then press **Start camera**. The playground is at http://localhost:5173/.
"In ISL" shows the **Avatar** by default; **Real signer** switches to the videos.

- Stop any server with **Ctrl+C** in its terminal.
- The backend reads `.env` only at start: restart it after changing the key or settings.
- Restart `vision_server.py` after changing Python recognition code or exporting a new model.
- `vision_server.py --camera 1` uses another webcam; `--video clip.mp4` replays a file (testing, demo fallback).
- `vision_server.py --signer-lock`: SAM 3.1 picks the signer when 2+ people are in view (slower tracking; uses
  credits only while 2+ people are visible). The status line shows `lock SAM` and the number of SAM calls.
- After exporting a new model, also run `python export_avatar.py` if the vocabulary changed.

## Activating a venv in PowerShell
VS Code may run this for you when a terminal opens:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned   # allow scripts in THIS terminal only
& .\python\.venv\Scripts\Activate.ps1                             # prompt shows (.venv)
```
Then `python` means the project's Python. Run the tools from inside `python\` (`cd python`). `deactivate` leaves it.

## Settings (`server/.env`)
| Name | Default | Notes |
| ---- | ------- | ----- |
| MODEL_API_KEY | — | Meta Model API key; never commit |
| META_CHAT_MODEL | muse-spark-1.3 | never a `-contributor` model with real users (their data trains Meta's models) |
| META_REASONING | minimal | Muse Spark always reasons; `minimal` is fastest and cheapest |
| META_STT_MODEL | muse-voice-transcribe-1.0 | accepts WAV only (the page converts) |
| META_SAM_MODEL | sam-3.1 | used by `vision_server.py --signer-lock` and `make_clips.py --sam` |

## Train on your own signing
1. In the playground, choose the sign next to **Record 3 s of features**, then record 5–10 examples per sign,
   plus NOTHING clips of ordinary movement (scratching, adjusting glasses, talking with your hands).
2. Move the downloaded `features-*.json` into `python/data/own/` (any subfolders).
3. Train, check and export:
   ```bash
   cd python
   python train.py --epochs 80 --seed 1 --name mine
   python evaluate.py runs/mine/best.pt
   python export_onnx.py runs/mine/best.pt
   ```
4. Restart `vision_server.py` and reload the page.

## Common problems
| Symptom | Cause / fix |
| ------- | ----------- |
| `can't open file '...\Meta\vision_server.py'` | run it from the `python` folder |
| `Could not find platform independent libraries <prefix>` | harmless venv quirk; ignore (or recreate the venv) |
| `W0000 ...`, `INFO: Created TensorFlow Lite XNNPACK delegate` | normal MediaPipe start-up messages; worry only about `E`/`F` lines or a Traceback |
| No voice; pill says "Server not running" or "offline" | start the backend; check `.env` has the key; restart it |
| First sentence takes ~1 minute | the Meta voices are downloading/loading; later sentences take ~3 s |
| Pill says "No API key" | add `MODEL_API_KEY` to `server/.env` and restart the backend |
| `signer lock: off` though you passed `--signer-lock` | no `MODEL_API_KEY` in `server/.env`, or 3 SAM calls failed in a row |
| Camera menu shows only "Camera in browser" | `vision_server.py` wasn't running when the page loaded: start it, reload |
| "Couldn't open camera 0" | another app (or the browser camera mode) is using the webcam |
| 1–2 fps | the laptop is out of memory: close other apps; use native mode |
| Signs split, merged or missed | pause with hands down between signs; move into static signs |
| Signs recognized as the wrong word | see design.md: record your own examples and retrain |
| `UnicodeEncodeError` printing "←" from `train.py` when piped | set `PYTHONIOENCODING=utf-8` |
