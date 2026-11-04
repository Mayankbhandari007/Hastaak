# Coding standards

Read alongside architecture.md and data-formats.md. Root CLAUDE.md is the index that points here.

## Principles (non-negotiable)
1. Simplest thing that works. No abstraction until there is a second use.
2. **Twins stay identical.** These pairs must give the same result for the same input:

   | Python | TypeScript |
   | ------ | ---------- |
   | `python/features.py` | `web/src/features.ts` |
   | `python/segmenter.py` (`Segmenter`, `hand_presence`, `low_fps_settings`) | `web/src/segmenter.ts` (`Segmenter`, `handPresence`, `segmenterForFps`) |
   | `python/dataset.py` (`resample`, `model_input`: face zeroed) | `web/src/segmenter.ts` (`resample`) + `web/src/classifier.ts` (face zeroed) |

   Change one, change the other in the same change, and run the parity check (testing.md).
3. **Train what you run.** Training clips are cut by `segmenter.cut_sign`, the same segmenter the live demo
   uses. Never trim training clips by any other rule.
4. **The feature spec is a contract.** `FEATURE_SPEC_VERSION` (currently 1) is written into `signnet.json` by
   `export_onnx.py`. The browser refuses a model with another version. Changing the layout means a new version,
   re-extraction and retraining.
5. **One Meta module.** Only `server/meta.py` calls Meta. The backend uses its `chat_json`, `transcribe` and
   `speak`; the Python tools use its `find_people` and `person_masks` (SAM 3.1).
6. **No secret in code, logs, the browser or git.** The API key lives only in `server/.env`.
7. **Every AI step has a fallback.** If a Meta call fails, the backend answers with the rules in
   `server/rules.py` and says so (`source`, `error`). The page never hangs on a failed call.
8. **Measure, then keep or revert.** A change meant to improve recognition is kept only if it measurably helps
   (testing.md). Record it in decisions.md either way. No experimental code is left switched off "for later".
9. Constants with a reason get a comment with that reason and the evidence (e.g. `MIN_HAND_PRESENCE = 0.6`:
   real INCLUDE signs have ≥ 81%).

## Python
- Python 3.12, type hints where they help, plain functions, small files.
- Run tools from inside `python/`: paths like `data/`, `runs/`, `models/` are relative to it.
- NumPy arrays are `float32`. A frame with no signer is all zeros. An absent hand is presence 0 with zero coordinates.
- MediaPipe in `VIDEO` mode needs strictly increasing timestamps. Always pass a monotonic, increasing `ts`.
- Raise specific errors with a readable message. Never `except Exception: pass`.
  (The backend's `except Exception` blocks are deliberate: they switch to a fallback and return the error text.)

## TypeScript (web/)
- `strict` TypeScript, no framework. One page = one `.ts` entry (`demo.ts`, `main.ts`). Shared logic lives in
  `features.ts`, `segmenter.ts`, `classifier.ts`, `landmarkers.ts`, `actions.ts`.
- All runtime files are served locally (`public/wasm`, `public/ort`, `public/models`). No CDN, so the demo works offline.
- Anything called every frame must be cheap: no DOM writes unless the value changed (see `banner()`).
- `npx tsc --noEmit` must pass before a change is done.

## Backend (server/)
- Prompts are constants in `server/main.py` (prompts.md). Model replies are JSON. Every field is validated
  (signs must exist in the vocabulary) before use.
- Pydantic models for request bodies. Validate the language against `LANGUAGES`.
- Timeouts on every Meta call (chat 20 s, speech 30 s).

## Models and data
- A training run goes to `python/runs/<name>/`. Never overwrite a run you have exported; use a new name.
- Export the chosen run with `export_onnx.py`. It checks the ONNX output against PyTorch (difference < 1e-3).
- After changing the vocabulary, re-run `make_clips.py` so every sign the model knows has a video.

## Git and sharing
- Never commit or share: `server/.env`, `python/data/` (INCLUDE videos + your recordings), `python/recordings/`,
  `python/runs/`, `node_modules/`, `.venv/`. `.gitignore` covers all of these; check `git status` before each commit.
- Committed on purpose: the exported model (`web/public/models/signnet.onnx` + `.json`) and the sign videos
  (`web/public/clips/`), because the demo needs them and they can't be rebuilt without the dataset.
  The MediaPipe `.task` and WASM files are not committed: `npm install` and `npm run models` fetch them.
- Commit messages: `<area>: <what>` (e.g. `segmenter: bridge short hand gaps`).
