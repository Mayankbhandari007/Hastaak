# Data formats

Read alongside pipeline.md. These shapes are shared by Python, the browser and the servers.

## Feature spec v1: 156 numbers per frame
| Index | Content |
| ----- | ------- |
| 0 | slot A present (1/0): the hand nearest the person's **left** wrist (pose landmark 15) |
| 1–63 | slot A: 21 hand landmarks × (x, y, z) |
| 64 | slot B present: the hand nearest the person's **right** wrist (pose 16) |
| 65–127 | slot B: 21 × (x, y, z) |
| 128–148 | 7 body points × (x, y, z): nose, L shoulder, R shoulder, L elbow, R elbow, L wrist, R wrist |
| 149–155 | 7 face blendshapes: browInnerUp, browOuterUpLeft, browOuterUpRight, browDownLeft, browDownRight, jawOpen, mouthPucker |

Rules:
- x and z are multiplied by the frame's aspect ratio (width / height).
- The origin is the shoulder midpoint; the unit is the shoulder width. y points **down** (positive = below the shoulders).
- Hands are matched to the nearest pose wrist (closest pair first, at most 0.8 shoulder widths away).
  MediaPipe's left/right labels are not used.
- The face is the one whose nose tip is nearest the body's nose.
- No hand in a slot: presence 0, zeros. No body: the whole frame is zeros.
- **Model input** (`dataset.model_input`, `classifier.ts`): a segment resampled to 32 frames by linear
  interpolation, with 149–155 set to 0.

## Vocabulary (current model)
`ALRIGHT, BLACK, COLOUR, GOOD_AFTERNOON, GOOD_MORNING, GREY, HELLO, HOW_ARE_YOU, ORANGE, PINK, WHITE`
(model `python/runs/hastaak_v2`). `NOTHING` is added only when you have recordings labelled NOTHING.

## Model files (`web/public/models/`)
- `signnet.onnx`: input `x` [1, 32, 156] float32, output `logits` [1, number of signs]. One file, no `.data` file.
- `signnet.json`: `{"labels": [...in output order...], "featureSpecVersion": 1}`.
- `runs/<name>/best.pt` (PyTorch): `{"model": state_dict, "labels": [...], "feature_spec_version": 1}`.

## Recordings
**Playground "Record 3 s"** (`features-<LABEL>-<time>.json`, downloaded by the browser):
```json
{"featureSpecVersion": 1, "label": "HELLO", "width": 1280, "height": 720, "frameCount": 90, "frames": [[156 numbers], ...]}
```
Put labelled files anywhere under `python/data/own/` to train on them. The label comes from `label`, or else
from the folder name. The split (train/val/test = 70/15/15) is fixed by a hash of the file name.

**Native "Save recording"** (`python/recordings/<YYYYMMDD-HHMMSS>/`):
| File | Content |
| ---- | ------- |
| `preview.webm` | the last ~20 s as shown: mirrored, with body, face and hands drawn and the status line |
| `features.json` | as above, plus `fps` and `times` (seconds per frame); `label` is null |
| `events.json` | the `state`, `sign`, `thumb`, `status` events in that window, `t` in seconds from the start |

## Sign videos (`web/public/clips/`)
`<SIGN>.webm` (VP8, 480 px high, cropped to the upper body, trimmed to the sign) and `clips.json`
(`{"HELLO": "HELLO.webm", ...}`). The backend only plans signs that are in `clips.json`.

## Vision events (`GET /vision/events`, one JSON per message)
| type | Fields |
| ---- | ------ |
| `state` | `state`: `ACTIVE` (signing), `REST`, `NOBODY` |
| `sign` | `candidates` [{label, p}] × 3, `question` (bool), `presence` (share of frames with a hand), `frames` |
| `thumb` | thumbs-up held 0.35 s: send the sentence |
| `fps` | `fps`, `ms` {pose, hands, face} |
| `status` / `error` | `text` |
Every event also has `id` (increasing) and `t` (Unix time).

## Backend bodies
`POST /api/sentence`:
```json
{"signs": [{"candidates": [{"label": "HELLO", "p": 0.82}, {"label": "WHITE", "p": 0.1}], "question": false}],
 "language": "hi-IN", "speak": true}
```
Reply: `{"text": "नमस्ते", "source": "sarvam" | "template", "signs": [...], "audio": "<base64 WAV>", "error": "..."}`.

`POST /api/listen` (multipart `file`, `language` = `unknown` to auto-detect). Reply:
`{"text", "language", "english", "signs": ["HELLO"], "glosses": ["hello"], "source": "sarvam" | "keywords" | "none", "error"}`.

Languages: `en-IN, hi-IN, bn-IN, ta-IN, te-IN, mr-IN, gu-IN, kn-IN, ml-IN, pa-IN, od-IN`.
