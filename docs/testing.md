# Testing

There is no automated test suite yet. These are the checks used so far, how to repeat them, and the last results.
Turning the scripted checks below into `python/tests/` (pytest) is an open task.

## Levels
| Level | How | Needs | When |
| ----- | --- | ----- | ---- |
| Type check | `npx tsc --noEmit` in `web/` | — | every web change |
| Model accuracy | `python evaluate.py runs/<run>/best.pt` | INCLUDE features | every training run |
| Live-style accuracy | each test clip through `cut_sign` → `model_input` → model | INCLUDE features | segmenter / feature / model changes |
| Robustness | the same with simulated dropouts, low fps or framing changes | INCLUDE features | segmenter changes; before claiming a fix |
| Browser parity | test clips through the browser's segmenter + classifier, compared with Python | dev server | any change to a twin (coding-standards.md) |
| Real user | native mode → **Save recording** → replay `features.json` offline | webcam, a person | any recognition change |
| Speech and language | Sarvam text-to-speech → `/api/listen` round trip with fixed phrases | API key | prompt or provider changes |

## Test data
- **INCLUDE test split:** 33 clips (3 per sign) from `data/include_manifest.csv`. The split is random per clip,
  so the same signers appear in training. Treat these numbers as an upper bound.
- **Real-user recordings:** `python/recordings/<time>/` (native mode). Note which signs were made, in order.

## Results so far (model `hastaak_v2`, 11 signs)
| Check | Result |
| ----- | ------ |
| INCLUDE test, live-style cuts | 32–33 / 33 |
| Hand lost in 2 bursts of 3–10 frames per sign | 91% |
| Hand lost in 4 bursts | 72% |
| 12.5 fps / 8.3 fps (subsampled clips, low-fps settings) | 32 / 33, 33 / 33 |
| Face values present (your real face values) vs zeroed | 30 / 33 vs 32 / 33 → face zeroed (D-015) |
| Signing 1.5 shoulder widths higher (seated close) | 5 / 33 |
| Browser vs Python | identical cuts on 33/33 clips; same answers; ONNX vs PyTorch < 3e-6 |
| Real user (native mode, 2026-10-30) | tracking 100%, but signs misread (HELLO → GOOD_AFTERNOON 70%) |
| `/api/listen`, 7 phrases (Hindi, English, Tamil, one with no sign) | 7 / 7 correct, 1.4–2.5 s |
| `/api/sentence`, hello + how are you in Hindi | "नमस्ते, आप कैसे हैं?", audio played, ~2.2 s |

## How to repeat the key checks
**Model accuracy:**
```bash
cd python
python evaluate.py runs/hastaak_v2/best.pt
```
**Live-style / robustness** (short scripts: load each test clip's `.npy`, change it, cut, predict):
```python
from dataset import model_input; from segmenter import cut_sign
seg = cut_sign(np.load(path))                    # live-style cut
x = model_input(seg)                             # what the model sees
```
Dropouts: zero `[:, :128]` for runs of 3–10 frames. Low fps: take every 2nd/3rd frame and use `low_fps_settings`.
Framing: subtract from all hand y values and the pose wrists.

**Replay a real recording:**
```python
X = np.array(json.load(open("recordings/<time>/features.json"))["frames"], np.float32)
# push frames through Segmenter() + 15 zero frames, model_input each segment, print top 3
```
**Browser parity:** write a fixture (raw test clips + Python's cut and prediction) into `web/public/__parity.json`,
open the dev server, import `/src/segmenter.ts` and `/src/classifier.ts` in the console, and compare. Delete the
fixture afterwards.

**Speech:** with the backend running, generate speech with `sarvam.speak(text, lang)` and POST it to
`/api/listen`. Phrases used: "नमस्ते, आप कैसे हैं?", "सुप्रभात! आपको कौन सा रंग पसंद है?",
"Good afternoon. I want the black one, not the white one.", "Where is the nearest bus stop?" (no signs),
"मुझे गुलाबी और नारंगी रंग अच्छे लगते हैं।", "Hi! Are you alright?", "வணக்கம், எப்படி இருக்கிறீர்கள்?".

## Before keeping a recognition change
1. INCLUDE live-style accuracy didn't drop.
2. A robustness check that targets the problem improved it.
3. A real-user recording, replayed, improved or stayed the same.
4. Browser parity still holds (if a twin changed).
5. decisions.md has an entry with the numbers.
