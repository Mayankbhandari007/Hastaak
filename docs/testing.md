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
| Speech and language | MMS-TTS speech → `/api/listen` round trip with fixed phrases | API key | prompt or model changes |

## Test data
- **INCLUDE test split:** 33 clips (3 per sign) from `data/include_manifest.csv`. The split is random per clip,
  so the same signers appear in training. Treat these numbers as an upper bound.
- **Real-user recordings:** `python/recordings/<time>/` (native mode). Note which signs were made, in order.

## Earlier results (model `signbridge_v2`, 11 signs)
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
| MMS-TTS (Meta voice, local) transcribed back | Hindi and English word for word, Tamil one letter off; ~3 s per sentence |
| Native server PyTorch vs browser ONNX | same top answer 33/33, probabilities within 7e-7 |
| Page WAV conversion (`wav.ts`) | 16 kHz mono 16-bit WAV, the format Muse Voice Transcribe accepts |
| Tracking 3 people + 4 hands (signer lock) | pose 39 → 79 ms, hands 69 → 100 ms per frame; ~5 fps → opt-in (D-023) |
| Avatar | every sign exported and drawn; HELLO's salute and fingers visible |
| Sapiens2 0.4B vs MediaPipe (CPU, 2 resting frames) | 127 s vs 0.5 s per frame; fingertips 20/20 within 0.25 hand sizes |

## Results: Meta speech and language (2026-11-04)
The seven test phrases spoken by MMS-TTS (`meta.speak`), sent to `/api/listen` (Muse Voice Transcribe → Muse
Spark sign plan), plus one `/api/sentence` call. All replies came from Meta (`source: meta`), no fallbacks.
| Phrase | Heard | Signs | Right? |
| ------ | ----- | ----- | ------ |
| नमस्ते, आप कैसे हैं? | same | HELLO, HOW_ARE_YOU | yes |
| सुप्रभात! आपको कौन सा रंग पसंद है? | "…कौन सा रन रन…" (रंग misheard) | GOOD_MORNING | COLOUR missed |
| Good afternoon. I want the black one, not the white one. | same | GOOD_AFTERNOON, I, BLACK, WHITE | yes |
| Where is the nearest bus stop? | "…nearest Bastard?" | none | yes (no sign) |
| मुझे गुलाबी और नारंगी रंग अच्छे लगते हैं। | same | PINK, ORANGE, COLOUR | yes |
| Hi! Are you alright? | "IR you alright?" | YOU, ALRIGHT | HELLO missed |
| வணக்கம், எப்படி இருக்கிறீர்கள்? | same | HELLO, HOW_ARE_YOU | yes |

- Sign plans right on 5 of 7. Both misses were mishearings of the synthetic MMS voice; check with a real voice.
- `/api/listen` took 6.8–10.8 s per phrase on this laptop; `/api/sentence` (hello + how are you, Hindi, spoken):
  "नमस्ते, आप कैसे हैं?" with audio in 7.0 s.
- SAM 3.1 (signer lock, `make_clips.py --sam`) not re-measured here.

## Results: model `hastaak_final47_synth` (47 signs + webcam-style variants, 2026-11-04)
Same 734 clips and split as `hastaak_final47`, plus 783 webcam-style variants of the Greetings/Pronouns training
clips (`synth_variants.py`; 1,228 made, 937 kept by the hand-presence check, 291 dropped, mostly 360p and dim).
The robustness test is 154 kept variants of the test clips: the model never saw these clips in any form.
| Check | `hastaak_final47` | `hastaak_final47_synth` |
| ----- | ----------------- | ----------------------- |
| Clean test, all (103) | 95.1% | **97.1%** |
| Clean test: Adjectives / Greetings / Pronouns | 96.2 / 96.3 / 91.7% | 96.2 / **100** / **95.8%** |
| Webcam variants of test, all (154) | 53.9% | **85.7%** |
| Webcam variants: Greetings (92) / Pronouns (62) | 53.3 / 54.8% | **89.1 / 80.6%** |
| ONNX vs PyTorch, 103 test clips | | same top answer 103/103, probabilities within 9.5e-7 |
| Real user | not measured | **not yet measured** |

The webcam variants are synthetic, so they show the direction, not a real-world number. Adjectives got no variants
and did not drop. Deployed 2026-11-04 so it can be tried on a real webcam; the real-user check (rule 6) is still
owed. If that recording shows it worse, revert with `python export_onnx.py runs/hastaak_final47/best.pt`.

## Results: model `hastaak_final47` (47 signs, `E:\12\Final_Dataset` only, 2026-11-04)
734 clips (Adjectives, Greetings, Pronouns), split 528 / 103 / 103, no own recordings. Live-style cuts throughout.
| Check | Result |
| ----- | ------ |
| Validation (best epoch 67 of 80) | 98.1% |
| Test, 103 clips | top-1 95.1%, top-3 98.1%; wrong: FLAT→RICH, HELLO→THICK, UGLY→YOU_(PLURAL), YOU→I, YOU_(PLURAL)→IT |
| 20 test clips neither model trained on | new 18/20; previous 114-sign model (`include_data_run`) 17/20, also 17/20 limited to these 47 signs |
| Browser (ONNX) vs PyTorch, 103 test clips | same top answer 103/103, probabilities within 1.1e-6 |
| Real user | **not yet measured**: no recordings in `python/recordings/` |

The previous model's manifest split the same clips differently: 83 of these 103 test clips were in its training
set, so only the 20 above give a fair comparison. 20 clips is too few to call either model better.

## How to repeat the key checks
**Model accuracy:**
```bash
cd python
python evaluate.py runs/signbridge_v2/best.pt
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

**Speech:** with the backend running, generate speech with `meta.speak(text, lang)` (MMS-TTS) and POST it to
`/api/listen`. Phrases used: "नमस्ते, आप कैसे हैं?", "सुप्रभात! आपको कौन सा रंग पसंद है?",
"Good afternoon. I want the black one, not the white one.", "Where is the nearest bus stop?" (no signs),
"मुझे गुलाबी और नारंगी रंग अच्छे लगते हैं।", "Hi! Are you alright?", "வணக்கம், எப்படி இருக்கிறீர்கள்?".

## Before keeping a recognition change
1. INCLUDE live-style accuracy didn't drop.
2. A robustness check that targets the problem improved it.
3. A real-user recording, replayed, improved or stayed the same.
4. Browser parity still holds (if a twin changed).
5. decisions.md has an entry with the numbers.
