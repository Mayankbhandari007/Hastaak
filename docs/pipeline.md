# Pipeline

Read alongside data-formats.md (shapes) and prompts.md (the AI calls).

## 1. Live recognition (per frame, both camera modes)
| Step | Where | What happens |
| ---- | ----- | ------------ |
| 1 | webcam | 1280×720. Native mode keeps a 1-frame buffer, so it always processes the newest frame. |
| 2 | MediaPipe | hands every frame (up to 2); pose lite every 2nd frame; face every 3rd. The last result is reused in between. |
| 3 | features | 156 numbers (data-formats.md). The largest body is the signer. |
| 4 | eyebrows | the first 30 frames with a face set the neutral brow level (median). Later, above it by > 0.15 = raised. |
| 5 | thumbs-up | held ≥ 0.35 s → send the sentence (cool-down 1.5 s; the segmenter resets). |
| 6 | segmenter | see below → a finished segment (one sign). |
| 7 | checks | hand seen in < 60% of the segment → "hand went out of view" + Did-you-mean buttons. |
| 8 | SignNet | `model_input` (32 frames, face zeroed) → top 3. Top is NOTHING or p < 0.35 → Did-you-mean buttons. |
| 9 | sentence | accepted signs collect as chips. Question = brows above baseline in the top 20% of the sign's frames. |

### The segmenter
States REST and ACTIVE. Settings (tuned at 25 fps): `rest_y` 1.8, `start_motion` 0.05, `stop_motion` 0.015,
`stop_frames` 8, `min_len` 8, `max_len` 90, `hold` 4.
- **Hands up:** a hand slot is present with its wrist above `rest_y` (1.8 shoulder widths below the shoulders).
- **Motion:** wrist movement since the previous frame, counted only for hands present in both frames.
- **Start:** hands up and motion > `start_motion`.
- **End:** `stop_frames` frames in a row that are still or have the hands down. The trailing still frames are dropped;
  a sign shorter than `min_len` is discarded.
- **Bridging:** a hand that vanishes while raised keeps its last values for up to `hold` frames. Bridged-only
  frames don't count as "still". A hand that vanishes while lowered ends the sign at once.
- **Low frame rates** (`low_fps_settings` / `segmenterForFps`, applied every second while REST): wait the same
  *time* before ending a sign (at least 5 frames), and accept the same minimum *duration*. Motion thresholds are not scaled.

## 2. Deaf → hearing (`POST /api/sentence`)
1. The page sends every accepted sign with its top-3 candidates and question flag, plus the listener's language.
2. Template fallback first: the top glosses joined into a sentence ("Hello how are you?").
3. Muse Spark 1.3 (`SENTENCE_PROMPT`) picks the most plausible candidate per
   sign and writes one sentence in that language. A reply under 400 characters replaces the template.
4. The voice: MMS-TTS on this laptop, in the listener's language (Latin-letter text uses the English voice) →
   base64 WAV → played in the page.
5. Any failure: the template text, and `error` in the reply. MMS-TTS still speaks it without a key.

## 3. Hearing → Deaf (`POST /api/listen`)
1. The page records up to 25 s, converts it to 16 kHz mono WAV (`wav.ts`) and uploads it.
2. Muse Voice Transcribe (language auto-detected) → caption in the spoken language.
3. Keyword signs: English and Hindi words for each sign (`rules.KEYWORDS`) found in the caption, in spoken order.
4. Muse Spark (`PLAN_PROMPT`, with each sign's meanings) → English translation + sign list.
   If keywords matched, the model's signs must be among them (stops invented greetings); otherwise its list is used.
5. The page shows the caption, the English line and the sign chips, then signs them: the **avatar** (default)
   replays each sign's motion with smooth blends between signs, or **Real signer** plays `/clips/<SIGN>.webm`.

## 4. Training (offline, from `python/`)
| Step | Command | Output |
| ---- | ------- | ------ |
| 1 | put INCLUDE videos in `data/include/<Category>/<NN. Sign>/` | |
| 2 | `python include_manifest.py` | `data/include_manifest.csv` (random 70/15/15 split per clip) |
| 3 | `python extract_include.py` | `data/include_feats/…/*.npy` (frames × 156) |
| 4 | (optional) your recordings → `data/own/` | labelled JSON (data-formats.md) |
| 5 | `python train.py --epochs 80 --seed 1 --name <run>` | `runs/<run>/best.pt` (best validation epoch) |
| 6 | `python evaluate.py runs/<run>/best.pt` | test accuracy, weakest signs, confusions |
| 7 | `python export_onnx.py runs/<run>/best.pt` | `web/public/models/signnet.onnx` + `.json` |
| 8 | `python make_clips.py` | `web/public/clips/*.webm` + `clips.json` |

**Another folder of INCLUDE-style videos** (e.g. `E:\12\Final_Dataset`): pass the paths instead of the defaults.
Features are found by the video's path relative to `--root`, so a folder that was already extracted elsewhere
(moved, not re-encoded) reuses its `.npy` files. `--no-own` leaves out `data/own/` so the model sees only that folder.
```bash
python include_manifest.py --root E:/12/Final_Dataset --out data/final_manifest.csv
python extract_include.py --manifest data/final_manifest.csv --videos E:/12/Final_Dataset --feats data/include_data_feats
python train.py --manifest data/final_manifest.csv --feats data/include_data_feats --no-own --name hastaak_final47
python evaluate.py runs/hastaak_final47/best.pt --manifest data/final_manifest.csv --feats data/include_data_feats
```
The sign name comes from the nearest `<number>. <sign>` folder, so clips in a subfolder such as `5. Beautiful/Extra/`
count as BEAUTIFUL.

**Webcam-style variants** (`synth_variants.py`, Greetings and Pronouns only by default). The INCLUDE videos are
1080p, well lit and framed full-body; a laptop webcam is low-res, dim, tilted, close and blurry. Each train or test
clip gets 4 re-rendered copies (a random mix of 360p/540p, close framing from the pose's shoulders, tilt, dim /
bright / side light, colour cast, defocus or motion blur, sensor noise, JPEG), written to `E:\12\Synthetic_GP`.
Variants keep their source clip's split: train copies become extra training data, test copies become a
robustness test, and val clips get none. `--finish` drops any variant whose live cut has hand presence below
`MIN_HAND_PRESENCE` (the live app would reject it too). Every step skips work already done, so a stopped run resumes.
```bash
python synth_variants.py --workers 2                     # ~23 min for 1,228 videos
python extract_include.py --manifest data/synth_manifest.csv --videos E:/12 --feats data/include_data_feats --workers 2   # ~3.5 h
python synth_variants.py --finish                        # QC → data/final_plus_synth_manifest.csv, data/synth_test_manifest.csv
python train.py --manifest data/final_plus_synth_manifest.csv --feats data/include_data_feats --no-own --name hastaak_final47_synth
python compare_models.py runs/hastaak_final47/best.pt runs/hastaak_final47_synth/best.pt
```

Training cuts every clip with `cut_sign` (falling back to `trim_active` for the ~1% where no sign is found).
Augmentation: 50% mirror (left ↔ right signer), crop ≤ 10% at each end, ±10° rotation, ±10% scale, small noise,
5% dropped hand frames. AdamW, lr 1e-3, label smoothing 0.1, cosine schedule, batch 64. Takes minutes on a CPU.

## 5. Sign videos and avatar motion
`make_clips.py` picks, per sign, the clip whose sign length is closest to the median. It trims to the sign ± 6
frames, crops the upper body around the shoulders (MediaPipe pose), and writes 480 px VP8 WebM (~1 MB each).
With `--sam`, SAM 3.1 first finds the signer in the whole original clip ("person"; the person with the most mask
pixels), and every frame is composited onto one plain background with a soft edge.

`export_avatar.py` uses the same clip choice. It takes the sign ± 6 frames of the extracted features, fills hand
gaps by linear interpolation (holding the ends), holds the last body pose over frames without one, and smooths
with a 3-frame moving average. Output: `web/public/avatar/<SIGN>.json`.

## 6. Signer lock (`vision_server.py --signer-lock`)
MediaPipe tracks up to 3 people and 4 hands. When 2+ people are in view, at most every 3 s, one 640 px frame goes
to SAM 3.1 ("person") in the background. The box with the best area × centrality is the signer for 10 s; the
signer is the pose whose shoulder midpoint lies in it (the largest one if several do). Hands are matched to that
pose's wrists as usual, so a bystander's hands are ignored. The box is drawn on the preview.
