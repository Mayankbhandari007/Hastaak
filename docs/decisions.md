# Decision log

Newest first. One entry per decision: context, decision, consequence. Never delete an entry; add a new one that
supersedes it. Numbers in entries come from the checks in testing.md.

## D-030 The Hastaak page fits one screen, with no page scroll (2026-11-05)
Across a table, nobody scrolls: the camera, the signed sentence, "They said" and the avatar must all be in view
at 1440×900 and 1920×1080. The page is exactly the viewport (`demo.css`): two equal panels, and only the text areas
inside them scroll. The status banner is a pill over the bottom of the camera picture. "They said" keeps every
turn, with the newest highlighted. The tap-to-add vocabulary and the conversation history moved into popovers.
The avatar canvas is portrait (420×480, same figure size: the motion never leaves its middle 62%). Below 1024px
the panels stack, and **Flip** turns the hearing panel 180° to face the person opposite. Phones (<640px) scroll.

## D-029 Train Greetings/Pronouns on webcam-style video variants, without masks (2026-11-04)
The model scored 95% on INCLUDE's studio clips but failed on real webcams, and on webcam-style copies of its own
test clips it scored 53.9%. The planned approach (SAM 3.1 person masks, then new backgrounds) cost Meta credits and
only changed the background, which the landmark model barely sees. `synth_variants.py` instead re-renders whole
frames (resolution, framing, light, colour, blur, noise, compression), which changes what MediaPipe tracks, and needs
no API. Variants are re-extracted with the same pipeline and cut by `cut_sign` (rule 3), inherit their source's
split, and are dropped if the live hand-presence check would reject them. `hastaak_final47_synth`: clean test 97.1%
(was 95.1%), webcam variants 85.7% (was 53.9%), Adjectives unchanged (testing.md). Deployed 2026-11-04 (ONNX
matches PyTorch 103/103) so it can be tried live; a real-user recording must still confirm it (rule 6), else revert
with `python export_onnx.py runs/hastaak_final47/best.pt`. `synth_masks.py` is not used.

## D-028 Signs are named by their numbered folder; a 47-sign model trained on Final_Dataset only (2026-11-04)
`include_manifest.py` named a sign after a clip's immediate folder, so three clips in `…/extra/` subfolders
became a fake sign EXTRA in the 114-sign model (`include_data_run`). It now uses the nearest `<number>. <sign>`
folder. `train.py --no-own` trains on the manifest alone. `runs/hastaak_final47` is trained only on
`E:\12\Final_Dataset` (47 signs, 734 clips), reusing the features already extracted from those files: test 95.1%,
and 18/20 vs 17/20 for the 114-sign model on the clips neither saw (testing.md). It is now the deployed model
(exported 2026-11-04; browser matches PyTorch on all 103 test clips). Still to do: a real-user recording (rule 6).
To go back: `python export_onnx.py runs/include_data_run/best.pt`.

## D-027 The product is called Hastaak; logos live in web/public/brand (2026-11-04)
SignBridge is renamed Hastaak everywhere a person reads the name (pages, tab titles, README, docs, API titles).
The logo files were traced in two slightly different navies and oranges; they are unified to `#123B78` and
`#FF9F0A`. The navy is nearly invisible on the dark UI (`--bg #0e1113`), so the pages use `*-light.svg` (white
with orange marks); the favicon sits on a white tile so it reads on light and dark tab bars. The README switches
between the two with `prefers-color-scheme`. Model run folders (`runs/signbridge_v2`) and the GitHub repository
keep their old names: renaming them would break paths and links for no user-visible gain.

## D-026 Sapiens2 is evaluated before any switch of tracker (2026-11-02)
The Meta Model API has no model that gives finger joints; Sapiens2 (open weights, 308 keypoints incl. both hands)
is Meta's option. It needs a new feature layout, re-extraction and retraining, and a GPU for live use.
- `python/sapiens_check.py` runs Sapiens2 and MediaPipe on the same frames and reports speed, how many MediaPipe
  fingertips have a Sapiens2 keypoint within 0.25 hand sizes, and side-by-side pictures. Switch only if fingers
  are as good. MediaPipe stays meanwhile (ask the organisers whether it is allowed).
- First run (0.4B, this laptop's CPU, 2 frames of INCLUDE HELLO):
  - **~127 s per frame** (MediaPipe ~0.5 s). Far too slow for live use without a GPU.
  - 20/20 MediaPipe fingertips had a Sapiens2 keypoint nearby (median 0.07 hand sizes). Both frames had the
    hands resting, so mid-sign hand shapes are still untested; run `--frames 8` on a GPU.
  - Sapiens2 found the **face in detail at INCLUDE's distance**, where MediaPipe finds none: a possible route to
    facial-expression training data.

## D-025 The signing animation is a 2D avatar replaying recorded motion (2026-11-02)
The problem statement asks for an on-screen signing avatar. SAM 3D Body would need a GPU and days of retargeting.
- `export_avatar.py` exports each sign's motion from the INCLUDE features (same clip as the video, gaps
  interpolated, smoothed); `web/src/avatar.ts` draws head, torso, arms and all finger joints, blending between
  signs. The page offers **Avatar / Real signer**. No credits, no GPU.
- No facial expression: INCLUDE has no face data (D-015). A close-up recording of a signer would add it.

## D-024 SAM 3.1 cuts signers out of the sign videos (2026-11-02)
`make_clips.py --sam` sends each sign's original clip (MOV, H.264) to SAM 3.1 ("person") and places the signer on
one plain background, so signs from different recordings look like one signer. ~770 frames ≈ $0.15. Not yet run:
needs the key.

## D-023 SAM 3.1 signer lock is opt-in (2026-11-02)
Seeing bystanders needs MediaPipe to track up to 3 people and 4 hands. Measured on this laptop: pose 39 → 79 ms,
hands 69 → 100 ms per frame, which dropped the native server from 7–9 to ~5 fps.
- `vision_server.py --signer-lock` turns it on. SAM 3.1 is called only when 2+ people are in view, at most every
  3 s, and picks the largest, most central person; their pose is the signer's for 10 s. One person alone costs
  nothing. Failures fall back to "largest person"; 3 failures in a row turn the lock off.
- Default off, so normal use keeps the faster single-person tracking.

## D-022 The native server runs SignNet with PyTorch (2026-11-02)
`export_onnx.py` also writes `signnet.pt` next to the ONNX file; `vision_server.py` loads it with PyTorch (Meta)
instead of ONNX Runtime (Microsoft). Same top answer on 33/33 test clips, probabilities within 7e-7.

## D-021 The page uploads speech as 16 kHz mono WAV (2026-11-02)
Muse Voice Transcribe accepts only WAV (16-bit PCM, 16 or 24 kHz). `web/src/wav.ts` decodes the browser's
WebM/Opus recording and resamples it to 16 kHz WAV.

## D-020 Language and speech use Meta: Muse Spark 1.3, Muse Voice Transcribe, MMS-TTS (2026-11-02)
`server/meta.py` implements the three functions the backend calls:
- `chat_json`: Muse Spark 1.3 with `reasoning_effort: "minimal"`. It can't be turned off; reasoning tokens count
  toward `max_completion_tokens`, so that limit is 2000.
- `transcribe`: Muse Voice Transcribe (`/v1/asr/transcribe`, multipart `request` JSON + `audio` WAV).
- `speak`: MMS-TTS, Meta's open-weights voices run locally, because the Model API has no text-to-speech.

Checks so far:
- MMS-TTS audio transcribed back word for word in Hindi and English (Tamil one letter off).
- About 3 s per sentence on this laptop. Hindi and English voices load at start-up.
- Voices read only their own script, so Latin-letter text (the offline template) uses the English voice.
- Muse Spark + Voice Transcribe tested 2026-11-04: sign plans right on 5 of 7 test phrases (both misses were
  mishearings of the synthetic test voice), 7-11 s per reply (testing.md).

## D-019 Rejected or unsure signs become "Did you mean" buttons (2026-10-29)
Signs with the hand mostly out of view, a NOTHING result or confidence < 0.35 were dropped with a message, and the
user had to sign again, often several times on a slow laptop.
- The top two real candidates are shown as buttons; one tap adds the sign. Tap-to-add from the full vocabulary
  stays as the last fallback.
- Nothing wrong is added silently: an accepted sign still needs p ≥ 0.35 and the hand seen in ≥ 60% of frames.

## D-018 Native mode can save the last 20 seconds for debugging (2026-10-30)
Reports like "accuracy is low" couldn't be diagnosed without seeing what the camera and tracker saw.
- `POST /vision/save` writes `preview.webm` (with body, face and hands drawn, plus a status line),
  `features.json` (the exact model input frames) and `events.json` to `python/recordings/<time>/`.
- The preview now draws pose and face too, not only hands, which earlier looked like "not tracked".
- Recordings contain video of people: security.md.

## D-017 No training on the full INCLUDE dataset; own recordings instead (2026-10-30)
A real user's signs, done as INCLUDE does them, were still misread. The question was whether all 263 INCLUDE signs
would help.
- No: the full set has the same signers and studio framing, so it adds more of what the model already learned,
  not people like the user. It is about 4,300 videos (tens of GB) and hours of extraction on this laptop, and 252
  more classes to tell apart.
- The fix is 5–10 recordings per sign of the actual user, in the demo position (`data/own/`). A "teach mode" in the
  demo to record them is the next step.

## D-016 Hiding the resting hand in training: tried, reverted (2026-10-30)
Seated close to a laptop, the non-signing hand is below the frame; INCLUDE signers stand with it visible.
- Removing the lower hand from every test clip roughly halved accuracy, but that test also removed hands taking
  part in two-handed signs. Removing only hands below the rest line affected 37 of 225 clips and cost nothing.
- Training with that augmentation scored 33/33 on INCLUDE but did not fix the user's recordings, so it was removed.
- Also ruled out as the cause of misreading the user: left-handed signing (INCLUDE HELLO mirrored still scores
  90%) and larger-looking hands (30/33 vs 32/33). Signing height matters most: 1.5 shoulder widths higher → 5/33.

## D-015 The model never sees face values (2026-10-30)
INCLUDE signers stand too far away for face tracking: face data in 1 of 14,170 frames. A user at a laptop is
tracked, so the 7 face values were input the model had never seen. With a real user's face values, INCLUDE test
accuracy fell from 32/33 to 30/33.
- `dataset.model_input` (Python) and `classifier.ts` zero features 149–155 before the model. The face is still
  tracked for question detection.
- Facial expressions can't help recognition until training data includes faces (own recordings).

## D-014 Native camera mode: recognition in Python, page as display (2026-10-30)
In the user's browser the demo ran at 1–2 fps. The laptop (i3-10110U, 7.8 GB RAM) had ~700 MB free and Windows
spent a third of the CPU compressing memory. The earlier version that "worked better" was the Python
`live_classify.py` window. MediaPipe natively runs at 7–9 fps on the same machine under the same load.
- `python/vision_server.py` reads the webcam (1-frame buffer), runs tracking, the segmenter and SignNet, and
  streams the preview (MJPEG) and events (SSE). The page picks it automatically when it is running.
- Browser mode stays as a fallback. Both modes share the model file and the twin logic.
- `--video` replays a file, for tests without a camera and as a demo fallback.

## D-013 Segmenter timing follows the frame rate (2026-10-30)
Thresholds counted in frames were tuned for 25 fps. At 7–9 fps, "8 still frames to end a sign" was ~1 s, and
"8 frames minimum" threw away quick signs (WHITE was never recognized).
- Below 25 fps, the end-wait and minimum length are scaled to the same *time* (`low_fps_settings` /
  `segmenterForFps`). With that, 8.3 fps went from 30/33 to 33/33, and 6.25 fps from 28/33 to 31/33.
- Motion thresholds are not scaled: scaling them too lost signs (25/33 at 8.3 fps).
- At least 5 still frames end a sign. 3 split a real user's HELLO at its pause by the forehead; no cost on INCLUDE.
- Thumbs-up is time-based (0.35 s), not 10 frames.

## D-012 Hands every frame; pose every 2nd; face every 3rd (2026-10-30)
Measured on this laptop (GPU path): hands 32 ms, pose 41 ms, face 16 ms per frame. Body and eyebrows change slowly.
- Pose and face results are reused between runs. This takes ~89 ms to ~55–60 ms per frame. Same schedule in
  both camera modes.

## D-011 Training with simulated camera dropouts: tried, removed (2026-10-29)
Hands vanishing for 3–10 frames in training (then bridged) gave 91–94% clean and 69–71% with 4 dropout bursts, the
same as without it (91–100% and 67–73%). Removed rather than kept unused.

## D-010 Short hand-tracking gaps are bridged (2026-10-29)
Laptop tracking flickers, and every lost frame counted as "hands down", cutting signs short.
- A hand lost while **raised** keeps its last values for up to 4 frames. Bridged-only frames don't count as still.
  A hand lost while lowered ends the sign at once (no delay).
- Same in Python and the browser; training cuts use it too.
- Bug found and fixed: the training cutter added too few rest frames to close a bridged sign, so most training
  clips silently fell back to the old trimming. Now only 2 of 227 do.

## D-009 No synthetic "no sign" class (2026-10-29)
A NOTHING class trained on real signs with the hands removed called flickery real signs "no sign": 43–51 of 99 test
signs under dropout bursts, accuracy 42–48%. Also using "hands rising" clips cost ~6 points on clean signs.
- NOTHING now exists only if there are recordings labelled NOTHING in `data/own/`.

## D-008 Signs with the hand mostly out of view aren't classified (2026-10-28)
A real HELLO attempt had the hand tracked in ~15% of frames and came out as WHITE 79%: a confident wrong answer.
- A segment with a hand in < 60% of its frames is not accepted. Real INCLUDE signs have ≥ 81%, so none are lost.

## D-007 The sign plan only uses signs that were said (2026-10-29)
- "नमस्ते" was planned as GOOD_MORNING, and "which colour do you like?" as HOW_ARE_YOU. The plan prompt now lists
  each sign's English and Hindi meanings, translates first, and forbids adding unsaid signs. When keywords match,
  the model's signs must be among them; repeated signs are merged. 7 of 7 test phrases correct.

## D-005 Sign videos are cropped INCLUDE clips (2026-10-28)
An avatar is its own research problem, and Deaf users often find synthetic avatars stilted. One clip per sign
(median length) is trimmed and cropped to the upper body as ~1 MB WebM. Credit INCLUDE (CC BY-NC-SA).

## D-004 AI calls through one module; key only in the backend (2026-10-28)
`server/meta.py` is the only file that calls Meta (`chat_json`, `transcribe`, `speak`, plus SAM 3.1 for the Python
tools). The key never reaches the browser. Every call has an offline fallback (`rules.py`).

## D-003 Train on clips cut by the live segmenter (2026-10-28)
Training used its own trimming rule while the browser used the segmenter. Under live cutting, the model scored
87.9% on the test clips. Trained on segmenter cuts (`cut_sign`): 97–100% across three runs, and the browser cuts
clips identically to Python.

## D-002 SignNet runs from local files, no CDN (2026-10-28)
`onnxruntime-web/wasm` (CPU) needs one 15 MB `.wasm`, copied to `public/ort/` at install. The demo works offline.
The browser matches PyTorch on all test clips (difference < 1e-6).

## D-001 Recognize signs on the device; use Meta's models for language and speech (2026-10-28)
Meta credits are limited and video is personal. Tracking (MediaPipe) and the sign classifier (SignNet) run on the
laptop for free; only text and the hearing person's voice go to Meta's models (meta-models.md).
