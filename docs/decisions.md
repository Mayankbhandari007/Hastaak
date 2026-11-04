# Decision log

Newest first. One entry per decision: context, decision, consequence. Never delete an entry; add a new one that
supersedes it. Numbers in entries come from the checks in testing.md.

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

## D-007 Speech-to-signs fixes from the first test with a key (2026-10-29)
- Chrome records `audio/webm;codecs=opus`; Sarvam rejected that type string. The provider now sends `audio/webm`.
- "नमस्ते" was planned as GOOD_MORNING, and "which colour do you like?" as HOW_ARE_YOU. The plan prompt now lists
  each sign's English and Hindi meanings, translates first, and forbids adding unsaid signs. When keywords match,
  the model's signs must be among them; repeated signs are merged. 7 of 7 test phrases correct.

## D-006 Sarvam chat: `sarvam-105b`, reasoning off, JSON replies (2026-10-29)
`sarvam-30b` and `sarvam-m` are deprecated. With reasoning on (`low`), replies took ~10 s and often used all
tokens with no answer. With it off: ~1 s, correct on the test phrases.

## D-005 Sign videos are cropped INCLUDE clips (2026-10-28)
An avatar is its own research problem, and Deaf users often find synthetic avatars stilted. One clip per sign
(median length) is trimmed and cropped to the upper body as ~1 MB WebM. Credit INCLUDE (CC BY-NC-SA).

## D-004 AI calls through one provider module; key only in the backend (2026-10-28)
Sarvam is the stand-in for Meta's models. `server/sarvam.py` is the only file that calls it (`chat_json`,
`transcribe`, `speak`). A Meta module with the same functions is selected with `PROVIDER`. Every call has an offline
fallback (`rules.py`).

## D-003 Train on clips cut by the live segmenter (2026-10-28)
Training used its own trimming rule while the browser used the segmenter. Under live cutting, the model scored
87.9% on the test clips. Trained on segmenter cuts (`cut_sign`): 97–100% across three runs, and the browser cuts
clips identically to Python.

## D-002 SignNet runs from local files, no CDN (2026-10-28)
`onnxruntime-web/wasm` (CPU) needs one 15 MB `.wasm`, copied to `public/ort/` at install. The demo works offline.
The browser matches PyTorch on all test clips (difference < 1e-6).

## D-001 Build a non-Meta prototype first (2026-10-28)
Meta credits are limited. The prototype uses MediaPipe + a small custom classifier (SignNet) + Sarvam, so it can be
tested end to end. The move to Meta's models is planned in meta-migration.md.
