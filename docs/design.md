# Design: why the system looks like this

Read alongside architecture.md and pipeline.md.

## Problem
Deaf people who use Indian Sign Language often can't have a quick face-to-face conversation with someone who
doesn't sign, without an interpreter or writing things down. Hastaak sits between them:

| Direction | Input | Output |
| --------- | ----- | ------ |
| Deaf → hearing | ISL signs in front of a camera | a natural sentence, shown and spoken in the listener's language |
| Hearing → Deaf | speech in any Indian language or English | a caption, its English translation, and ISL sign videos |

It is built for the Meta hackathon. The prototype uses non-Meta parts (MediaPipe, ONNX Runtime, Sarvam), so
it could be tested without spending Meta credits. The swap is planned in meta-migration.md.

## What we learned from real tests
These findings shaped the current design (details and numbers in testing.md and decisions.md):
- **Sign recognition only works on clips cut like live.** Trained on hand-trimmed clips, accuracy under live
  cutting was 87.9%; trained on live-style cuts, 97–100% (D-003).
- **Laptop cameras lose hands.** Tracking flickers, so short gaps are bridged (D-010). Signs that are mostly
  missing hands are rejected with a message instead of a confident wrong answer (D-008).
- **A "no sign" class made from real signs backfired.** It rejected real, flickery signs (D-009). A NOTHING class
  now exists only if you record real "no sign" clips.
- **A busy laptop runs the browser at 1–2 fps.** The same models run natively in Python at 7–9 fps, so the
  demo has a native camera mode (D-014). The segmenter's timing adapts to the frame rate (D-013).
- **INCLUDE has no face data** (1 of 14,170 frames): the signers stand too far away. The model's face input
  is therefore always zero (D-015).
- **The model doesn't carry over to a new signer.** Your signs come out as ORANGE / GOOD_AFTERNOON even when
  tracking is perfect. Signing about 1.5 shoulder widths higher than INCLUDE's signers (natural when seated
  close) drops accuracy on INCLUDE's own clips from 32/33 to 5/33. Left-handedness, hand size, the face and the
  resting hand were tested and are not the cause (D-016, D-017). **Your own recordings are the fix.**

## Core approach
1. **Track:** MediaPipe gives hand, body and face landmarks per frame. Pose runs every 2nd frame, face every 3rd.
2. **Describe:** each frame becomes 156 numbers relative to the shoulders (feature spec v1).
3. **Cut:** the segmenter finds each sign: it starts when a hand is raised and moving, and ends when the hands
   are still or lowered.
4. **Recognize:** SignNet scores the segment (top 3). Weak or unclear results become "Did you mean" buttons,
   not wrong words.
5. **Speak:** Sarvam chat turns the signs (with their top-3 guesses and the question flag) into one sentence in
   the listener's language. Sarvam text-to-speech voices it.
6. **Listen:** Sarvam speech-to-text transcribes the reply. Sarvam chat picks matching signs, checked by keyword
   rules, and the page plays the sign videos.

Every AI step has a deterministic fallback (template sentence, keyword sign matching), so the demo still runs offline.

## Decisions taken
| Topic | Decision |
| ----- | -------- |
| Vocabulary | 11 INCLUDE signs (Greetings + Colours) for now |
| Tracking | MediaPipe (hands, pose lite, face), the same `.task` files in Python and the browser |
| Classifier | SignNet: 2 × Conv1d + BiGRU, ~0.6M parameters, 32 frames × 156 features |
| Recognition location | native Python (default when running) or in the browser (fallback) |
| Language + speech | Sarvam behind one provider module (`server/sarvam.py`) |
| Chat model | `sarvam-105b`, reasoning off, JSON replies |
| Sign videos | one cropped INCLUDE clip per sign (`make_clips.py`) |
| "No sign" | only from real NOTHING recordings, never synthetic |
| Face | tracked for questions; zeroed for the model |
| Out of scope (now) | continuous fluent signing, fingerspelling, signing avatar, signer lock |

## Open points
- Teach mode: record your own examples per sign straight into `data/own/` from the demo (not built).
- The INCLUDE split is random per clip, so test signers are also in training. Reported accuracies are optimistic.
- The eyebrow baseline is measured once at start and sometimes reads "raised" on a neutral face.
- Muse Spark and Muse Voice Transcribe names and availability need checking (meta-migration.md).

## Risks
| Risk | Mitigation |
| ---- | ---------- |
| Model misreads a new signer | own recordings + retraining; "Did you mean" buttons; tap-to-add fallback |
| Slow laptop | native camera mode; lighter frame schedule; close other apps |
| Hand leaves the frame | bridging of short gaps; "hand went out of view" message; framing advice |
| Network / API down | template sentences + keyword sign plans (no voice) |
| Demo failure on stage | record a backup video; `vision_server.py --video` replays a file |
| Dataset licence | INCLUDE is CC BY-NC-SA 4.0: fine for the hackathon, not for commercial use (security.md) |
