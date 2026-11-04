# Meta's models: what runs where

Which Meta model does each job, where it runs, what it costs, and how it was tested (decisions.md D-020 to D-026).

## Status
| Job | Model | Where it runs | Code | Tested? |
| --- | ----- | ------------- | ---- | ------- |
| Signs → sentence; speech → sign plan | **Muse Spark 1.3** (`muse-spark-1.3`, reasoning `minimal`) | Meta Model API | `server/meta.py` `chat_json` | yes: 5 of 7 test phrases planned right; Hindi sentence spoken (testing.md) |
| Hearing person's speech → text | **Muse Voice Transcribe** (`muse-voice-transcribe-1.0`) | Meta Model API | `server/meta.py` `transcribe` | yes: Hindi, English, Tamil; mishears some synthetic-voice words |
| Sentence → voice | **MMS-TTS** (Meta open weights, `facebook/mms-tts-<lang>`) | this machine, CPU | `server/meta.py` `speak` | yes: Hindi, English and Tamil intelligible |
| Recognition model | SignNet in **PyTorch** | `vision_server.py` | `signnet.pt` | yes: identical to ONNX on 33/33 clips |
| Signer lock | **SAM 3.1** (image, prompt "person") | Meta Model API, only with 2+ people | `vision_server.py --signer-lock` | not measured yet |
| Clean sign videos | **SAM 3.1** (video) | Meta Model API, once | `make_clips.py --sam` | not measured yet |
| Hand/body/face tracking | MediaPipe (Google) | this machine | `features.py`, `web/src/` | — see below |
| Signing animation | 2D avatar from recorded motion (no model) | browser | `export_avatar.py`, `web/src/avatar.ts` | yes: renders every sign |
| Browser camera mode | MediaPipe + ONNX Runtime | browser | `web/src/` | fallback only |

## Switching on
1. Put your key in `server/.env`: `MODEL_API_KEY=...` (see `.env.example`).
2. Restart the backend. The pill should read **Connected: Meta**. The first start downloads the Hindi and English
   voices (~150 MB each, cached afterwards).
3. Test (testing.md, "Speech and language"): the seven phrases through `/api/listen`, and one sentence in Hindi.
4. Optional: `python make_clips.py --sam` (~$0.15) for cut-out sign videos; `python vision_server.py --signer-lock`
   for the bystander demo.

## Things to know
- **Voice:** MMS-TTS runs on your laptop: no cost, about 3 s per sentence on this machine. Each language's voice downloads on first use.
- **Speech-to-text languages:** Muse Voice Transcribe supports Hindi, Bengali, Kannada, Marathi, Tamil, Telugu and
  English (not Gujarati, Punjabi, Malayalam or Odia). The voice output still covers all 11 languages.
- **Audio format:** Voice Transcribe accepts only WAV, so the page converts its recording to 16 kHz mono WAV
  (`web/src/wav.ts`).
- **Reasoning:** Muse Spark always reasons. `minimal` keeps it short. Reasoning tokens are billed as output.
- **Data:** transcripts, sentences and (with signer lock) single camera frames go to Meta (security.md). Never use
  the Contributor models (`*-contributor`) with real users: their data trains Meta's models.

## Costs (from the price list, $148 credit)
| Use | Price | Typical |
| --- | ----- | ------- |
| Muse Spark 1.3 | $1.25 / $4.25 per 1M tokens in / out | ~$0.001 per call → ~$1 per 1,000 |
| Muse Voice Transcribe | $0.18 per hour of audio | 500 replies of 10 s ≈ $0.25 |
| SAM 3.1 image (signer lock) | $2.50 per 1,000 images | ≤ 1 call per 3 s, only with 2+ people → ≤ $3 / hour |
| SAM 3.1 video (sign videos) | $0.20 per 1,000 frames (min 50 per video) | ~770 frames ≈ $0.15 once |
| MMS-TTS | free (local) | — |

Don't code with Muse Wonder ($4 / $20): a coding session can use much of the credit.

## Tracking: MediaPipe or Sapiens2
None of the Meta Model API models give finger joints. Two paths:
1. **MediaPipe stays** if the organisers allow it (ask them). Nothing changes.
2. **Sapiens2** (Meta, open weights, 308 keypoints incl. both hands) if it must be Meta. First run
   `python sapiens_check.py <video>` to compare it with MediaPipe on the same frames (speed and fingertips, with
   side-by-side pictures). Switch only if its fingers are as good. Switching means:
   - feature spec v2 (Sapiens2's hand points), with a new `FEATURE_SPEC_VERSION`;
   - re-extracting all training data on a GPU, then retraining;
   - running recognition where the GPU is: `vision_server.py` becomes the GPU service.

   On a laptop CPU Sapiens2 takes seconds per frame, so live use needs a GPU.

**Free GPUs:** Kaggle notebooks (T4 or P100, 16 GB, about 30 hours a week, needs phone verification) and Google
Colab (T4, time-limited sessions). Both fit the 0.4B–1B models for **batch** work (extracting features, the
comparison above), not for serving a live demo. For live use, an AWS `g4dn.xlarge` (T4, ~$0.5–0.6/hour, request
the GPU quota early) in Mumbai.
