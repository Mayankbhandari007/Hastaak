# Hastaak: index

Two-way Indian Sign Language (ISL) ⇄ speech assistant for the Meta hackathon. A Deaf person signs, and the hearing
listener hears a sentence in their language. The hearing person speaks, and the Deaf person sees captions and ISL
sign videos. This file is the index. Read the file that matches your task before changing anything.

| Read this                  | When you are...                                                        |
| -------------------------- | ---------------------------------------------------------------------- |
| docs/design.md             | new to the project, or asking "why is it like this?"                   |
| docs/architecture.md       | adding a component, endpoint, page or model                            |
| docs/coding-standards.md   | writing ANY code (non-negotiable rules, incl. the Python/TS twins)     |
| docs/data-formats.md       | touching features, recordings, model files or API bodies               |
| docs/pipeline.md           | working on recognition, the segmenter, training or the sign videos     |
| docs/prompts.md            | changing a prompt, the sign plan or the fallbacks                      |
| docs/testing.md            | checking accuracy, or changing anything that affects recognition       |
| docs/security.md           | handling the API key, recordings of people, or the dataset licence     |
| docs/local-setup.md        | setting up a machine, running the demo, or fixing a start-up problem   |
| docs/meta-models.md        | checking which Meta model does what, its cost, or the tracking choice  |
| docs/decisions.md          | making a choice that future readers need to know about                 |

## The six rules you must not break
1. Python and the browser compute features, cut signs and build the model input **identically**. Change one twin,
   change the other, and check parity (coding-standards.md).
2. The feature layout is a contract (feature spec v1, 156 numbers). Any change bumps `FEATURE_SPEC_VERSION`
   on both sides, then re-extract and retrain.
3. Train on clips cut **exactly as the live segmenter cuts them** (`segmenter.cut_sign`), never on hand-trimmed clips.
4. The model never sees face values (`dataset.model_input`). The training data has none, so the face is used
   only for question detection.
5. Only server-side code talks to Meta, and only through one module, `server/meta.py` (`vision_server.py` and
   `make_clips.py` import it for SAM 3.1). The API key lives only
   in `server/.env`: never the browser, git or a log.
6. Every change that can affect recognition is measured on the INCLUDE test clips **and** on a recording of a
   real user before it is kept (testing.md). A change that doesn't help is reverted, not left in "for now".

## Glossary
- **Sign / gloss**: one ISL word, written in capitals (`HELLO`, `GOOD_MORNING`). The model knows 47 (data-formats.md).
- **Frame / feature vector**: 156 numbers per camera frame (hands, 7 body points, 7 face values), in shoulder widths.
- **Segment**: the frames of one sign, cut by the segmenter (hands up and moving → still or lowered).
- **SignNet**: the small sequence classifier (PyTorch, exported to ONNX) that turns one segment into a sign.
- **INCLUDE**: AI4Bharat's ISL dataset. The model uses 47 signs (Adjectives, Greetings,
  Pronouns), 734 clips (`E:\12\Final_Dataset`).
- **Native camera mode**: Python (`vision_server.py`) reads the webcam and recognizes signs. **Browser mode**: the
  page does it with MediaPipe + ONNX in JavaScript.
- **Sign plan**: the ordered signs chosen to show what the hearing person said.
- **Own recordings**: your labelled clips in `python/data/own/`, added to training.
- **Meta models**: Muse Spark, Muse Voice Transcribe and SAM 3.1 via the Meta Model API, plus MMS-TTS run
  locally; all called from `server/meta.py`.
- **Avatar**: the 2D figure that signs in the hearing → Deaf direction, replaying real signers' recorded motion.
- **Signer lock**: SAM 3.1 picks which person is signing when several are in view (`vision_server.py --signer-lock`).
