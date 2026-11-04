# Security and data handling

Hastaak handles video and voice of people, some of them Deaf users who may be filmed while testing.
Treat camera recordings, voice and your own sign recordings as personal data.

## Secrets
- The Meta key (`MODEL_API_KEY`) lives only in `server/.env`. `.env.example` has the names,
  never values. `vision_server.py` and `make_clips.py` read the Meta key from that same file (via `server/meta.py`).
- The browser talks to `/api/*` and `/vision/*` and never sees a key. CORS allows only `http://localhost:5173`.
- Never paste a key into chat, code, a doc or a screenshot. If one leaks, revoke it in the Meta developer console
  and create a new one.
- Set a spending alert or limit on the Meta account if the console offers one.

## What leaves the laptop
| Data | Goes to | When |
| ---- | ------- | ---- |
| Camera video | **nowhere** in normal use: both camera modes process it locally | — |
| One camera frame (640 px JPEG) | SAM 3.1 (Meta) | only with `--signer-lock` and 2+ people in view, ≤ 1 per 3 s |
| Recognized sign names + probabilities | Muse Spark (Meta) | each "Speak sentence" |
| The hearing person's voice (≤ 25 s, WAV) | Muse Voice Transcribe (Meta) | each mic recording |
| The transcript | Muse Spark (Meta) | each mic recording |
| The sentence to speak | **nowhere**: MMS-TTS runs on this laptop | — |
| INCLUDE clips (dataset, not people you recorded) | SAM 3.1 (Meta) | once, `make_clips.py --sam` |

Tell people being recorded where their voice goes. **Never use the `*-contributor` Meta models with real users:**
Meta trains its models on their inputs and outputs. If tracking later moves to Sapiens2 on a GPU server, camera
video will leave the device too (meta-models.md); say so in the pitch and in consent.

## Recordings on disk
- `python/recordings/<time>/preview.webm` is video of a person's face and body. `features.json` describes their
  movements. Keep them on this laptop; don't put them in git, chats or shared drives without the person's consent.
- Own training recordings (`python/data/own/`) and playground downloads are numbers, not video, but still come
  from a person: same rules.
- **Volunteers:** before recording someone, get consent covering what is recorded, why, how long it is kept
  (delete after the hackathon unless they agree otherwise) and their right to withdraw.
- Delete `python/recordings/` folders you no longer need.

## Logs
- The backend's uvicorn logs show request paths only, never transcripts or sentences. Keep it that way: don't
  log text, audio or features.
- The vision server logs nothing per frame.

## Licences
- **INCLUDE** (AI4Bharat): CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike). Fine for the hackathon;
  credited under the sign videos. The avatar's motion comes from it too. Commercial use would need other data
  or the authors' permission.
- **MMS-TTS** voices (Meta): CC BY-NC 4.0, non-commercial.
- **Sapiens2** (Meta): its own Sapiens2 licence; read it before using it beyond testing.
- **MediaPipe** models: Apache 2.0.

## Local servers
Both servers listen on localhost only and have no sign-in. Don't expose ports 8000, 8001 or 5173 to a network
(no port forwarding, no `--host 0.0.0.0`) without adding authentication first.
