# Security and data handling

Hastaak handles video and voice of people, some of them Deaf users who may be filmed while testing.
Treat camera recordings, voice and your own sign recordings as personal data.

## Secrets
- The Sarvam key lives only in `server/.env`. `.env.example` has the names, never values.
- Only the backend reads it. The browser talks to `/api/*` and never sees the key. CORS allows only
  `http://localhost:5173`.
- Never paste the key into chat, code, a doc or a screenshot. If it leaks, revoke it in the Sarvam dashboard and
  create a new one.
- The same rules apply to the Meta API key later (meta-migration.md).

## What leaves the laptop
| Data | Goes to | When |
| ---- | ------- | ---- |
| Camera video | **nowhere**: native mode and browser mode both process it locally | — |
| Recognized sign names + probabilities | Sarvam chat | each "Speak sentence" |
| The hearing person's voice (≤ 25 s) | Sarvam speech-to-text | each mic recording |
| The transcript | Sarvam chat | each mic recording |
| The sentence to speak | Sarvam text-to-speech | each "Speak sentence" |

Tell people being recorded that their voice is sent to Sarvam for transcription. After the move to Meta models,
the camera video may also leave the device for a GPU server (meta-migration.md). Say so in the pitch and in consent.

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

## Dataset licence
INCLUDE (AI4Bharat) is **CC BY-NC-SA 4.0**: non-commercial, attribution, share-alike. That's fine for the
hackathon. The demo credits it under the sign videos. Commercial use would need retraining on other data or the
authors' permission. MediaPipe models are Apache 2.0.

## Local servers
Both servers listen on localhost only and have no sign-in. Don't expose ports 8000, 8001 or 5173 to a network
(no port forwarding, no `--host 0.0.0.0`) without adding authentication first.
