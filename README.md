# Hastaak

Two-way Indian Sign Language ⇄ speech assistant. The Deaf person signs, and the listener hears a sentence in their
language (Hindi, English and 9 more Indian languages). The hearing person speaks, and the Deaf person sees captions
plus ISL sign videos.

    web/      browser app: demo.html (Hastaak) and index.html (playground / recorder)
    server/   FastAPI backend: keeps the API key, calls Sarvam (chat, speech-to-text, text-to-speech)
    python/   tracking, training, evaluation, ONNX export, sign videos, native camera server
    docs/     how everything works and why: start with CLAUDE.md (the index)

## Quick start
First-time setup (virtual environments, `npm install`, the `.env` file) is in
[docs/local-setup.md](docs/local-setup.md). Then open three terminals in the project folder and run:

    # terminal 1: backend
    cd server
    .venv\Scripts\python -m uvicorn main:app --reload --port 8000

    # terminal 2: web app
    cd web
    npm run dev

    # terminal 3: native camera
    cd python
    .venv\Scripts\python vision_server.py

Open http://localhost:5173/demo.html in Chrome. Check it says **Connected: sarvam** and **Native camera (faster)**,
then press **Start camera**.
- **Deaf → hearing:** raise your hands, sign, lower them. Repeat, then hold a thumbs-up (or press Speak).
- **Hearing → Deaf:** tap the microphone, speak, tap again.

## Documentation
| Read | For |
| ---- | --- |
| [CLAUDE.md](CLAUDE.md) | index, rules, glossary |
| [docs/design.md](docs/design.md) | the problem, what we learned, why it's built this way |
| [docs/architecture.md](docs/architecture.md) | components, folders, endpoints |
| [docs/local-setup.md](docs/local-setup.md) | setup, running, training on your own signing, common problems |
| [docs/pipeline.md](docs/pipeline.md) | recognition, sentence, listening and training step by step |
| [docs/data-formats.md](docs/data-formats.md) | feature spec, recordings, model files, API bodies |
| [docs/prompts.md](docs/prompts.md) | the AI prompts and their fallbacks |
| [docs/testing.md](docs/testing.md) | how accuracy is checked, and the results |
| [docs/security.md](docs/security.md) | API key, recordings of people, dataset licence |
| [docs/coding-standards.md](docs/coding-standards.md) | rules for changing code |
| [docs/meta-migration.md](docs/meta-migration.md) | moving to Meta's models |
| [docs/decisions.md](docs/decisions.md) | every design decision and the evidence behind it |

Sign data and videos: INCLUDE dataset (AI4Bharat), CC BY-NC-SA 4.0 (non-commercial).
