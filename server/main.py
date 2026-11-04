"""Hastaak backend: keeps the API key, turns signs into speech and speech into signs.
Run:  uvicorn main:app --reload --port 8000     (from the server folder)
The web app's dev server proxies /api here."""
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import meta  # the one module that talks to Meta (rule 5)
from rules import KEYWORDS, keyword_signs, pretty, template_sentence

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

WEB_PUBLIC = HERE.parent / "web" / "public"
LANGUAGES = {"en-IN": "English", "hi-IN": "Hindi", "bn-IN": "Bengali", "ta-IN": "Tamil", "te-IN": "Telugu",
             "mr-IN": "Marathi", "gu-IN": "Gujarati", "kn-IN": "Kannada", "ml-IN": "Malayalam",
             "pa-IN": "Punjabi", "od-IN": "Odia"}

app = FastAPI(title="Hastaak")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def warm_up():
    """Load MMS-TTS in the background, so the first spoken sentence is fast."""
    asyncio.get_running_loop().run_in_executor(None, meta.warm_up)


def recognizable_signs() -> list[str]:
    model = json.loads((WEB_PUBLIC / "models" / "signnet.json").read_text())
    return [l for l in model["labels"] if l != "NOTHING"]


def showable_signs() -> list[str]:
    """Signs we have a video clip for (the hearing → Deaf direction can only show these)."""
    p = WEB_PUBLIC / "clips" / "clips.json"
    return sorted(json.loads(p.read_text())) if p.exists() else []


def sign_meanings(signs: list[str]) -> str:
    """'- HELLO: hello, hi, नमस्ते, …' lines, so the model maps e.g. नमस्ते to HELLO, not GOOD_MORNING."""
    return "\n".join(f"- {s}: {', '.join(dict.fromkeys([pretty(s), *KEYWORDS.get(s, [])]))}" for s in signs)


class Candidate(BaseModel):
    label: str
    p: float


class SignIn(BaseModel):
    candidates: list[Candidate] = Field(min_length=1, max_length=3)
    question: bool = False


class SentenceIn(BaseModel):
    signs: list[SignIn] = Field(min_length=1, max_length=12)
    language: str = "en-IN"
    speak: bool = True


SENTENCE_PROMPT = """You voice a Deaf person who is signing in Indian Sign Language (ISL) to a hearing listener.
Input: the signs in order. Each sign has up to 3 candidate glosses with probabilities from a recognizer
that makes mistakes, and question=true when the signer raised their eyebrows (a yes/no question).
Choose the most plausible candidate for each sign, then write ONE short, natural sentence the Deaf person
means, in {language} (native script). ISL word order differs from English; reorder freely.
Do not add facts the signs don't imply. Reply with JSON: {{"text": "<sentence>", "signs": ["<chosen gloss>", ...]}}"""

PLAN_PROMPT = """You help a Deaf person read what a hearing person said, using Indian Sign Language (ISL) sign videos.
You may ONLY use these signs (each with words it covers):
{signs}
First translate what was said into English, faithfully. Then pick a sign only for a word or phrase that was
actually said and matches that sign's meaning; never add greetings, questions or anything not said, and
never use a sign just because it is available. Skip everything without a matching sign; an empty list is fine.
Use ISL order (time first, question words last).
Reply with JSON: {{"english": "<faithful English translation>", "signs": ["<SIGN>", ...]}}"""


@app.get("/api/health")
def health():
    return {"provider": "Meta", "configured": meta.has_key(),
            "signs": recognizable_signs(), "showable": showable_signs(), "languages": LANGUAGES}


@app.post("/api/sentence")
async def sentence(req: SentenceIn):
    """Deaf → hearing: recognized signs → natural sentence (+ spoken audio)."""
    language = req.language if req.language in LANGUAGES else "en-IN"
    top = [s.candidates[0].label for s in req.signs]
    question = any(s.question for s in req.signs)
    result = {"text": template_sentence(top, question), "source": "template", "signs": top}
    try:
        payload = [{"candidates": [c.model_dump() for c in s.candidates], "question": s.question} for s in req.signs]
        out = await meta.chat_json(SENTENCE_PROMPT.format(language=LANGUAGES[language]), json.dumps(payload))
        text = str(out.get("text", "")).strip()
        if text and len(text) < 400:
            result.update(text=text, source="meta",
                          signs=[s for s in out.get("signs", []) if isinstance(s, str)] or top)
    except Exception as e:                      # keep going with the template sentence
        result["error"] = str(e)
    if req.speak and result["text"]:
        try:
            result["audio"] = await meta.speak(result["text"], language)
        except Exception as e:
            result["error"] = f"{result.get('error', '')} {e}".strip()
    return result


@app.post("/api/listen")
async def listen(file: UploadFile = File(...), language: str = Form("unknown")):
    """Hearing → Deaf: speech → caption → ISL sign sequence (only signs that have a clip)."""
    audio = await file.read()
    try:
        heard = await meta.transcribe(audio, file.filename or "speech.webm",
                                          file.content_type or "audio/webm", language)
    except Exception as e:
        return {"text": "", "signs": [], "source": "none", "error": str(e)}
    vocab = showable_signs()
    result = {"text": heard["text"], "language": heard["language"], "english": None,
              "signs": keyword_signs(heard["text"], vocab), "source": "keywords"}
    if heard["text"]:
        try:
            out = await meta.chat_json(PLAN_PROMPT.format(signs=sign_meanings(vocab)), heard["text"])
            signs = [s for s in out.get("signs", []) if s in vocab]
            # Without reasoning the model sometimes adds signs that weren't said ("which colour…" → HOW_ARE_YOU).
            # When keywords matched, they act as a check: keep the model's order, drop what keywords didn't find.
            if result["signs"]:
                signs = [s for s in signs if s in result["signs"]] or result["signs"]
            result.update(english=out.get("english"), source="meta",
                          signs=[s for i, s in enumerate(signs) if i == 0 or s != signs[i - 1]])
        except Exception as e:
            result["error"] = str(e)
    result["glosses"] = [pretty(s) for s in result["signs"]]
    return result
