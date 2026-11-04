"""Sarvam AI provider: chat (sentence building / sign planning), speech-to-text, text-to-speech.
Any provider module (e.g. a future meta.py) exposes the same three async functions."""
import json
import os
import re

import httpx

BASE = "https://api.sarvam.ai"


def _key() -> str:
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set in server/.env")
    return key


def _check(r: httpx.Response):
    if r.status_code >= 400:
        try:
            msg = r.json().get("error", {}).get("message") or r.text
        except ValueError:
            msg = r.text
        raise RuntimeError(f"Sarvam {r.request.url.path} → HTTP {r.status_code}: {msg[:300]}")


async def chat_json(system: str, user: str) -> dict:
    """One chat turn that must answer with a JSON object. Reasoning is switched off for speed."""
    key = _key()
    body = {
        "model": os.environ.get("SARVAM_CHAT_MODEL", "sarvam-105b"),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": 400,
        "reasoning_effort": None,
        "response_format": {"type": "json_object"},
    }
    # The docs show both auth styles for chat; sending both is accepted.
    headers = {"api-subscription-key": key, "Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{BASE}/v1/chat/completions", json=body, headers=headers)
    _check(r)
    content = r.json()["choices"][0]["message"].get("content") or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    m = re.search(r"\{.*\}", content, flags=re.S)
    if not m:
        raise RuntimeError(f"Chat reply was not JSON: {content[:200]}")
    return json.loads(m.group(0))


async def transcribe(audio: bytes, filename: str, mime: str, language: str = "unknown") -> dict:
    """Speech → text in the spoken language. Returns {"text", "language"}. REST limit: 30 s of audio."""
    data = {"model": os.environ.get("SARVAM_STT_MODEL", "saaras:v3"), "mode": "transcribe",
            "language_code": language or "unknown"}
    mime = mime.split(";")[0].strip() or "audio/webm"   # Sarvam rejects "audio/webm;codecs=opus" (Chrome's recorder type)
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{BASE}/speech-to-text", data=data, files={"file": (filename, audio, mime)},
                         headers={"api-subscription-key": _key()})
    _check(r)
    j = r.json()
    return {"text": (j.get("transcript") or "").strip(), "language": j.get("language_code")}


async def speak(text: str, language: str) -> str:
    """Text → base64 WAV."""
    body = {"text": text[:2500], "language_code": language,
            "model": os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")}
    if os.environ.get("SARVAM_TTS_SPEAKER"):
        body["speaker"] = os.environ["SARVAM_TTS_SPEAKER"]
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{BASE}/text-to-speech", json=body, headers={"api-subscription-key": _key()})
    _check(r)
    return r.json()["audios"][0]
