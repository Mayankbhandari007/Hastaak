"""All calls to Meta: Muse Spark (chat) and Muse Voice Transcribe (speech → text) through the Meta Model API, and
MMS-TTS (Meta's open-weights text-to-speech, run on this machine: the Model API has no text-to-speech).
Used by server/main.py (the backend) and by the python/ tools below.
Also SAM 3.1 (find people in an image or video), used by python/vision_server.py and python/make_clips.py,
which import this module so the key and every Meta call stay in one place."""
import asyncio
import base64
import io
import json
import os
import re
import wave
from pathlib import Path

import httpx

BASE = "https://api.meta.ai/v1"
KEY_ENV = "MODEL_API_KEY"
ENV_FILE = Path(__file__).parent / ".env"


def load_env():
    """For tools outside the backend (python/): read server/.env into the environment if not already set."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                os.environ.setdefault(name.strip(), value.strip())


def has_key() -> bool:
    return bool(os.environ.get(KEY_ENV, "").strip())

# Language hints for Muse Voice Transcribe (it takes names; it also auto-detects).
# It supports Hindi, Bengali, Kannada, Marathi, Tamil, Telugu and English among India's languages.
TRANSCRIBE_NAMES = {"en-IN": "English", "hi-IN": "Hindi", "bn-IN": "Bengali", "kn-IN": "Kannada",
                    "mr-IN": "Marathi", "ta-IN": "Tamil", "te-IN": "Telugu"}
# MMS-TTS voices (facebook/mms-tts-<code>), one small model per language, downloaded on first use
TTS_CODES = {"en-IN": "eng", "hi-IN": "hin", "bn-IN": "ben", "ta-IN": "tam", "te-IN": "tel", "mr-IN": "mar",
             "gu-IN": "guj", "kn-IN": "kan", "ml-IN": "mal", "pa-IN": "pan", "od-IN": "ory"}


def _headers() -> dict:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"{KEY_ENV} is not set in server/.env")
    return {"Authorization": f"Bearer {key}"}


def _check(r: httpx.Response):
    if r.status_code >= 400:
        try:
            msg = r.json().get("error", {}).get("message") or r.text
        except ValueError:
            msg = r.text
        raise RuntimeError(f"Meta {r.request.url.path} → HTTP {r.status_code}: {msg[:300]}")


async def chat_json(system: str, user: str) -> dict:
    """One Muse Spark turn that must answer with a JSON object. Muse Spark always reasons; "minimal" keeps it
    short. Reasoning tokens count toward max_completion_tokens, so leave room for them."""
    body = {
        "model": os.environ.get("META_CHAT_MODEL", "muse-spark-1.3"),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "reasoning_effort": os.environ.get("META_REASONING", "minimal"),
        "max_completion_tokens": 2000,
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{BASE}/chat/completions", json=body, headers=_headers())
    _check(r)
    content = r.json()["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", content, flags=re.S)
    if not m:
        raise RuntimeError(f"Chat reply was not JSON: {content[:200]}")
    return json.loads(m.group(0))


async def transcribe(audio: bytes, filename: str, mime: str, language: str = "unknown") -> dict:
    """Speech → text with Muse Voice Transcribe. It accepts only WAV (mono, 16-bit PCM, 16 or 24 kHz); the demo
    page converts its recording to 16 kHz WAV before uploading. Returns {"text", "language"}."""
    if audio[:4] != b"RIFF":
        raise RuntimeError("Muse Voice Transcribe needs WAV audio (16 kHz mono); got " + (mime or "unknown"))
    req = {"model": os.environ.get("META_STT_MODEL", "muse-voice-transcribe-1.0"),
           "audioEncoding": "WAV", "mode": "PUSH_TO_TALK"}
    if language in TRANSCRIBE_NAMES:
        req["languageBias"] = [TRANSCRIBE_NAMES[language]]
    files = {"request": (None, json.dumps(req), "application/json"), "audio": (filename, audio, "audio/wav")}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BASE}/asr/transcribe", files=files, headers=_headers())
    _check(r)
    return {"text": (r.json().get("transcript") or "").strip(), "language": None}


_tts = {}            # code → (model, tokenizer), loaded once


def _load_tts(code: str):
    if code not in _tts:
        from transformers import AutoTokenizer, VitsModel          # heavy imports only when first needed
        name = f"facebook/mms-tts-{code}"
        _tts[code] = (VitsModel.from_pretrained(name).eval(), AutoTokenizer.from_pretrained(name))
    return _tts[code]


def _speak_sync(text: str, language: str) -> str:
    import numpy as np
    # Each voice reads only its own script: Latin-letter text (e.g. the offline English template) uses English
    code = "eng" if text.isascii() else TTS_CODES.get(language, "eng")
    try:
        wav, rate = _synthesize(text, code)
    except Exception:                                         # a voice that can't read the text: try English
        if code == "eng":
            raise
        wav, rate = _synthesize(text, "eng")
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def _synthesize(text: str, code: str):
    """(waveform float array, sample rate) from the MMS-TTS voice for `code`."""
    import torch
    model, tok = _load_tts(code)
    if getattr(tok, "is_uroman", False):                      # some voices read romanized text
        import uroman
        text = uroman.Uroman().romanize_string(text)
    with torch.no_grad():
        return model(**tok(text, return_tensors="pt")).waveform[0].numpy(), model.config.sampling_rate


# ---------------------------------------------------------------- SAM 3.1 (synchronous: called from threads/tools)
SAM_MODEL = "sam-3.1"


def _sam_text(content: list, timeout: float) -> str:
    """One SAM 3.1 request through the Responses API; returns its output text (special-token records)."""
    body = {"model": os.environ.get("META_SAM_MODEL", SAM_MODEL), "stream": True,
            "input": [{"type": "message", "role": "user", "content": content}]}
    parts = []
    with httpx.stream("POST", f"{BASE}/responses", json=body, timeout=timeout,
                      headers={**_headers(), "Accept": "text/event-stream"}) as r:
        if r.status_code >= 400:
            r.read()
            _check(r)
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("type") == "response.output_text.delta":
                parts.append(ev.get("delta", ""))
            elif ev.get("type") in ("response.failed", "error"):
                raise RuntimeError(f"SAM 3.1 failed: {json.dumps(ev)[:300]}")
    return "".join(parts)


def _parse(text: str, video: bool):
    import meta_sam_parser as sp
    fmt = sp.video_segmentation_format() if video else sp.image_segmentation_format()
    parser = fmt.create_parser()
    parser.push(text, emit=False)
    return parser.finish(sp.CompletedOutcome()).result.records


def find_people(jpeg: bytes, timeout: float = 6.0) -> list[dict]:
    """SAM 3.1 on one image with the prompt "person". Returns [{"box": (left, top, right, bottom)}] in pixels."""
    import meta_sam_parser as sp
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    text = _sam_text([{"type": "input_text", "text": "person"}, {"type": "input_image", "image_url": url}], timeout)
    return [{"box": (r.left, r.top, r.right, r.bottom)} for r in _parse(text, video=False)
            if isinstance(r, sp.SegmentationBoxRecord)]


def person_masks(video_path: str, timeout: float = 300.0) -> dict[int, list]:
    """SAM 3.1 on a whole video (MP4/MOV, H.264) with the prompt "person". Uploads it with the Files API.
    Returns {frame_index: [(object_id, mask uint8 h×w (1 = person) covering its box, (left, top) of the box)]}."""
    import numpy as np
    import meta_sam_parser as sp
    with open(video_path, "rb") as fh, httpx.Client(timeout=timeout) as c:
        r = c.post(f"{BASE}/files", headers=_headers(), data={"purpose": "user_data"},
                   files={"file": (Path(video_path).name, fh, "video/quicktime")})
    _check(r)
    file_id = r.json()["id"]
    text = _sam_text([{"type": "input_text", "text": "person"}, {"type": "input_video", "file_id": file_id}], timeout)
    frames: dict[int, list] = {}
    for rec in _parse(text, video=True):
        if not isinstance(rec, sp.SegmentationMaskRecord):
            continue
        m = rec.mask
        raw = np.frombuffer(sp.decode_mask_to_raster(m), np.uint8)
        local = (raw.reshape(m.height, m.width) if raw.size == m.height * m.width
                 else np.unpackbits(raw)[: m.height * m.width].reshape(m.height, m.width))
        frames.setdefault(rec.frame.frame_index if rec.frame else 0, []).append(
            (rec.object_id, local, (int(rec.bounds.left), int(rec.bounds.top))))
    return frames


def warm_up():
    """Load the Hindi and English voices at start-up (first load downloads ~150 MB each, then cached)."""
    for code in ("hin", "eng"):
        _load_tts(code)


async def speak(text: str, language: str) -> str:
    """Text → base64 WAV with MMS-TTS on this machine (CPU, about a second per sentence once loaded)."""
    return await asyncio.to_thread(_speak_sync, text[:500], language)
