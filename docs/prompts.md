# Prompts

The two prompts are constants in `server/main.py`. The provider call that uses them is `chat_json` in
`server/sarvam.py`. This file lists what exists, how it is filled, and the fallbacks around it.

## The prompts
| Constant | Used by | Input (user message) | Reply (JSON) |
| -------- | ------- | -------------------- | ------------ |
| `SENTENCE_PROMPT` | `POST /api/sentence` | the signs in order: each with up to 3 `{label, p}` candidates and `question` | `{"text": "<sentence>", "signs": ["<chosen gloss>", ...]}` |
| `PLAN_PROMPT` | `POST /api/listen` | the transcript, as heard | `{"english": "<faithful translation>", "signs": ["<SIGN>", ...]}` |

What each one asks for:
- **SENTENCE_PROMPT:** voice the Deaf person; pick the most plausible candidate per sign; ONE short natural
  sentence in `{language}` (native script); ISL order differs from English, so reorder freely; add no facts the
  signs don't imply.
- **PLAN_PROMPT:** use ONLY the listed signs; translate first, then pick a sign only for a word or phrase that was
  actually said; never add greetings or questions; an empty list is fine; ISL order (time first, question words last).

## Placeholders
| Name | Filled with |
| ---- | ----------- |
| `{language}` | the language name from `LANGUAGES` (e.g. "Hindi") |
| `{signs}` | `sign_meanings(vocab)`: one line per showable sign, e.g. `- HELLO: hello, hi, hey, नमस्ते, नमस्कार, हेलो, हैलो` |

Literal braces in the JSON examples are doubled (`{{ }}`) because the prompts use `str.format`.

## Call settings (`sarvam.chat_json`)
- Model `SARVAM_CHAT_MODEL` (default `sarvam-105b`), temperature 0.2, `max_tokens` 400.
- `reasoning_effort: null` (off). With reasoning on, replies took about 10 s and often ran out of tokens with no
  answer (D-006).
- `response_format: {"type": "json_object"}`. Any `<think>…</think>` text is stripped, then the first `{…}` is
  parsed. No JSON → an error → the fallback.

## Checks around the model
- **Sentence:** the reply is used only if `text` is non-empty and under 400 characters. Otherwise the template
  sentence is kept (`rules.template_sentence`).
- **Sign plan:** signs not in `clips.json` are dropped. If the keyword matcher found signs in the transcript, the
  model's signs must be among them; if none survive, the keyword signs are used. Repeated signs are merged (D-007).
- **Offline:** no key or a failed call gives the template sentence (no audio), or keyword signs only. The reply's
  `source` says which (`sarvam`, `template`, `keywords`).

## Keywords (`server/rules.py`)
`KEYWORDS` maps each sign to the English and Hindi words that mean it. The sign's own name ("good morning")
always counts. Matching is case-insensitive, whole words for English, longest phrase first, in spoken order.
**Adding a sign:** add its English + Hindi words to `KEYWORDS`. The prompt picks them up through `sign_meanings`.

## Changing a prompt
1. Edit the constant in `server/main.py` and restart the backend (`--reload` restarts it on save).
2. Re-run the test phrases (testing.md, "Speech and language"): Hindi, English and Tamil, including a phrase
   with no matching sign. All must still map correctly.
3. Note the change and its test result in decisions.md.

Prompts are not versioned files yet. If results are stored or compared across versions later, move them to
`server/prompts/<purpose>_v<N>.md` and never edit a shipped version.
