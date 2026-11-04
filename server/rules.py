"""Deterministic fallbacks so the demo still works when a Meta call fails or is slow."""
import re

# Extra ways people say each sign (English + Hindi). Sign names themselves ("good morning") always match.
KEYWORDS = {
    "HELLO": ["hello", "hi", "hey", "नमस्ते", "नमस्कार", "हेलो", "हैलो"],
    "HOW_ARE_YOU": ["how are you", "how're you", "कैसे हो", "कैसे हैं", "कैसी हो", "क्या हाल"],
    "ALRIGHT": ["alright", "all right", "okay", "ok", "fine", "ठीक"],
    "GOOD_MORNING": ["good morning", "सुप्रभात", "गुड मॉर्निंग"],
    "GOOD_AFTERNOON": ["good afternoon", "गुड आफ्टरनून"],
    "BLACK": ["black", "काला", "काली", "काले"],
    "WHITE": ["white", "सफेद", "सफ़ेद"],
    "GREY": ["grey", "gray", "स्लेटी", "ग्रे"],
    "ORANGE": ["orange", "नारंगी", "संतरी"],
    "PINK": ["pink", "गुलाबी"],
    "COLOUR": ["colour", "color", "colours", "colors", "रंग"],
}


def pretty(sign: str) -> str:
    return sign.replace("_", " ").lower()


def keyword_signs(text: str, vocab: list[str]) -> list[str]:
    """Signs mentioned in the text, in spoken order. Longer phrases win over words inside them."""
    low = text.lower()
    hits, taken = [], [False] * len(low)
    phrases = sorted(((p, s) for s in vocab for p in {pretty(s), *KEYWORDS.get(s, [])}),
                     key=lambda ps: -len(ps[0]))
    for phrase, sign in phrases:
        pat = re.escape(phrase) if not phrase.isascii() else rf"\b{re.escape(phrase)}\b"
        for m in re.finditer(pat, low):
            if not any(taken[m.start():m.end()]):
                hits.append((m.start(), sign))
                for i in range(m.start(), m.end()):
                    taken[i] = True
    return [s for _, s in sorted(hits)]


def template_sentence(signs: list[str], question: bool) -> str:
    """'HELLO HOW_ARE_YOU' → 'Hello how are you?'"""
    text = " ".join(pretty(s) for s in signs).strip()
    return (text[:1].upper() + text[1:] + ("?" if question else ".")) if text else ""
