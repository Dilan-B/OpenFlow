"""Spoken punctuation and layout commands.

"send the report by friday period new paragraph second thing..."

Multi-word commands ("question mark", "new paragraph") are unambiguous and
apply anywhere. Single-word commands ("period", "comma") are real English
words -- "that period of time" -- so they only apply at the very end of the
utterance, where dictating punctuation is the overwhelmingly likely intent.
"""

from __future__ import annotations

import re

# Applied anywhere. Order matters: longer phrases first.
_ANYWHERE: tuple[tuple[str, str], ...] = (
    (r"new\s+paragraph", "\n\n"),
    (r"new\s+line", "\n"),
    (r"next\s+line", "\n"),
    (r"question\s+mark", "?"),
    (r"exclamation\s+(?:mark|point)", "!"),
    (r"full\s+stop", "."),
    (r"semi\s*colon", ";"),
    (r"open\s+quote", "“"),
    (r"close\s+quote", "”"),
)

# Applied only when they are the final word(s) spoken.
_AT_END: tuple[tuple[str, str], ...] = (
    (r"period", "."),
    (r"comma", ","),
    (r"colon", ":"),
    (r"dash", "—"),
)

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?—])")
_PUNCT_RUN = re.compile(r"([.,;:!?])\s*([.,;:!?])")
_SPACE_AROUND_NL = re.compile(r"[ \t]*\n[ \t]*")


def apply_spoken_punctuation(text: str) -> str:
    if not text:
        return text

    for pattern, mark in _ANYWHERE:
        text = re.sub(rf"[,.]?\s*\b{pattern}\b[,.]?", f" {mark} ", text, flags=re.IGNORECASE)

    for pattern, mark in _AT_END:
        text = re.sub(rf"\s+\b{pattern}\b\s*[.!?]?\s*$", mark, text, flags=re.IGNORECASE)

    # Attach punctuation to the preceding word and collapse doubles that come
    # from "did you send it, question mark" style dictation.
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _PUNCT_RUN.sub(r"\2", text)
    text = _SPACE_AROUND_NL.sub("\n", text)
    return text.strip()
