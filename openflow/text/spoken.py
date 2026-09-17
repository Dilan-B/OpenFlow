"""Spoken punctuation and layout commands.

"send the report by friday period new paragraph second thing..."

Wispr Flow's documented vocabulary, split by how safe each command is to read
literally:

* Multi-word commands ("question mark", "new paragraph", "em dash") are
  unambiguous and apply anywhere.
* Single words that are rarely said for their own sake ("semicolon",
  "ellipsis") also apply anywhere.
* Single words that are ordinary English -- "that period of time", "a comma
  splice", "dash to the car" -- only apply at the very end of the utterance,
  where dictating punctuation is the overwhelmingly likely intent.

Symbols that are not sentence punctuation ("at sign", "tilde") live in
``autofix.SPOKEN_SYMBOLS`` and run after the token pipeline, which would
otherwise re-space them.
"""

from __future__ import annotations

import re

# Applied anywhere. Order matters: longer phrases first.
_ANYWHERE: tuple[tuple[str, str], ...] = (
    # Layout. "skip a line" leaves a blank line, like a paragraph break.
    (r"start\s+a\s+new\s+paragraph", "\n\n"),
    (r"new\s+paragraph", "\n\n"),
    (r"skip\s+a\s+line", "\n\n"),
    (r"new\s+line", "\n"),
    (r"next\s+line", "\n"),
    (r"line\s+break", "\n"),
    # Punctuation.
    (r"question\s+mark", "?"),
    (r"exclamation\s+(?:mark|point)", "!"),
    (r"full\s+stop", "."),
    (r"semi\s*-?\s*colon", ";"),
    (r"em\s*-?\s*dash", "—"),
    (r"dot\s+dot\s+dot", "…"),
    (r"ellipsis", "…"),
    (r"open\s+quote", "“"),
    (r"close\s+quote", "”"),
    (r"end\s+quote", "”"),
)

# "quotation mark" is one command for both ends: the first opens, the next
# closes. Resolved separately because it depends on how many came before.
_QUOTATION_MARK = re.compile(r"[,.]?\s*\bquotation\s+marks?\b[,.]?", re.IGNORECASE)

# Applied only when they are the final word(s) spoken.
_AT_END: tuple[tuple[str, str], ...] = (
    (r"period", "."),
    (r"comma", ","),
    (r"colon", ":"),
    (r"dash", "—"),
)

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?—…”])")
_SPACE_AFTER_OPEN = re.compile(r"([“])\s+")
_SPACE_AROUND_DASH = re.compile(r"\s*—\s*")
_PUNCT_RUN = re.compile(r"([.,;:!?])\s*([.,;:!?])")
_SPACE_AROUND_NL = re.compile(r"[ \t]*\n[ \t]*")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def _resolve_quotation_marks(text: str) -> str:
    count = 0

    def _mark(_match: re.Match) -> str:
        nonlocal count
        count += 1
        return " “" if count % 2 else "” "

    return _QUOTATION_MARK.sub(_mark, text)


def apply_spoken_punctuation(text: str) -> str:
    if not text:
        return text

    for pattern, mark in _ANYWHERE:
        text = re.sub(rf"[,.]?\s*\b{pattern}\b[,.]?", f" {mark} ", text, flags=re.IGNORECASE)
    text = _resolve_quotation_marks(text)

    for pattern, mark in _AT_END:
        text = re.sub(rf"\s+\b{pattern}\b\s*[.!?]?\s*$", mark, text, flags=re.IGNORECASE)

    # Attach punctuation to the preceding word and collapse doubles that come
    # from "did you send it, question mark" style dictation.
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    # An em dash joins words without spaces: "fast—really fast".
    text = _SPACE_AROUND_DASH.sub("—", text)
    text = _PUNCT_RUN.sub(r"\2", text)
    text = _SPACE_AROUND_NL.sub("\n", text)
    # Each command was replaced as " mark ", so neighbours can leave doubles.
    text = _MULTISPACE.sub(" ", text)
    return text.strip()
