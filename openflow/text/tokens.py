"""Minimal word/punctuation tokenizer shared by the text-cleanup passes.

Pure stdlib and dependency free so the test harness runs without installing
any of the audio / LLM extras.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*|\d+(?::\d+)?|[^\sA-Za-z0-9]")

SENTENCE_ENDERS = frozenset({".", "!", "?"})
CLAUSE_BREAKS = frozenset({",", ";", ":", "--", "—"})


@dataclass(slots=True)
class Token:
    text: str          # original surface form
    lower: str         # normalized for matching (curly quotes folded)
    is_word: bool

    @property
    def is_sentence_end(self) -> bool:
        return not self.is_word and self.text in SENTENCE_ENDERS

    @property
    def is_clause_break(self) -> bool:
        return not self.is_word and self.text in CLAUSE_BREAKS


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    for raw in _TOKEN_RE.findall(text):
        is_word = raw[0].isalnum()
        tokens.append(Token(raw, raw.lower().replace("’", "'"), is_word))
    return tokens


def detokenize(tokens: list[Token]) -> str:
    """Rejoin tokens with conventional English spacing."""
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if not out:
            out.append(tok.text)
            continue
        prev = tokens[i - 1]
        no_space = (
            (not tok.is_word and tok.text not in "([{“")
            or (not prev.is_word and prev.text in "([{“")
            or tok.text.startswith("'")
        )
        out.append(tok.text if no_space else " " + tok.text)
    return "".join(out).strip()


def words(tokens: list[Token]) -> list[Token]:
    return [t for t in tokens if t.is_word]


def match_phrase(tokens: list[Token], index: int, phrase: tuple[str, ...]) -> int:
    """If ``phrase`` starts at ``index`` (skipping punctuation), return the
    index just past it; otherwise return -1."""
    i = index
    for expected in phrase:
        while i < len(tokens) and not tokens[i].is_word:
            # A sentence boundary inside a phrase means it is not that phrase.
            if tokens[i].is_sentence_end:
                return -1
            i += 1
        if i >= len(tokens) or tokens[i].lower != expected:
            return -1
        i += 1
    return i
