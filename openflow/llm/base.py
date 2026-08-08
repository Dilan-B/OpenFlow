"""Provider interface + shared output sanitation."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from ..text.punctuation import normalize_whitespace, strip_wrapping_quotes

_PREAMBLE_RE = re.compile(
    r"^\s*(here(?:'s| is)[^:\n]*:|sure[,!.][^\n]*|polished(?: transcription)?:|output:|"
    r"cleaned(?: up)? text:)\s*",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Contractions and splits the prompt legitimately produces from the input.
# "i'll" <- "i will", "cant" <- "can't", and the reverse of both.
_ALLOWED_NEW_WORDS = frozenset({
    "a", "an", "the", "i", "is", "am", "are", "was", "were", "be", "will",
    "would", "not", "have", "has", "had", "do", "does", "did", "to", "of",
    "and", "s", "t", "re", "ve", "ll", "d", "m",
})


class ProviderError(RuntimeError):
    """Backend unavailable, over quota, or returned something unusable."""


@runtime_checkable
class Provider(Protocol):
    name: str
    is_local: bool

    def available(self) -> bool: ...

    def complete(self, system: str, user: str, *, strict: bool = True) -> str: ...


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower().replace("’", "'"))


def _expand(word: str) -> set[str]:
    """Both halves of a contraction, so "i'll" satisfies "i" and "will"."""
    forms = {word}
    if "'" in word:
        forms.update(part for part in word.split("'") if part)
        forms.add(word.replace("'", ""))
    return forms


def check_containment(output: str, original: str) -> list[str]:
    """Return output words that do not appear in the input.

    ASR post-correction models that rewrite rather than edit will substitute
    similar-sounding entities -- the documented failure is "I like algorithms"
    becoming "I like Al Gore" (arXiv 2601.15397). Length checks miss that
    entirely, because the substitution is the same size as the original.

    Cleanup is a *deleting* operation: every word in the output should already
    be in the input. Contractions and a small closed class of function words
    are exempt, since instruction 3 (fix formatting) legitimately produces
    them.
    """
    source: set[str] = set()
    for word in _tokens(original):
        source |= _expand(word)

    unseen: list[str] = []
    for word in _tokens(output):
        if word in _ALLOWED_NEW_WORDS:
            continue
        if _expand(word) & source:
            continue
        unseen.append(word)
    return unseen


def sanitize(output: str, *, original: str, strict: bool = True) -> str:
    """Enforce PRD instruction 4 defensively.

    Models -- especially small local ones -- leak preambles and quote wrapping
    no matter what the prompt says. In strict mode (dictation cleanup) we add
    two structural checks that no prompt can be talked out of: the output may
    not grow, and it may not contain words the speaker never said. Transforms
    (deliberate rewrites) run with ``strict=False``.
    """
    text = normalize_whitespace(output)
    text = _PREAMBLE_RE.sub("", text)
    text = strip_wrapping_quotes(text)
    text = text.strip()

    if not text:
        raise ProviderError("empty completion")

    if not strict:
        return text

    # Cleanup only ever removes words. A 30% grow-margin absorbs added
    # punctuation and expanded contractions without letting a hallucinated
    # paragraph through.
    if len(text) > max(40, int(len(original.strip()) * 1.3)):
        raise ProviderError("completion longer than input; model rewrote instead of edited")

    invented = check_containment(text, original)
    if invented:
        raise ProviderError(
            f"completion introduced words the speaker did not say: {invented[:5]}"
        )

    return text
