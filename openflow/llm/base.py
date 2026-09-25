"""Provider interface + shared output sanitation."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
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

    def complete(self, system: str, user: str, *, strict: bool = True,
                 allowed: frozenset[str] = frozenset()) -> str: ...


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower().replace("’", "'"))


def _expand(word: str) -> set[str]:
    """Both halves of a contraction, so "i'll" satisfies "i" and "will"."""
    forms = {word}
    if "'" in word:
        forms.update(part for part in word.split("'") if part)
        forms.add(word.replace("'", ""))
    return forms


def check_containment(output: str, original: str,
                      allowed: frozenset[str] | set[str] = frozenset()) -> list[str]:
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
    source: set[str] = set(allowed)
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


# A repaired mishearing sounds like what the recognizer wrote, and nearly
# always shares its spelling: "except" -> "accept", "supported" ->
# "supportive", "buy" -> "by". A wholesale substitution does not: "algorithms"
# -> "Al Gore". Beyond sound-alikes, one unrelated word per this many spoken
# words covers a dropped "it" or "on" without admitting an invented clause.
NEAR_FORM_RATIO = 0.6
NEAR_FORM_PREFIX = 5
WORDS_PER_FREE_REPAIR = 12


def _near_form(word: str, source: set[str]) -> bool:
    for seen in source:
        if len(word) >= NEAR_FORM_PREFIX and word[:NEAR_FORM_PREFIX] == seen[:NEAR_FORM_PREFIX]:
            return True
        if SequenceMatcher(None, word, seen).ratio() >= NEAR_FORM_RATIO:
            return True
    return False


def unexplained_words(output: str, original: str,
                      allowed: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """New words that are not a sound-alike repair of anything the speaker
    said -- what check_containment finds, minus plausible mishearing fixes."""
    invented = check_containment(output, original, allowed)
    if not invented:
        return []
    source = {w for word in _tokens(original) for w in _expand(word)}
    return [word for word in invented if not _near_form(word, source)]


def sanitize(output: str, *, original: str, strict: bool = True,
             allowed: frozenset[str] | set[str] = frozenset()) -> str:
    """Enforce PRD instruction 4 defensively.

    Models -- especially small local ones -- leak preambles and quote wrapping
    no matter what the prompt says. In strict mode (dictation cleanup) we add
    two structural checks that no prompt can be talked out of: the output may
    not grow, and it may not contain words the speaker never said beyond
    sound-alike repairs and a small per-dictation allowance. Transforms
    (deliberate rewrites) run with ``strict=False``.

    ``allowed`` holds words the model may write although the speaker did not
    say them in that form: names spelled the way the screen spells them, the
    files and identifiers of the open project. Everything else still has to
    come from the transcript.
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
    # File tags and identifiers ("@auth.ts", "getUserName") are longer than
    # the words they replace, so the margin grows with what was allowed in.
    grow = 1.3 if not allowed else 1.5
    if len(text) > max(40, int(len(original.strip()) * grow)):
        raise ProviderError("completion longer than input; model rewrote instead of edited")

    # Cleanup also repairs what the recognizer misheard, so a new word is
    # accepted when it sounds like one the speaker said, and a few others per
    # dictation for dropped small words. More than that is a rewrite.
    invented = unexplained_words(text, original, allowed)
    budget = len(_tokens(original)) // WORDS_PER_FREE_REPAIR
    if len(invented) > budget:
        raise ProviderError(
            f"completion introduced words the speaker did not say: {invented[:5]}"
        )

    return text
