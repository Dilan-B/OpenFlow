"""Capitalization and punctuation repair (PRD §3, instruction 3).

Deliberately conservative: we only ever *add* capitals that English requires
and never lowercase an existing token, so technical terms ("PostgreSQL",
"iOS", "gRPC") that Whisper got right survive untouched.
"""

from __future__ import annotations

import re

from .pivots import MONTHS, WEEKDAYS
from .tokens import Token, detokenize, tokenize

ALWAYS_CAPITAL = {"i", "i'm", "i'll", "i've", "i'd"}
PROPER_LOWER = WEEKDAYS | MONTHS

_MULTISPACE_RE = re.compile(r"[ \t]{2,}")

# Question inference. Speech has no punctuation, and a dictated question that
# ends in a period reads as brusque. These two bigram patterns are the
# high-precision ones -- "can we ...", "what is ..." are questions almost
# without exception, where a bare wh-word is not ("How to reset it" is a title,
# "What we need is time" is a statement).
_AUX = {
    "can", "could", "would", "should", "will", "shall", "do", "does", "did",
    "is", "are", "was", "were", "am", "have", "has", "had", "may", "might",
}
_SUBJECT = {
    "i", "we", "you", "he", "she", "they", "it", "there", "that", "this",
    "my", "your", "our", "his", "her", "their", "anyone", "anybody",
}
_WH = {"what", "when", "where", "why", "who", "whom", "whose", "which", "how"}
_PRONOUNS = {"i", "we", "you", "he", "she", "they", "it"}


def _capitalize(word: str) -> str:
    # A token carrying any uppercase was cased deliberately -- by the
    # transcriber, the autofix pass, or the user's dictionary. Forcing the
    # first letter would turn "iPhone" into "IPhone" and "gRPC" into "GRPC".
    if any(ch.isupper() for ch in word):
        return word
    return word[:1].upper() + word[1:]


def looks_like_question(text: str) -> bool:
    """True when a punctuation-less utterance is plainly a question.

    Only the final sentence is examined, since that is the one we are about to
    terminate.
    """
    tokens = tokenize(text)
    words: list[str] = []
    for tok in reversed(tokens):
        if tok.is_sentence_end and words:
            break
        if tok.is_word:
            words.append(tok.lower)
    words.reverse()
    if len(words) < 2:
        return False

    first, second = words[0], words[1]
    if first in _AUX and second in _SUBJECT:
        return True
    if first not in _WH:
        return False
    if second in _AUX:
        return True
    # "how many people are coming", "what time is it" -- an auxiliary shortly
    # after the wh-word. But a pronoun straight after it marks a free relative
    # acting as the subject ("what we need is time"), which is a statement.
    if second in _PRONOUNS:
        return False
    return any(word in _AUX for word in words[1:4])


def fix_punctuation(text: str, *, terminal: bool = True) -> str:
    tokens = tokenize(text)
    if not tokens:
        return ""

    at_sentence_start = True
    for tok in tokens:
        if not tok.is_word:
            if tok.is_sentence_end:
                at_sentence_start = True
            continue

        if at_sentence_start:
            tok.text = _capitalize(tok.text)
            at_sentence_start = False
        elif tok.lower in ALWAYS_CAPITAL:
            tok.text = _capitalize(tok.text)
        elif tok.lower in PROPER_LOWER and len(tok.lower) > 3:
            # Full weekday/month names only -- "mar" and "sat" are too risky.
            tok.text = _capitalize(tok.text)

    out = detokenize(tokens)
    out = _MULTISPACE_RE.sub(" ", out).strip()

    if terminal and out and out[-1] not in ".!?:;":
        # Only *infer* a question mark when the transcriber gave us no
        # punctuation of its own. If ASR already ended the sentence, its
        # judgement wins -- overriding it would contradict the PRD's own
        # worked example, which ends a "Can we ...' sentence with a period.
        out += "?" if looks_like_question(out) else "."
    return out


def normalize_whitespace(text: str) -> str:
    return _MULTISPACE_RE.sub(" ", text.replace("\r\n", "\n")).strip()


def strip_wrapping_quotes(text: str) -> str:
    """Undo an LLM that wrapped its answer in quotes despite instruction 4."""
    stripped = text.strip()
    pairs = (('"', '"'), ("'", "'"), ("“", "”"), ("`", "`"))
    for lead, trail in pairs:
        if len(stripped) > 1 and stripped.startswith(lead) and stripped.endswith(trail):
            return stripped[1:-1].strip()
    return stripped
