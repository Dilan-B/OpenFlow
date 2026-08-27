"""False-start / sentence-stem removal.

The PRD delegates this to the LLM layer. We keep a deterministic implementation
anyway for three reasons:

1. It is the only part of the pipeline that can be unit tested against a golden
   corpus, so it defines the *contract* the LLM prompt must also satisfy
   (``tests/harness.py`` scores any cleaner against the same cases).
2. It runs in microseconds, so the overlay can show corrected text immediately
   and the LLM pass becomes an optional refinement rather than a latency wall.
3. It is the fallback when both Ollama and the cloud tier are unavailable.

Algorithm
---------
For each pivot phrase ("or actually", "wait no", ...) we split the enclosing
sentence into HEAD (the retracted premise) and TAIL (the replacement), then
decide how much of HEAD the speaker actually threw away:

  A. Re-anchor  -- TAIL repeats HEAD's opening words ("can we meet up on ...").
                   The speaker restarted the clause: drop HEAD entirely.
  B. Restart    -- TAIL opens a fresh independent clause (subject/aux/imperative)
                   and is long enough to stand alone: drop HEAD entirely.
  C. Slot patch -- TAIL is a fragment ("Friday at 3"). Remove only the tokens at
                   the end of HEAD that fill the same slots (date, time, number,
                   name), together with any preposition attached to them.
  D. Bail out   -- ambiguous: keep both sides and drop just the pivot phrase.

Rule D is the safety net: when in doubt we prefer a slightly verbose
transcript over deleting words the user meant to keep. It reports strategy
"keep-both", distinct from the "pivot-only" case where one side was empty and
there was nothing to be ambiguous about -- callers use that difference to tell
"I was unsure" from "there was nothing to do".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pivots import (
    CLAUSE_STARTERS,
    MERIDIEMS,
    MONTHS,
    RELATIVE_DAYS,
    STRONG_PIVOTS,
    TIME_PREPOSITIONS,
    WEAK_PIVOTS,
    WEEKDAYS,
)
from .tokens import Token, detokenize, match_phrase, tokenize

_TIME_RE = re.compile(r"^\d{1,2}(:\d{2})?$")
_MIN_REANCHOR_PREFIX = 2   # shared opening words needed to call it a restart
_MIN_RESTART_WORDS = 3     # words a fresh clause needs before we trust it


@dataclass(slots=True)
class Retraction:
    """What a single pivot removed -- surfaced for debugging and the harness."""

    pivot: str
    removed: str
    strategy: str


# ---------------------------------------------------------------------------
# Pivot detection
# ---------------------------------------------------------------------------

def _find_pivot(tokens: list[Token], start: int) -> tuple[int, int, str] | None:
    """Return ``(begin, end, phrase)`` for the first pivot at or after ``start``."""
    for i in range(start, len(tokens)):
        if not tokens[i].is_word:
            continue

        best: tuple[int, str] | None = None
        for phrase in STRONG_PIVOTS:
            end = match_phrase(tokens, i, phrase)
            if end != -1 and (best is None or end > best[0]):
                best = (end, " ".join(phrase))
        if best is not None:
            return i, best[0], best[1]

        for phrase in WEAK_PIVOTS:
            end = match_phrase(tokens, i, phrase)
            if end == -1 or not _weak_pivot_is_boundary(tokens, i, end):
                continue
            return i, end, " ".join(phrase)
    return None


def _weak_pivot_is_boundary(tokens: list[Token], begin: int, end: int) -> bool:
    """A bare "actually" only retracts when it is set off as its own aside."""
    before = tokens[begin - 1] if begin > 0 else None
    after = tokens[end] if end < len(tokens) else None

    opens = before is None or before.is_clause_break or before.is_sentence_end
    closes = after is not None and (after.is_clause_break or after.is_sentence_end)

    # ", actually," -- unambiguous. ", actually X" -- only if a clause precedes.
    if opens and closes:
        return begin > 0
    if opens:
        return begin > 0 and after is not None and after.is_word
    return False


# ---------------------------------------------------------------------------
# Sentence and clause boundaries
# ---------------------------------------------------------------------------

def _sentence_bounds(tokens: list[Token], index: int) -> tuple[int, int]:
    start = 0
    for i in range(index - 1, -1, -1):
        if tokens[i].is_sentence_end:
            start = i + 1
            break
    end = len(tokens)
    for i in range(index, len(tokens)):
        if tokens[i].is_sentence_end:
            end = i
            break
    return start, end


def _strip_edges(tokens: list[Token]) -> list[Token]:
    lo, hi = 0, len(tokens)
    while lo < hi and not tokens[lo].is_word and not tokens[lo].is_sentence_end:
        lo += 1
    while hi > lo and not tokens[hi - 1].is_word and not tokens[hi - 1].is_sentence_end:
        hi -= 1
    return tokens[lo:hi]


# ---------------------------------------------------------------------------
# Slot typing
# ---------------------------------------------------------------------------

def _slot_of(tok: Token, prev: Token | None) -> str | None:
    if not tok.is_word:
        return None
    low = tok.lower
    if low in WEEKDAYS or low in MONTHS or low in RELATIVE_DAYS:
        return "date"
    if low in MERIDIEMS:
        return "time"
    if _TIME_RE.match(low):
        # "at 5" / "5:30" reads as a time; a bare number stays a number.
        if ":" in low or (prev is not None and prev.lower in {"at", "by", "till", "until"}):
            return "time"
        return "number"
    if low.isdigit():
        return "number"
    if tok.text[:1].isupper() and not tok.text.isupper():
        return "name"
    return None


def _slots_in(tokens: list[Token]) -> set[str]:
    found: set[str] = set()
    prev: Token | None = None
    for tok in tokens:
        slot = _slot_of(tok, prev)
        if slot:
            found.add(slot)
        if tok.is_word:
            prev = tok
    return found


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_MAX_SEAM_OVERLAP = 3


def _splice(prefix: list[Token], tail: list[Token], suffix: list[Token]) -> list[Token]:
    """Drop words repeated across a splice seam.

    A replacement often restates part of what it replaces -- "ask the Docker
    team, or actually the Kubernetes team" repeats both "the" and "team". Left
    alone the splice reads "ask the the Kubernetes team team".
    """
    def words(tokens: list[Token]) -> list[str]:
        return [t.lower for t in tokens if t.is_word]

    prefix_words, tail_words, suffix_words = words(prefix), words(tail), words(suffix)

    # Trailing words of the prefix that the tail already opens with.
    for n in range(min(_MAX_SEAM_OVERLAP, len(prefix_words), len(tail_words)), 0, -1):
        if prefix_words[-n:] == tail_words[:n]:
            prefix = _drop_last_words(prefix, n)
            break

    # Leading words of the suffix that the tail already closes with.
    for n in range(min(_MAX_SEAM_OVERLAP, len(suffix_words), len(tail_words)), 0, -1):
        if suffix_words[:n] == tail_words[-n:]:
            suffix = _drop_first_words(suffix, n)
            break

    return _strip_edges(prefix) + tail + suffix


def _drop_last_words(tokens: list[Token], count: int) -> list[Token]:
    seen = 0
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i].is_word:
            seen += 1
            if seen == count:
                return tokens[:i]
    return []


def _drop_first_words(tokens: list[Token], count: int) -> list[Token]:
    seen = 0
    for i, token in enumerate(tokens):
        if token.is_word:
            seen += 1
            if seen == count:
                return tokens[i + 1:]
    return []


def _join(head: list[Token], tail: list[Token], break_before: bool) -> list[Token]:
    """Concatenate the two halves when we decline to delete either of them,
    restoring the comma that the pivot phrase was sitting behind."""
    if head and tail and break_before:
        return head + [Token(",", ",", False)] + tail
    return head + tail


def _shared_prefix_len(head: list[Token], tail: list[Token]) -> int:
    hw = [t.lower for t in head if t.is_word]
    tw = [t.lower for t in tail if t.is_word]
    n = 0
    for a, b in zip(hw, tw):
        if a != b:
            break
        n += 1
    return n


# Pivots that announce a full restatement. After "scratch that" the speaker is
# starting the sentence again; after a bare "or actually" they are as likely to
# be swapping one noun phrase for another.
FULL_RESTART_PIVOTS = frozenset({
    "let me rephrase that", "let me rephrase", "let me start over",
    "scratch that", "strike that", "correction",
    "sorry i mean", "sorry i meant", "i mean", "i meant",
})

# Determiners open a noun phrase far more often than an independent clause, so
# they only count as a restart when the pivot already promised one.
_DETERMINERS = frozenset({
    "the", "my", "our", "your", "his", "her", "their", "this", "that",
})


def _is_restart(tail: list[Token], pivot: str) -> bool:
    tw = [t for t in tail if t.is_word]
    if len(tw) < _MIN_RESTART_WORDS:
        return False
    if tw[0].lower not in CLAUSE_STARTERS:
        return False
    if tw[0].lower in _DETERMINERS and pivot not in FULL_RESTART_PIVOTS:
        # "...ask the Docker team, or actually the Kubernetes team" replaces a
        # noun phrase; dropping the whole head would delete the sentence.
        return False
    return True


def _slot_patch(
    head: list[Token], tail: list[Token]
) -> tuple[list[Token], list[Token]] | None:
    """Locate the slots in HEAD that TAIL overwrites.

    Returns ``(prefix, suffix)`` -- the parts of HEAD that survive on either
    side of the replaced span -- or ``None`` when no slot lines up.
    """
    target = _slots_in(tail)
    if not target:
        return None

    word_positions = [i for i, t in enumerate(head) if t.is_word]
    cut = len(head)
    span_end = len(head)   # first index after the rightmost token we remove
    removed_any = False
    p = len(word_positions) - 1
    while p >= 0:
        i = word_positions[p]
        prev_word = head[word_positions[p - 1]] if p > 0 else None
        slot = _slot_of(head[i], prev_word)
        # Sentence-initial capitalization is not evidence of a proper noun.
        if slot == "name" and p == 0:
            slot = None
        if slot in target:
            if not removed_any:
                span_end = i + 1
            cut = i
            removed_any = True
            # Pull an attached preposition along: "at 5" -> drop "at" too.
            if prev_word is not None and prev_word.lower in TIME_PREPOSITIONS:
                cut = word_positions[p - 1]
                p -= 2
            else:
                p -= 1
            continue
        if removed_any:
            break
        p -= 1

    if not removed_any:
        return None
    # The replacement goes back where the old slot was, so a trailing noun
    # survives: "Order 12 units, or actually 20." -> "Order 20 units."
    return _strip_edges(head[:cut]), _strip_edges(head[span_end:])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remove_false_starts(text: str) -> tuple[str, list[Retraction]]:
    """Strip retracted premises and their pivot phrases from ``text``.

    Returns the rewritten text plus one :class:`Retraction` per pivot handled.
    """
    if not text or not text.strip():
        return "", []

    tokens = tokenize(text)
    retractions: list[Retraction] = []
    cursor = 0

    while True:
        found = _find_pivot(tokens, cursor)
        if found is None:
            break
        begin, end, phrase = found
        sent_start, sent_end = _sentence_bounds(tokens, begin)

        head = _strip_edges(tokens[sent_start:begin])
        tail = _strip_edges(tokens[end:sent_end])
        break_before = begin > 0 and tokens[begin - 1].is_clause_break

        if not any(t.is_word for t in tail) or not any(t.is_word for t in head):
            # Nothing on one side -- the pivot is just noise.
            kept, strategy, dropped = _join(head, tail, break_before), "pivot-only", []
        elif _shared_prefix_len(head, tail) >= _MIN_REANCHOR_PREFIX:
            kept, strategy, dropped = tail, "re-anchor", head
        elif _is_restart(tail, phrase):
            kept, strategy, dropped = tail, "restart", head
        else:
            patched = _slot_patch(head, tail)
            if patched is not None:
                prefix, suffix = patched
                kept = _splice(prefix, tail, suffix)
                strategy = "slot-patch"
                dropped = head[len(prefix):len(head) - len(suffix)]
            else:
                # Rule D. Both sides carry content and nothing told us which
                # the speaker meant, so keep everything and drop only the
                # pivot. Named apart from "pivot-only" because this is the
                # ambiguous case -- CleanResult.uncertain keys off it to
                # decide whether an LLM second opinion is worth the latency.
                kept, strategy, dropped = _join(head, tail, break_before), "keep-both", []

        retractions.append(
            Retraction(pivot=phrase, removed=detokenize(dropped), strategy=strategy)
        )
        tokens = tokens[:sent_start] + kept + tokens[sent_end:]
        # Rescan from the start of the rewritten sentence: a speaker can
        # correct the same clause twice ("at 4, or actually 5, wait no, 6").
        # Every branch deletes the pivot, so this always makes progress.
        cursor = sent_start

    return detokenize(tokens), retractions
