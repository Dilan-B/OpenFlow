"""Stutter and repeated-phrase collapse.

Speech repeats in ways writing does not: "the the the deploy", "I I think",
"can we can we meet". ASR transcribes all of it faithfully, and none of it
survives into text anyone wants to read.

This is deliberately conservative, because English does contain real doubled
words -- "had had", "that that", "very very". Two guards keep those:

  * a small allowlist of pairs that are genuinely idiomatic, and
  * intensifiers ("very", "really") are never collapsed, since repeating them
    is a rhetorical choice rather than a disfluency.

Longer repeated phrases are treated the opposite way: nobody says "can we meet
can we meet" on purpose, so a repeated run of 2+ words collapses without an
allowlist. Only immediately adjacent repeats count -- "the cat sat on the mat"
must never lose its second "the".
"""

from __future__ import annotations

from .tokens import Token, detokenize, tokenize

# Doubling these is normal English, not a stutter.
LEGITIMATE_DOUBLES: frozenset[tuple[str, str]] = frozenset({
    ("had", "had"),       # "he had had enough"
    ("that", "that"),     # "I think that that is wrong"
    ("is", "is"),         # "the thing is, is that..."  (colloquial but said)
    ("no", "no"),         # "no no, keep it"
    ("very", "very"),
    ("really", "really"),
    ("so", "so"),
    ("bye", "bye"),
    ("ha", "ha"),
})

# Repeating one of these is emphasis, so never collapse it even in a longer run.
INTENSIFIERS: frozenset[str] = frozenset({
    "very", "really", "so", "much", "many", "far", "way", "no", "yes",
})

MAX_PHRASE = 4       # longest repeated run we look for, in words


def _word_positions(tokens: list[Token]) -> list[int]:
    return [i for i, t in enumerate(tokens) if t.is_word]


def collapse_repetitions(text: str) -> tuple[str, list[str]]:
    """Return ``(cleaned_text, collapsed_phrases)``.

    Only adjacent repeats are removed, and only when nothing but whitespace
    sits between the two copies -- a comma or a full stop means the speaker
    said it twice on purpose ("no, no").
    """
    tokens = tokenize(text)
    positions = _word_positions(tokens)
    if len(positions) < 2:
        return text, []

    collapsed: list[str] = []
    drop: set[int] = set()

    # Longest phrases first: "can we can we" should collapse as a 2-word
    # repeat, not as two separate single-word ones.
    for size in range(MAX_PHRASE, 0, -1):
        w = 0
        while w + 2 * size <= len(positions):
            first = positions[w:w + size]
            second = positions[w + size:w + 2 * size]
            if any(i in drop for i in first + second):
                w += 1
                continue

            # Nothing but words may separate the two runs; punctuation between
            # them means it was intentional.
            span = range(first[0], second[-1] + 1)
            if any(not tokens[i].is_word and i not in drop for i in span):
                w += 1
                continue

            a = [tokens[i].lower for i in first]
            b = [tokens[i].lower for i in second]
            if a != b:
                w += 1
                continue
            if size == 1:
                if (a[0], b[0]) in LEGITIMATE_DOUBLES or a[0] in INTENSIFIERS:
                    w += 1
                    continue
            elif any(word in INTENSIFIERS for word in a):
                w += 1
                continue

            # Drop the *first* copy: the second is the one the speaker
            # finished, and it carries whatever punctuation follows.
            drop.update(first)
            collapsed.append(" ".join(a))
            w += size
        positions = [i for i in _word_positions(tokens) if i not in drop]

    if not drop:
        return text, []

    kept = [t for i, t in enumerate(tokens) if i not in drop]
    return detokenize(kept), collapsed
