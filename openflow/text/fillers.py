"""Filler-word stripping (PRD §3, instruction 2).

Hard fillers ("um", "you know") go anywhere. Soft fillers ("like", "right")
are real words in most positions, so they are only removed when they sit in
discourse-marker position -- comma-adjacent or clause-initial.
"""

from __future__ import annotations

from .pivots import CLAUSE_CONNECTIVES, HARD_FILLERS, SOFT_FILLERS
from .tokens import Token, detokenize, match_phrase, tokenize


def _is_discourse_position(tokens: list[Token], begin: int, end: int) -> bool:
    before = tokens[begin - 1] if begin > 0 else None
    after = tokens[end] if end < len(tokens) else None
    opens = before is None or before.is_clause_break or before.is_sentence_end
    closes = after is None or after.is_clause_break or after.is_sentence_end
    return opens or closes


def strip_fillers(text: str) -> tuple[str, list[str]]:
    """Return ``(cleaned_text, removed_phrases)``."""
    tokens = tokenize(text)
    removed: list[str] = []
    out: list[Token] = []
    i = 0

    while i < len(tokens):
        if not tokens[i].is_word:
            out.append(tokens[i])
            i += 1
            continue

        hit: tuple[int, str] | None = None
        for phrase in HARD_FILLERS:
            end = match_phrase(tokens, i, phrase)
            if end != -1 and (hit is None or end > hit[0]):
                hit = (end, " ".join(phrase))
        if hit is None:
            for phrase in SOFT_FILLERS:
                end = match_phrase(tokens, i, phrase)
                if end != -1 and _is_discourse_position(tokens, i, end):
                    if hit is None or end > hit[0]:
                        hit = (end, " ".join(phrase))

        if hit is None:
            out.append(tokens[i])
            i += 1
            continue

        end, phrase = hit
        removed.append(phrase)
        # Absorb the commas that were setting the filler off, so we do not
        # leave ", ," or a stranded comma behind.
        prev_comma = bool(out) and out[-1].is_clause_break
        next_comma = end < len(tokens) and tokens[end].is_clause_break
        if prev_comma and next_comma:
            follower = next((t for t in tokens[end + 1:] if t.is_word), None)
            keeps_comma = follower is not None and follower.lower in CLAUSE_CONNECTIVES
            if not keeps_comma:
                out.pop()
            end += 1
        elif next_comma and not out:
            end += 1
        elif prev_comma:
            out.pop()
        i = end

    return detokenize(out), removed
