"""Tell a filler or pivot phrase apart from the same words used literally.

"you know" and "I mean" are the two phrases that are both a disfluency and
ordinary English. The filler and false-start passes both need the same answer,
and getting it wrong in the delete direction is the expensive mistake:
"you know the answer already" becoming "The answer already." is a broken
sentence, where a surviving "you know" is only an untidy one.
"""

from __future__ import annotations

from .pivots import KNOW_LEADERS, KNOW_OBJECTS, MEAN_OBJECTS
from .tokens import Token

_MEAN_PHRASES = frozenset({
    ("i", "mean"), ("i", "meant"), ("sorry", "i", "mean"), ("sorry", "i", "meant"),
})


def _next_word(tokens: list[Token], end: int) -> Token | None:
    """The word directly after the phrase, or None if punctuation intervenes.

    A comma after the phrase ("you know, it's flaky") marks an aside, which is
    the filler reading -- so only an unpunctuated follower counts.
    """
    if end < len(tokens) and tokens[end].is_word:
        return tokens[end]
    return None


def _previous_word(tokens: list[Token], begin: int) -> Token | None:
    if begin > 0 and tokens[begin - 1].is_word:
        return tokens[begin - 1]
    return None


def _starts_sentence(tokens: list[Token], begin: int) -> bool:
    return begin == 0 or tokens[begin - 1].is_sentence_end


def is_literal_use(tokens: list[Token], begin: int, end: int,
                   phrase: tuple[str, ...]) -> bool:
    """True when ``tokens[begin:end]`` (matching ``phrase``) is meant literally."""
    following = _next_word(tokens, end)

    if phrase in _MEAN_PHRASES:
        return following is not None and following.lower in MEAN_OBJECTS

    if phrase == ("you", "know"):
        if following is None:
            return False
        previous = _previous_word(tokens, begin)
        if previous is not None and previous.lower in KNOW_LEADERS:
            return True
        return _starts_sentence(tokens, begin) and following.lower in KNOW_OBJECTS

    return False
