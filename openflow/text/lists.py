"""Spoken numbered lists.

Wispr Flow's Smart Formatting example:

    "My top goals this week are one finish the report two send the presentation"
    -> "My top goals this week are:
        1. Finish the report
        2. Send the presentation"

Markers are counted in order -- one, two, three / first, second, third /
number one, number two -- and a list needs at least two of them.

The hard part is not finding markers but refusing sentences that merely
contain numbers: "I have one cat and two dogs" is not a list. Two guards do
nearly all of that work. The first marker must follow something that can
introduce a list (a colon, a sentence boundary, or a verb like "are"), and
no item may be a fragment that only makes sense as part of a number phrase
("one of two companies").
"""

from __future__ import annotations

import re

_CARDINALS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "firstly": 1, "secondly": 2, "thirdly": 3, "fourthly": 4, "fifthly": 5,
}

# Words that can sit directly before the first marker of a real list.
_INTRODUCERS = {
    "are", "is", "were", "was", "be", "include", "includes", "including",
    "following", "follows", "these", "those", "things", "steps", "goals",
    "points", "items", "reasons", "options", "tasks", "priorities", "list",
    "agenda", "todos", "todo", "need", "needs", "want", "plan", "here",
}

# An item starting with one of these belongs to a number phrase, not a list:
# "one of", "two or three", "one more", "one hundred".
_NOT_AN_ITEM_START = {
    "of", "or", "and", "more", "another", "by", "at", "to", "hundred",
    "thousand", "million", "times", "way", "point", "percent", "dollars",
    "minutes", "hours", "days", "weeks", "months", "years", "o'clock",
}

_TOKEN_RE = re.compile(r"\d+[.)]?|[A-Za-z]+(?:['’][A-Za-z]+)?|[^\sA-Za-z\d]")


class _Marker:
    def __init__(self, index: int, end: int, value: int) -> None:
        self.index, self.end, self.value = index, end, value


def _marker_at(tokens: list[str], i: int, family: str) -> _Marker | None:
    word = tokens[i].lower()
    # "number one" reads as the cardinal family.
    if family == "cardinal" and word == "number" and i + 1 < len(tokens):
        nxt = tokens[i + 1].lower()
        if nxt in _CARDINALS:
            return _Marker(i, i + 2, _CARDINALS[nxt])
    if family == "cardinal":
        if word in _CARDINALS:
            return _Marker(i, i + 1, _CARDINALS[word])
        digits = word.rstrip(".)")
        if digits.isdigit() and 1 <= int(digits) <= 20:
            return _Marker(i, i + 1, int(digits))
    if family == "ordinal" and word in _ORDINALS:
        return _Marker(i, i + 1, _ORDINALS[word])
    return None


def _is_word(token: str) -> bool:
    return token[:1].isalnum()


def _opens_list(tokens: list[str], index: int) -> bool:
    if index == 0:
        return True
    previous = tokens[index - 1]
    if previous in (":", ".", "!", "?", "\n", ","):
        return True
    return previous.lower() in _INTRODUCERS


def _item_words(tokens: list[str], start: int, end: int) -> list[str]:
    return [t for t in tokens[start:end] if _is_word(t)]


def _find_list(tokens: list[str]) -> tuple[str, list[_Marker]] | None:
    for family in ("cardinal", "ordinal"):
        for i in range(len(tokens)):
            first = _marker_at(tokens, i, family)
            if first is None or first.value != 1 or not _opens_list(tokens, i):
                continue
            markers = [first]
            j = first.end
            while j < len(tokens):
                marker = _marker_at(tokens, j, family)
                if marker is not None and marker.value == markers[-1].value + 1 \
                        and _item_words(tokens, markers[-1].end, j):
                    markers.append(marker)
                    j = marker.end
                    continue
                j += 1
            if len(markers) < 2:
                continue
            ok = True
            for k, marker in enumerate(markers):
                stop = markers[k + 1].index if k + 1 < len(markers) else len(tokens)
                words = _item_words(tokens, marker.end, stop)
                if not words or words[0].lower() in _NOT_AN_ITEM_START:
                    ok = False
                    break
            if ok:
                return family, markers
    return None


def _join(tokens: list[str]) -> str:
    out = ""
    for token in tokens:
        if not out or token in ".,;:!?)" or out.endswith(("(", "\n")) or token == "\n":
            out += token
        else:
            out += " " + token
    return out.strip()


def _clean_item(tokens: list[str]) -> str:
    text = _join(tokens).strip(" ,;:-—")
    text = text.rstrip(".")
    return text[:1].upper() + text[1:] if text else text


def format_lists(text: str) -> str:
    if not text:
        return text
    # Keep line breaks as tokens so an explicit "new line" survives.
    tokens = _TOKEN_RE.findall(text.replace("\n", " \n "))
    tokens = [t for t in tokens if t.strip() or t == "\n"]
    found = _find_list(tokens)
    if found is None:
        return text
    _family, markers = found

    intro = _join(tokens[:markers[0].index]).rstrip(" ,;:-—")
    lines = []
    tail = ""
    for k, marker in enumerate(markers):
        stop = markers[k + 1].index if k + 1 < len(markers) else len(tokens)
        item = tokens[marker.end:stop]
        if k + 1 == len(markers):
            # The last item runs to its first sentence end; anything spoken
            # after that is ordinary prose following the list.
            for n, token in enumerate(item):
                if token in (".", "!", "?", "\n") and _item_words(item, n + 1, len(item)):
                    keep = token if token in "!?" else ""
                    tail = _join(item[n + 1:])
                    item = item[:n] + ([keep] if keep else [])
                    break
        lines.append(f"{k + 1}. {_clean_item(item)}")

    out = ""
    if intro:
        out = intro if intro.endswith((".", "!", "?")) else intro + ":"
        out += "\n"
    out += "\n".join(lines)
    if tail:
        out += "\n\n" + tail[:1].upper() + tail[1:]
    return out
