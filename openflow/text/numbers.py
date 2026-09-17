"""Spoken numbers to digits, where writing would use digits.

Wispr Flow's Smart Formatting example: "Let's meet at seven period" ->
"Let's meet at 7." This runs after cleanup, never before it: the model's
containment guard forbids it adding a token that was not spoken, so "7" can
only appear once the model has finished editing "seven".

Converted:

* times       "at seven" -> "at 7", "seven thirty pm" -> "7:30 PM"
* percentages "fifty percent" -> "50%"
* money       "twenty dollars" -> "$20"
* any number of ten or more, and ordinals from tenth up:
              "twenty five people" -> "25 people", "the twenty first" -> "the 21st"

Left as words: one to nine in running prose ("one of the reasons", "two
options") -- the style-guide convention, and the reading that keeps idioms
like "no one" and "at one point" intact.
"""

from __future__ import annotations

import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000,
           "billion": 1_000_000_000}
_ORDINAL_UNITS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
    "hundredth": 100, "thousandth": 1_000,
}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES)

# Words after a bare hour that make it a clock time: "at seven tomorrow".
_TIME_FOLLOWERS = {
    "today", "tomorrow", "tonight", "yesterday", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "on", "and",
    "or", "sharp", "instead", "then", "please", "okay", "ok", "works",
}
_TIME_PREPOSITIONS = {"at", "by", "until", "till", "before", "after", "around", "from"}

_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|\d+(?::\d\d)?|[^\sA-Za-z\d]+|\s+")


def _suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _words(token: str) -> list[str]:
    """"twenty-five" is one token to the regex and two number words."""
    return token.lower().split("-")


def _can_extend(current: int, word: str) -> bool:
    """Whether ``word`` can continue a number whose running group is ``current``."""
    below_hundred = current % 100
    if word in _TENS or word in ("twentieth", "thirtieth", "fortieth", "fiftieth",
                                 "sixtieth", "seventieth", "eightieth", "ninetieth"):
        return below_hundred == 0
    if word in _UNITS or word in _ORDINAL_UNITS:
        teen_or_more = (word in _UNITS and _UNITS[word] >= 10) or (
            word in _ORDINAL_UNITS and 10 <= _ORDINAL_UNITS[word] < 100)
        if teen_or_more:
            return below_hundred == 0
        return below_hundred == 0 or (below_hundred >= 20 and below_hundred % 10 == 0)
    return True


class _Run:
    """A maximal run of number words starting at a token index."""

    def __init__(self, value: int, end: int, ordinal: bool) -> None:
        self.value, self.end, self.ordinal = value, end, ordinal


def _read_number(tokens: list[str], start: int) -> _Run | None:
    """Parse number words from ``tokens[start]``, skipping single spaces and a
    joining "and" ("one hundred and five"). Returns None if none start here."""
    total = current = 0
    seen = False
    ordinal = False
    i = start
    last_word_end = start
    while i < len(tokens):
        token = tokens[i]
        if token.isspace():
            i += 1
            continue
        parts = _words(token)
        if seen and not _can_extend(current, parts[0]):
            # "seven thirty" is two numbers (a clock time), not 37: English
            # never puts a tens word after a unit, or a unit after a unit.
            break
        if not all(p in _NUMBER_WORDS or p in _ORDINAL_UNITS for p in parts):
            if seen and token.lower() == "and" and i + 2 < len(tokens):
                nxt = _words(tokens[i + 2]) if tokens[i + 1].isspace() else []
                if nxt and all(p in _UNITS or p in _TENS for p in nxt):
                    i += 1
                    continue
            break
        for part in parts:
            if part in _ORDINAL_UNITS:
                current += _ORDINAL_UNITS[part] if part not in ("hundredth", "thousandth") \
                    else 0
                if part == "hundredth":
                    current = max(current, 1) * 100
                elif part == "thousandth":
                    total += max(current, 1) * 1000
                    current = 0
                ordinal = True
            elif part in _UNITS:
                current += _UNITS[part]
            elif part in _TENS:
                current += _TENS[part]
            else:
                scale = _SCALES[part]
                if scale == 100:
                    current = max(current, 1) * 100
                else:
                    total += max(current, 1) * scale
                    current = 0
        seen = True
        last_word_end = i + 1
        i += 1
        if ordinal:
            break
    if not seen:
        return None
    return _Run(total + current, last_word_end, ordinal)


def _next_word(tokens: list[str], index: int) -> tuple[str, int]:
    """The next non-space token at or after ``index``, lowercased, and its index."""
    while index < len(tokens) and tokens[index].isspace():
        index += 1
    if index >= len(tokens):
        return "", index
    return tokens[index].lower(), index


def _previous_word(out: list[str]) -> str:
    for token in reversed(out):
        if not token.isspace():
            return token.lower()
    return ""


def format_numbers(text: str) -> str:
    if not text:
        return text
    tokens = _WORD_RE.findall(text)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        first = _words(token)[0] if token[:1].isalpha() else ""
        if first not in _NUMBER_WORDS and first not in _ORDINAL_UNITS:
            out.append(token)
            i += 1
            continue

        run = _read_number(tokens, i)
        if run is None:
            out.append(token)
            i += 1
            continue

        follower, follower_at = _next_word(tokens, run.end)
        previous = _previous_word(out)
        value = run.value

        # Money: "twenty dollars" -> "$20".
        if not run.ordinal and follower in ("dollars", "dollar", "bucks"):
            out.append(f"${value}")
            i = follower_at + 1
            continue

        # Percentages: "fifty percent" -> "50%".
        if not run.ordinal and follower in ("percent", "per"):
            if follower == "percent":
                out.append(f"{value}%")
                i = follower_at + 1
                continue
            nxt, nxt_at = _next_word(tokens, follower_at + 1)
            if nxt == "cent":
                out.append(f"{value}%")
                i = nxt_at + 1
                continue

        # Clock times: "seven thirty pm", "at seven", "7 o'clock".
        if not run.ordinal and 1 <= value <= 12:
            minutes = None
            after = run.end
            minute_run = _read_number(tokens, follower_at) if follower else None
            if minute_run and not minute_run.ordinal and 0 <= minute_run.value <= 59 \
                    and (minute_run.value >= 10 or follower in ("oh", "o")):
                minutes = minute_run.value
                after = minute_run.end
                follower, follower_at = _next_word(tokens, after)
            meridiem = follower in ("am", "pm", "a.m.", "p.m.")
            oclock = follower in ("o'clock", "oclock")
            ends = follower == "" or not follower[:1].isalnum()
            timed = previous in _TIME_PREPOSITIONS and (
                ends or follower in _TIME_FOLLOWERS or minutes is not None)
            if meridiem or oclock or timed or (minutes is not None and previous in _TIME_PREPOSITIONS):
                clock = f"{value}:{minutes:02d}" if minutes is not None else str(value)
                if meridiem:
                    out.append(f"{clock} {follower.replace('.', '').upper()}")
                    i = follower_at + 1
                elif oclock:
                    out.append(f"{clock} o'clock")
                    i = follower_at + 1
                else:
                    out.append(clock)
                    i = after
                continue

        # Everything else: digits from ten up.
        if value >= 10:
            out.append(f"{value}{_suffix(value)}" if run.ordinal else str(value))
            i = run.end
            continue

        out.append(token)
        i += 1
    return "".join(out)
