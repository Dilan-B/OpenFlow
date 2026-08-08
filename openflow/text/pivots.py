"""Lexicons for false-start detection, filler stripping, and slot typing.

Everything here is data only -- no logic -- so the tables can be tuned from the
test harness without touching the algorithm in ``stem_removal.py``.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Pivot phrases: the spoken signal that the speaker just retracted something.
# --------------------------------------------------------------------------
# Each entry is a tuple of lowercase word tokens (punctuation is stripped
# before matching). Longer phrases are matched first, so order does not matter.
STRONG_PIVOTS: tuple[tuple[str, ...], ...] = (
    ("or", "actually"),
    ("or", "rather"),
    ("no", "wait"),
    ("wait", "no"),
    ("wait", "sorry"),
    ("sorry", "i", "mean"),
    ("sorry", "i", "meant"),
    ("i", "mean"),
    ("i", "meant"),
    ("let", "me", "rephrase", "that"),
    ("let", "me", "rephrase"),
    ("let", "me", "start", "over"),
    ("scratch", "that"),
    ("strike", "that"),
    ("correction",),
    ("actually", "no"),
    ("no", "sorry"),
    ("or", "maybe"),
)

# Words that only act as a pivot when they sit on a clause boundary -- i.e.
# preceded by a comma / sentence start and (usually) followed by a comma.
# "actually" mid-clause ("I actually like it") must never trigger a retraction.
WEAK_PIVOTS: tuple[tuple[str, ...], ...] = (
    ("actually",),
    ("rather",),
    ("meanwhile",),
    ("wait",),
    ("sorry",),
)

# --------------------------------------------------------------------------
# Fillers
# --------------------------------------------------------------------------
# Always deleted, anywhere they appear.
HARD_FILLERS: tuple[tuple[str, ...], ...] = (
    ("um",),
    ("umm",),
    ("uh",),
    ("uhh",),
    ("erm",),
    ("hmm",),
    ("mhm",),
    ("you", "know"),
    ("so", "yeah"),
    ("i", "guess", "you", "know"),
)

# Deleted only in discourse-marker position (comma-adjacent or clause-initial),
# because they are legitimate words elsewhere: "I like it", "turn right".
SOFT_FILLERS: tuple[tuple[str, ...], ...] = (
    ("like",),
    ("right",),
    ("basically",),
    ("literally",),
    ("i", "mean"),
)

# --------------------------------------------------------------------------
# Slot typing -- used to splice a fragment replacement onto the retained head.
# "Let's meet Tuesday at 5, actually Friday at 3."
#   tail slots = {date, time}  ->  drop "Tuesday" and "at 5" from the head.
# --------------------------------------------------------------------------
# When a filler is set off by commas on both sides we normally delete both.
# But if the following word connects two clauses, the leading comma was doing
# real work ("...on the iOS build, um, before the cutover") -- keep one.
CLAUSE_CONNECTIVES = {
    "before", "after", "but", "so", "and", "then", "because", "which", "while",
    "until", "though", "although", "however", "unless", "since", "whereas",
}

WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mon", "tue", "tues", "wed", "thu", "thurs", "fri", "sat", "sun",
}

MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

RELATIVE_DAYS = {"today", "tomorrow", "tonight", "yesterday", "monday"}

TIME_PREPOSITIONS = {"at", "on", "in", "by", "for", "until", "till", "before", "after"}

MERIDIEMS = {"am", "pm", "a.m", "p.m", "oclock", "o'clock"}

# Tokens that can legitimately open a fresh independent clause. If the text
# after a pivot starts with one of these, the speaker restarted the whole
# sentence rather than patching a slot in it.
CLAUSE_STARTERS = {
    "i", "we", "you", "he", "she", "they", "it", "there", "that", "this",
    "let", "lets", "let's", "can", "could", "would", "should", "will", "shall",
    "do", "does", "did", "is", "are", "was", "were", "am", "please", "the",
    "my", "our", "your", "his", "her", "their", "make", "send", "give", "put",
    "how", "what", "when", "where", "why", "who",
}
