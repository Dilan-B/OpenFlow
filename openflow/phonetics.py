"""Metaphone-style phonetic keys, for repairing misheard proper nouns.

Edit distance is the wrong tool for names. "Groq" and "grock" are two edits
apart but sound identical; "Kubernetes" and "cuber netties" are unrecognisable
by character overlap. A phonetic key collapses both to the same string, which
is exactly the failure mode ASR has with words it was never trained on.

This is a compact Metaphone: enough of the English rules to be useful, small
enough to read. Ambiguity is resolved toward under-matching, because a false
positive silently rewrites a word the speaker actually said.
"""

from __future__ import annotations

VOWELS = frozenset("AEIOU")

# Words common enough that a phonetic collision with a dictionary term is far
# more likely to be the real word than the term. Guards against a dictionary
# entry like "Kate" swallowing every spoken "cat".
COMMON_WORDS = frozenset("""
a an and are as at be been but by call can come could day did do does down
each find first for from get give go had has have he her here him his how
into is it its just know like long look made make many may me more most my
new no not now of on one only or other our out over people said say see she
so some take than that the their them then there these they thing think this
those time to two up use very want was way we well were what when where which
who will with word work would write year you your about after all also am an
any back because been before big both came come does don each even every
first found give going good great group hand help high home house just keep
kind last left life little live look made man mean men might much must name
need never next night number off old open own part place point put read right
room run same saw school seem set show side small sound still such sure tell
thought three through today together took turn under until us used want water
week went while white why word world year
""".split())


def metaphone(word: str, max_length: int = 8) -> str:
    """Return a phonetic key for a single word."""
    text = "".join(ch for ch in word.upper() if ch.isalpha())
    if not text:
        return ""

    # Silent leading clusters.
    for prefix, keep in (("AE", 1), ("GN", 1), ("KN", 1), ("PN", 1),
                         ("WR", 1), ("PS", 1)):
        if text.startswith(prefix):
            text = text[keep:]
            break
    if text.startswith("X"):
        text = "S" + text[1:]
    elif text.startswith("WH"):
        text = "W" + text[2:]

    key: list[str] = []
    length = len(text)
    i = 0
    while i < length and len(key) < max_length:
        ch = text[i]
        prev = text[i - 1] if i else ""
        nxt = text[i + 1] if i + 1 < length else ""
        after = text[i + 2] if i + 2 < length else ""

        # Collapse doubles (CC is meaningful, the rest are not).
        if ch == prev and ch != "C":
            i += 1
            continue

        if ch in VOWELS:
            # Vowels are only pronounced distinctly at the start.
            if i == 0:
                key.append(ch)
        elif ch == "B":
            if not (i == length - 1 and prev == "M"):
                key.append("B")
        elif ch == "C":
            if nxt == "I" and after == "A":
                key.append("X")
            elif nxt == "H":
                key.append("X")
                i += 1
            elif nxt in "IEY":
                key.append("S")
            else:
                key.append("K")
        elif ch == "D":
            if nxt == "G" and after in "EYI":
                key.append("J")
                i += 2
            else:
                key.append("T")
        elif ch in "FJLMNR":
            key.append(ch)
        elif ch == "G":
            if nxt == "H":
                if not (after and after not in VOWELS):
                    key.append("K")
                i += 1
            elif nxt == "N":
                pass                       # silent: "sign", "gnome"
            elif nxt in "IEY":
                key.append("J")
            else:
                key.append("K")
        elif ch == "H":
            # Pronounced only between a vowel and a following vowel.
            if prev in VOWELS and nxt not in VOWELS:
                pass
            else:
                key.append("H")
        elif ch == "K":
            if prev != "C":
                key.append("K")
        elif ch == "P":
            if nxt == "H":
                key.append("F")
                i += 1
            else:
                key.append("P")
        elif ch == "Q":
            key.append("K")
        elif ch == "S":
            if nxt == "H":
                key.append("X")
                i += 1
            elif nxt == "I" and after in "OA":
                key.append("X")
            else:
                key.append("S")
        elif ch == "T":
            if nxt == "I" and after in "OA":
                key.append("X")
            elif nxt == "H":
                key.append("0")            # voiceless "th"
                i += 1
            else:
                key.append("T")
        elif ch == "V":
            key.append("F")
        elif ch in "WY":
            if nxt in VOWELS:
                key.append(ch)
        elif ch == "X":
            key.append("K")
            if len(key) < max_length:
                key.append("S")
        elif ch == "Z":
            key.append("S")
        i += 1

    return "".join(key)


def phrase_key(words: list[str]) -> str:
    """Phonetic key for a multi-word term, spaces collapsed.

    "open flow" and "OpenFlow" produce the same key, which is what lets the
    repair pass rejoin a name the transcriber split apart.
    """
    return "".join(metaphone(word) for word in words)


def phonetically_matchable(term: str) -> bool:
    """Whether a dictionary term is safe to match phonetically.

    Short keys collide with ordinary English far too easily -- "Kate" and
    "cat" share a key -- so those terms stay on the exact/edit-distance path.
    """
    return len(phrase_key(term.split())) >= 4


def is_common_word(candidate: str) -> bool:
    return candidate.lower() in COMMON_WORDS
