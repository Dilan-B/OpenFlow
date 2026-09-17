"""When a capital letter is only sentence case, and safe to lower.

Two Wispr Flow behaviours lowercase a first word: Very Casual style ("no
caps"), and dictating into the middle of a sentence. Both must leave names
alone -- "sarah is here" is a typo, not a style -- and there is no dictionary
here to consult.

So this errs one way. A word is lowered only with positive evidence that it is
an ordinary word: it is a common sentence opener, it carries a suffix names
almost never do, or the same text uses it in lowercase elsewhere. Anything
else keeps its capital. The cost of being wrong in that direction is a capital
letter nobody minds; the cost the other way is a misspelled name.
"""

from __future__ import annotations

import re

# Words that open sentences in dictated messages and are never names. "will"
# and "may" are deliberately absent: both are also first names.
OPENERS = frozenset("""
a about above absolutely across actually add after afternoon again against ago
agree ah ahead all almost alright already also although always am amazing an
and another answer any anybody anyone anything anyway anyways apparently are
around as ask asked at avoid awesome back bad basically be because been before
being believe best better between big bit both bring but buy by call called
calling came can can't cancel cannot check checked checking come coming cool
could couldn't customers day definitely did didn't do does doesn't doing done
don't down during each early either else end enough even evening ever every
everybody everyone everything exactly fair feel feeling few find fine finally
first fix for forget found from fun get gets getting give glad go goes going
gonna good got great guess guys had hadn't haha half happy has hasn't have
haven't having he he'd he'll he's hello help her here here's hey hi him his
hmm honestly hope hopefully how how's however i if in instead is isn't it it'd
it'll it's its just keep kind know last late later least left less let let's
like likely listen literally little lol look looking looks lot love made make
makes making many maybe me mean meanwhile might mine more morning most
much must my need needs never new next nice night no nobody none nope not
nothing now of off oh ok okay old on once one only oops or other others our
out over overall perfect perhaps please plus pretty probably put quick quickly
quite rather re read ready real really remember right said same saw say says
see seems seen send sent she she'd she'll she's should shouldn't since so
some somebody someone something sometimes soon sorry sound sounds speaking
start started still stop sure take talk tell thank thanks that that's the
their them then there there's these they they'd they'll they're they've thing
things think this those though thought through thx till to today together
told tomorrow tonight too totally try trying turns two ugh um under unless
until up update us use used usually very wait want wanted was wasn't watch we
we'd we'll we're we've welcome well went were weren't what what's whatever
when where which while who who's why wish with without won't wonder
work works would wouldn't wow yay yeah yep yes yesterday yet you you'd you'll
you're you've your yours
""".split()) | frozenset("""
accept adjust approve archive attach block book build cancel change choose
clean clear close commit confirm copy create cut delete deploy double drop
edit email enable export fill finish follow forward grab handle hold include
invite join launch leave lock log loop merge message move open order paste
pay pick ping plan post print pull push reach reject release reload remind
remove rename reply report request reschedule reset restart return review
revert run save schedule search select set share ship show sign skip sort
split submit swap switch sync tag text track turn undo unlock upload verify
write
""".split())
# Verbs that are also first names -- mark, bill, grant, chase, sue, pat, rob,
# jack, drew, hunter, will, may -- are left out on purpose.

# Suffixes that mark an ordinary word far more often than a name.
_COMMON_SUFFIXES = (
    "ing", "ed", "ly", "tion", "sion", "ness", "ment", "able", "ible", "ful",
    "less", "ous", "ive", "ize", "ise", "ity",
)

_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def lowercase_words_in(text: str) -> set[str]:
    """Words the text itself writes in lowercase -- proof they are not names."""
    return {w for w in _WORD.findall(text) if w.islower()}


def is_sentence_case_only(word: str, *, protected: set[str] = frozenset(),
                          seen_lowercase: set[str] = frozenset()) -> bool:
    """True when ``word``'s capital is just sentence case and may be lowered."""
    if len(word) < 2 and word != "A":
        return False
    if word in ("I", "I'm", "I'll", "I've", "I'd"):
        return False
    # Only a plain Capitalised word: "API", "iOS" and "McDonald" keep theirs.
    if not word[:1].isupper() or (len(word) > 1 and not word[1:].islower()):
        return False
    lower = word.lower()
    if lower in protected:
        return False
    if lower in OPENERS or lower in seen_lowercase:
        return True
    return len(lower) > 5 and lower.endswith(_COMMON_SUFFIXES)
