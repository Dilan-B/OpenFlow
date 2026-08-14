"""Silence hallucinations.

Whisper-family models were trained on subtitle corpora, so when handed audio
with no speech in it they emit the things subtitle files are full of: "Thank
you.", "Bye.", "Thanks for watching!", a translator credit. The decoder is not
confused -- it is doing exactly what its training data suggests.

``audio/conditioning.py`` already trims leading and trailing silence, which
removes most of the opportunity. This is the backstop for what gets through:
a clip that was short or quiet, whose entire transcript is one of these
stock phrases, is discarded rather than pasted into somebody's document.

The gate is deliberately narrow. A real "thank you" is a thing people dictate,
so the phrase alone is never enough -- the transcript has to be *nothing but*
the phrase, and the audio has to have been too short or too quiet to plausibly
contain it.
"""

from __future__ import annotations

import re

# Matched against the whole transcript, lowercased and stripped of punctuation.
STOCK_PHRASES: frozenset[str] = frozenset({
    "thank you",
    "thanks",
    "thank you very much",
    "thank you for watching",
    "thanks for watching",
    "thanks for watching!",
    "please subscribe",
    "like and subscribe",
    "bye",
    "bye bye",
    "goodbye",
    "you",
    "the",
    "so",
    "okay",
    "ok",
    "oh",
    "hmm",
    "mm",
    "uh",
    "um",
    "subtitles by the amara org community",
    "subtitles by the amara.org community",
    "transcription by castingwords",
    "copyright",
    "music",
    "applause",
    "silence",
    "end of transcript",
})

# Audio shorter than this cannot hold a real sentence, so a stock phrase from
# it is almost certainly invented. Above it we leave the transcript alone.
SUSPECT_DURATION_S = 1.6
# Quiet clips are the other tell: the trimmer found little to keep.
SUSPECT_RMS = 0.004

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace -- the form the
    phrase table is written in."""
    folded = _PUNCT_RE.sub(" ", (text or "").lower())
    return _WS_RE.sub(" ", folded).strip()


def is_silence_hallucination(text: str, duration_s: float, rms: float) -> bool:
    """True when ``text`` looks invented rather than heard.

    Requires all three: the transcript is nothing but a stock phrase, and the
    clip was either too short or too quiet to have contained it.
    """
    normalized = normalize(text)
    if not normalized:
        return False
    if normalized not in STOCK_PHRASES:
        return False
    return duration_s <= SUSPECT_DURATION_S or rms <= SUSPECT_RMS
