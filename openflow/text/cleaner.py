"""Cleaner protocol + the deterministic rule-based implementation.

Every cleaner -- rules, Ollama, Gemini, or a hybrid -- takes a raw transcript
and returns finished text, so ``tests/harness.py`` can score them all against
the same golden corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .autofix import apply_symbol_fixes, apply_word_fixes
from .fillers import strip_fillers
from .punctuation import fix_punctuation, normalize_whitespace
from .spoken import apply_spoken_punctuation
from .stem_removal import Retraction, remove_false_starts


@dataclass(slots=True)
class CleanResult:
    text: str
    raw: str
    retractions: list[Retraction] = field(default_factory=list)
    fillers_removed: list[str] = field(default_factory=list)
    autofixes: int = 0
    engine: str = "rules"
    latency_ms: float = 0.0


@runtime_checkable
class Cleaner(Protocol):
    name: str

    def clean(self, raw: str) -> CleanResult: ...


class RuleBasedCleaner:
    """Zero-dependency, sub-millisecond cleanup pass.

    Order matters: false starts are resolved *before* fillers, because the
    pivot lexicon and the soft-filler lexicon overlap ("I mean" is a pivot when
    it precedes a correction and a filler when it does not).
    """

    name = "rules"

    def __init__(self, *, terminal_punctuation: bool = True) -> None:
        self.terminal_punctuation = terminal_punctuation

    def clean(self, raw: str) -> CleanResult:
        text = normalize_whitespace(raw)
        if not text:
            return CleanResult(text="", raw=raw, engine=self.name)

        # Spoken commands run first so "new paragraph" survives -- the word
        # pipeline below would otherwise flatten it into a space.
        text = apply_spoken_punctuation(text)
        # Word-level repairs are safe here: they add no punctuation, so the
        # sentence splitter below still sees what it expects.
        text, autofixes = apply_word_fixes(text)

        retractions: list[Retraction] = []
        fillers: list[str] = []
        segments: list[str] = []
        for segment in text.split("\n"):
            if not segment.strip():
                segments.append("")
                continue
            cleaned, segment_retractions = remove_false_starts(segment)
            cleaned, segment_fillers = strip_fillers(cleaned)
            cleaned = fix_punctuation(cleaned, terminal=self.terminal_punctuation)
            retractions.extend(segment_retractions)
            fillers.extend(segment_fillers)
            segments.append(cleaned)

        # Symbol repairs run last: they introduce '.', '@' and ':' , which the
        # sentence splitter above would have read as terminators.
        final, symbol_fixes = apply_symbol_fixes("\n".join(segments).strip())

        return CleanResult(
            text=final,
            raw=raw,
            retractions=retractions,
            fillers_removed=fillers,
            autofixes=autofixes + symbol_fixes,
            engine=self.name,
        )
