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
from .repetitions import collapse_repetitions
from .spoken import apply_spoken_punctuation
from .stem_removal import Retraction, remove_false_starts


@dataclass(slots=True)
class CleanResult:
    text: str
    raw: str
    retractions: list[Retraction] = field(default_factory=list)
    fillers_removed: list[str] = field(default_factory=list)
    repetitions_collapsed: list[str] = field(default_factory=list)
    autofixes: int = 0
    engine: str = "rules"
    latency_ms: float = 0.0

    @property
    def uncertain(self) -> bool:
        """True when the deterministic pass hit something it could not resolve
        cleanly, so a second opinion from the LLM is worth its latency.

        Strategy "keep-both" is stem_removal's rule D -- it found a retraction
        pivot but could not tell how much the speaker threw away, so it kept
        everything. That is the one case where a model reliably does better.
        """
        return any(r.strategy == "keep-both" for r in self.retractions)


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
        repetitions: list[str] = []
        segments: list[str] = []
        for segment in text.split("\n"):
            if not segment.strip():
                segments.append("")
                continue
            # Stutters first: "can we can we meet up on Tuesday, or actually
            # Friday" should reach the pivot logic as one clean clause, not as
            # a doubled one that looks like a re-anchor.
            cleaned, segment_repetitions = collapse_repetitions(segment)
            cleaned, segment_retractions = remove_false_starts(cleaned)
            cleaned, segment_fillers = strip_fillers(cleaned)
            cleaned = fix_punctuation(cleaned, terminal=self.terminal_punctuation)
            retractions.extend(segment_retractions)
            fillers.extend(segment_fillers)
            repetitions.extend(segment_repetitions)
            segments.append(cleaned)

        # Symbol repairs run last: they introduce '.', '@' and ':' , which the
        # sentence splitter above would have read as terminators.
        final, symbol_fixes = apply_symbol_fixes("\n".join(segments).strip())

        return CleanResult(
            text=final,
            raw=raw,
            retractions=retractions,
            fillers_removed=fillers,
            repetitions_collapsed=repetitions,
            autofixes=autofixes + symbol_fixes,
            engine=self.name,
        )
