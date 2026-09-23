"""Transcription router: cloud first, local fallback, quota aware.

Long recordings are split at pauses into pieces no longer than
``stt.chunk_seconds`` and transcribed in parallel (cloud) or in turn (local),
so a twenty-minute dictation costs about as long as its longest piece.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..config import Config, stt_chain, unlimited
from ..llm.quota import QuotaLedger, is_daily_exhaustion
from ..text.hallucinations import is_silence_hallucination
from .engines import SttError, build_engine

log = logging.getLogger(__name__)

# Look this far either side of each chunk boundary for the quietest moment.
SPLIT_SEARCH_S = 10.0
PARALLEL_CHUNKS = 8


@dataclass(slots=True)
class Transcript:
    text: str
    engine: str
    latency_ms: float
    # ISO code of the language spoken, when the engine could tell.
    language: str | None = None


def split_points(audio, sample_rate: int, chunk_s: float) -> list[int]:
    """Sample offsets that cut ``audio`` into pieces of at most ``chunk_s``,
    each cut placed at the quietest 50 ms within SPLIT_SEARCH_S of the
    boundary so no word is cut in half."""
    import numpy as np

    total = len(audio)
    step = int(chunk_s * sample_rate)
    if step <= 0 or total <= step:
        return []
    window = max(1, int(0.05 * sample_rate))
    search = int(SPLIT_SEARCH_S * sample_rate)
    cuts: list[int] = []
    start = 0
    while total - start > step:
        target = start + step
        lo = max(start + step // 2, target - search)
        hi = min(total - window, target)
        if hi <= lo:
            cut = target
        else:
            frames = (hi - lo) // window
            segment = np.asarray(audio[lo:lo + frames * window], dtype=np.float32)
            energy = (segment.reshape(frames, window) ** 2).mean(axis=1)
            cut = lo + int(np.argmin(energy)) * window + window // 2
        cuts.append(cut)
        start = cut
    return cuts


def _join(pieces: list[str]) -> str:
    """Rejoin transcribed pieces. A piece cut mid-sentence comes back
    capitalized as if it started one ("Seventh, Check that..."); when the
    previous piece did not end its sentence, lower that first word -- unless
    it looks like a name."""
    from ..text.casing import is_sentence_case_only, lowercase_words_in

    out: list[str] = []
    for piece in (p.strip() for p in pieces if p and p.strip()):
        if out and not out[-1].endswith((".", "!", "?", ":")):
            first = piece.split(" ", 1)[0].rstrip(",;:")
            seen = lowercase_words_in(" ".join(out) + " " + piece)
            if first.isalpha() and is_sentence_case_only(first, seen_lowercase=seen):
                piece = piece[0].lower() + piece[1:]
        out.append(piece)
    return " ".join(out)


class SttRouter:
    def __init__(self, config: Config, quota: QuotaLedger | None = None) -> None:
        self.config = config
        self.quota = quota or QuotaLedger()
        self._engines = {}

    def _engine(self, name: str):
        if name not in self._engines:
            try:
                self._engines[name] = build_engine(name, self.config)
            except ValueError:
                log.warning("unknown STT backend %r in config; skipping", name)
                self._engines[name] = None
        return self._engines[name]

    def warm(self) -> None:
        """Pre-load whatever can be pre-loaded, off the hotkey path.

        Only the first local backend in the chain is warmed -- loading every
        engine would download weights the user may never need.
        """
        for name in stt_chain(self.config):
            engine = self._engine(name)
            if engine is None or not engine.is_local or not engine.available():
                continue
            if not hasattr(engine, "warm"):
                continue
            try:
                engine.warm()
            except Exception as exc:  # pragma: no cover - download/runtime issues
                log.warning("%s warm-up failed: %s", name, exc)
            return

    def transcribe(self, audio, sample_rate: int, context=None) -> Transcript:
        started = time.perf_counter()
        errors: list[str] = []
        duration_s = len(audio) / sample_rate if sample_rate else 0.0

        for name in stt_chain(self.config):
            engine = self._engine(name)
            if engine is None:
                continue
            chunk_s = min(self.config.stt.chunk_seconds,
                          getattr(engine, "max_chunk_s", None) or self.config.stt.chunk_seconds)
            cuts = split_points(audio, sample_rate, chunk_s)
            bounds = list(zip([0] + cuts, cuts + [len(audio)]))
            chunks = [audio[a:b] for a, b in bounds]

            free_tier = not unlimited(self.config, name)
            limit = self.config.llm.daily_limits.get(name) if free_tier else None
            audio_limit = self.config.llm.hourly_audio_seconds.get(name) if free_tier else None
            if not engine.is_local:
                if not self.quota.has_headroom(name, limit, len(chunks)):
                    log.info("%s over daily request limit; using local transcription", name)
                    continue
                if not self.quota.has_audio_headroom(name, audio_limit, duration_s):
                    log.info("%s over hourly audio budget; using local transcription", name)
                    continue
            if not engine.available():
                errors.append(f"{name}: unavailable")
                continue

            try:
                text = self._run(engine, chunks, sample_rate, context)
            except SttError as exc:
                errors.append(f"{name}: {exc}")
                if limit and "429" in str(exc) and is_daily_exhaustion(str(exc)):
                    self.quota.exhaust(name, limit)
                continue

            if not engine.is_local:
                for (a, b) in bounds:
                    self.quota.record(name, audio_seconds=(b - a) / sample_rate)
            if len(chunks) > 1:
                log.info("transcribed %.0fs in %d pieces via %s", duration_s, len(chunks), name)
            return Transcript(
                text=text, engine=name,
                latency_ms=(time.perf_counter() - started) * 1000,
                language=getattr(engine, "last_language", None),
            )

        raise SttError("no transcription backend succeeded: " + "; ".join(errors))

    @staticmethod
    def _run(engine, chunks, sample_rate: int, context) -> str:
        from ..audio.conditioning import measure, voiced_seconds

        def one(chunk) -> str:
            try:
                text = engine.transcribe(chunk, sample_rate, context=context)
            except TypeError:
                # An engine written before context existed.
                text = engine.transcribe(chunk, sample_rate)
            if len(chunks) > 1:
                # A piece that is mostly pause can come back as subtitle
                # boilerplate ("Thank you."); inside a long dictation that is
                # an invented sentence, not a stray tap to discard.
                rms, _peak = measure(chunk)
                if is_silence_hallucination(text, len(chunk) / sample_rate, rms,
                                            voiced_seconds(chunk, sample_rate)):
                    return ""
            return text

        if len(chunks) == 1:
            return one(chunks[0])
        if engine.is_local:
            return _join([one(chunk) for chunk in chunks])
        with ThreadPoolExecutor(max_workers=min(PARALLEL_CHUNKS, len(chunks))) as pool:
            return _join(list(pool.map(one, chunks)))
