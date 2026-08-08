"""Transcription router: cloud first, local fallback, quota aware."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..config import Config
from ..llm.quota import QuotaLedger
from .engines import SttError, build_engine

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Transcript:
    text: str
    engine: str
    latency_ms: float


class SttRouter:
    def __init__(self, config: Config, quota: QuotaLedger | None = None) -> None:
        self.config = config
        self.quota = quota or QuotaLedger()
        self._engines = {}
        for name in config.stt.backends:
            try:
                self._engines[name] = build_engine(name, config)
            except ValueError:
                log.warning("unknown STT backend %r in config; skipping", name)

    def warm(self) -> None:
        """Pre-load whatever can be pre-loaded, off the hotkey path.

        Only the first local backend in the chain is warmed -- loading every
        engine would download weights the user may never need.
        """
        for name in self.config.stt.backends:
            engine = self._engines.get(name)
            if engine is None or not engine.is_local or not engine.available():
                continue
            if not hasattr(engine, "warm"):
                continue
            try:
                engine.warm()
            except Exception as exc:  # pragma: no cover - download/runtime issues
                log.warning("%s warm-up failed: %s", name, exc)
            return

    def transcribe(self, audio, sample_rate: int) -> Transcript:
        started = time.perf_counter()
        errors: list[str] = []
        duration_s = len(audio) / sample_rate if sample_rate else 0.0

        for name in self.config.stt.backends:
            engine = self._engines.get(name)
            if engine is None:
                continue

            limit = self.config.llm.daily_limits.get(name)
            audio_limit = self.config.llm.hourly_audio_seconds.get(name)
            if not engine.is_local:
                if not self.quota.has_headroom(name, limit):
                    log.info("%s over daily request limit; using local transcription", name)
                    continue
                if not self.quota.has_audio_headroom(name, audio_limit, duration_s):
                    log.info("%s over hourly audio budget; using local transcription", name)
                    continue
            if not engine.available():
                errors.append(f"{name}: unavailable")
                continue

            try:
                text = engine.transcribe(audio, sample_rate)
            except SttError as exc:
                errors.append(f"{name}: {exc}")
                if limit and "429" in str(exc):
                    self.quota.exhaust(name, limit)
                continue

            if not engine.is_local:
                self.quota.record(name, audio_seconds=duration_s)
            return Transcript(
                text=text, engine=name, latency_ms=(time.perf_counter() - started) * 1000
            )

        raise SttError("no transcription backend succeeded: " + "; ".join(errors))
