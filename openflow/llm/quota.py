"""Free-tier budget tracking.

PRD section 4 requires falling back to local processing once a daily cloud
limit is reached. Two budgets are tracked, because Groq enforces both:

  * requests per calendar day (2,000 for speech-to-text)
  * audio seconds per rolling hour (7,200)

State lives in a small JSON file so both ceilings survive app restarts.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date
from pathlib import Path

from ..config import CONFIG_DIR

HOUR_S = 3600.0


class QuotaLedger:
    def __init__(self, path: Path | None = None, *, clock=time.time) -> None:
        self.path = path or (CONFIG_DIR / "quota.json")
        self._clock = clock
        self._lock = threading.Lock()
        self._day = date.today().isoformat()
        self._counts: dict[str, int] = {}
        # provider -> [(timestamp, seconds_of_audio), ...] within the last hour
        self._audio: dict[str, list[list[float]]] = {}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if data.get("day") == self._day:
            self._counts = {k: int(v) for k, v in data.get("counts", {}).items()}
        # Audio events carry their own timestamps, so they survive a day roll.
        raw_audio = data.get("audio", {})
        if isinstance(raw_audio, dict):
            self._audio = {
                provider: [[float(t), float(s)] for t, s in events]
                for provider, events in raw_audio.items()
            }
            self._prune()

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"day": self._day, "counts": self._counts, "audio": self._audio}),
                encoding="utf-8",
            )
        except OSError:
            pass  # A quota file we cannot write is not worth failing a dictation over.

    def _roll_day(self) -> None:
        today = date.today().isoformat()
        if today != self._day:
            self._day, self._counts = today, {}

    def _prune(self) -> None:
        cutoff = self._clock() - HOUR_S
        for provider, events in list(self._audio.items()):
            kept = [e for e in events if e[0] >= cutoff]
            if kept:
                self._audio[provider] = kept
            else:
                del self._audio[provider]

    # -- requests per day --------------------------------------------------
    def used(self, provider: str) -> int:
        with self._lock:
            self._roll_day()
            return self._counts.get(provider, 0)

    def has_headroom(self, provider: str, limit: int | None) -> bool:
        if not limit:
            return True
        return self.used(provider) < limit

    def record(self, provider: str, audio_seconds: float = 0.0) -> None:
        with self._lock:
            self._roll_day()
            self._counts[provider] = self._counts.get(provider, 0) + 1
            if audio_seconds > 0:
                self._audio.setdefault(provider, []).append(
                    [self._clock(), float(audio_seconds)]
                )
                self._prune()
            self._flush()

    def exhaust(self, provider: str, limit: int) -> None:
        """Mark a provider as spent for the rest of the day -- used when the
        API itself returns a rate-limit error before our counter got there."""
        with self._lock:
            self._roll_day()
            self._counts[provider] = max(self._counts.get(provider, 0), limit)
            self._flush()

    # -- audio seconds per rolling hour ------------------------------------
    def audio_seconds_used(self, provider: str) -> float:
        with self._lock:
            self._prune()
            return sum(seconds for _, seconds in self._audio.get(provider, []))

    def has_audio_headroom(self, provider: str, limit: int | None, wanted: float) -> bool:
        """True if ``wanted`` more seconds still fits under the hourly cap.

        Checked *before* sending, so we fall back to local rather than burn a
        request on a guaranteed 429.
        """
        if not limit:
            return True
        return self.audio_seconds_used(provider) + wanted <= limit

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            self._roll_day()
            self._prune()
            return {
                "day": self._day,
                "requests": dict(self._counts),
                "audio_seconds": {
                    provider: round(sum(s for _, s in events), 1)
                    for provider, events in self._audio.items()
                },
            }
