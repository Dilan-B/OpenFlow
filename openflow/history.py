"""Dictation history and lifetime stats.

Backs the "Recent" list and the stat tiles in the main window. Text is stored
only when ``log_transcripts`` is on -- otherwise we keep counts and timings and
throw the words away, because dictation content is sensitive by default.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import CONFIG_DIR

# Typing speed used to estimate time saved. 40 wpm is a realistic sustained
# rate for prose on a keyboard, not a peak burst number.
TYPING_WPM = 40.0


@dataclass(slots=True)
class Entry:
    at: float                 # unix timestamp
    words: int
    chars: int
    duration_s: float         # length of the audio
    latency_ms: float         # capture end -> text injected
    stt_engine: str
    clean_engine: str
    retractions: int
    text: str = ""            # only when log_transcripts is enabled

    @property
    def when(self) -> str:
        return time.strftime("%H:%M", time.localtime(self.at))


@dataclass(slots=True)
class Stats:
    dictations: int = 0
    words: int = 0
    speech_seconds: float = 0.0
    latency_ms_total: float = 0.0
    retractions: int = 0
    dict_fixes: int = 0
    # ISO date -> words dictated that day. Feeds the streak and the Insights
    # chart; pruned to the last 90 days.
    daily: dict = field(default_factory=dict)

    @property
    def longest_streak(self) -> int:
        import datetime as dt

        best = run = 0
        previous = None
        for iso in sorted(self.daily):
            day = dt.date.fromisoformat(iso)
            run = run + 1 if previous is not None and day - previous == dt.timedelta(days=1) else 1
            best = max(best, run)
            previous = day
        return best

    @property
    def streak_days(self) -> int:
        import datetime as dt

        day = dt.date.today()
        streak = 0
        while day.isoformat() in self.daily:
            streak += 1
            day -= dt.timedelta(days=1)
        return streak

    def last_days(self, count: int = 7) -> list[tuple[str, int]]:
        import datetime as dt

        today = dt.date.today()
        return [
            ((today - dt.timedelta(days=i)).isoformat(),
             int(self.daily.get((today - dt.timedelta(days=i)).isoformat(), 0)))
            for i in range(count - 1, -1, -1)
        ]

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_total / self.dictations if self.dictations else 0.0

    @property
    def speaking_wpm(self) -> float:
        minutes = self.speech_seconds / 60
        return self.words / minutes if minutes > 0.01 else 0.0

    @property
    def minutes_saved(self) -> float:
        """Time typing the same words would have taken, minus time spent
        speaking them."""
        typing_minutes = self.words / TYPING_WPM
        return max(0.0, typing_minutes - self.speech_seconds / 60)


class History:
    def __init__(self, path: Path | None = None, limit: int = 50) -> None:
        self.path = path or (CONFIG_DIR / "history.json")
        self.limit = limit
        self._lock = threading.Lock()
        self.entries: list[Entry] = []
        self.stats = Stats()
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        try:
            self.stats = Stats(**data.get("stats", {}))
            self.entries = [Entry(**e) for e in data.get("entries", [])][: self.limit]
        except TypeError:
            # Schema drift from an older build -- start clean rather than crash.
            self.entries, self.stats = [], Stats()

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "stats": asdict(self.stats),
                        "entries": [asdict(e) for e in self.entries[: self.limit]],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def add(self, entry: Entry, dict_fixes: int = 0) -> None:
        with self._lock:
            self.entries.insert(0, entry)
            del self.entries[self.limit:]
            self.stats.dictations += 1
            self.stats.words += entry.words
            self.stats.speech_seconds += entry.duration_s
            self.stats.latency_ms_total += entry.latency_ms
            self.stats.retractions += entry.retractions
            self.stats.dict_fixes += dict_fixes
            day = time.strftime("%Y-%m-%d", time.localtime(entry.at))
            self.stats.daily[day] = int(self.stats.daily.get(day, 0)) + entry.words
            if len(self.stats.daily) > 90:
                for old in sorted(self.stats.daily)[: len(self.stats.daily) - 90]:
                    del self.stats.daily[old]
            self._flush()

    def clear(self) -> None:
        with self._lock:
            self.entries = []
            self._flush()

    def recent(self, count: int = 12) -> list[Entry]:
        with self._lock:
            return list(self.entries[:count])
