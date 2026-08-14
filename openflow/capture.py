"""Opt-in local capture of dictation events, for building a real eval set.

Why this exists: ``tests/corpus/stem_cases.json`` scores 22/22, but those 22
cases were hand-written alongside the implementation they validate, and every
one is text-in / text-out. The cleaner never sees an ASR error in testing, and
ASR output is the only input it ever gets in production. That gap is invisible
from the score.

What this records is the one signal that comes free and is honestly labelled:
whether the user *kept* what we gave them. :meth:`Capture.reject` is called
from the undo hotkey, so every undo is a labelled failure with the audio and
both transcripts attached.

Privacy, because this is dictation and the promise is that it stays local:

* Off by default. Nothing is written until the user turns it on.
* Everything lands in ``~/.openflow/capture/`` and is never transmitted.
* Audio is a separate opt-in from text, because a recording of your voice is a
  different thing to keep than a sentence you typed.
* :func:`Capture.purge` deletes it all, and the Settings toggle calls it when
  capture is switched off.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import CONFIG_DIR

log = logging.getLogger(__name__)

CAPTURE_DIR = CONFIG_DIR / "capture"
EVENTS_PATH = CAPTURE_DIR / "events.jsonl"
AUDIO_DIR = CAPTURE_DIR / "audio"

# Share of records reserved for evaluation. Assigned by hashing the record id,
# so a given utterance lands in the same split forever -- re-running an export
# cannot leak a case from train into eval.
EVAL_FRACTION = 0.2


@dataclass(slots=True)
class Event:
    id: str
    at: float
    raw: str                    # what the STT engine returned
    cleaned: str                # what we actually inserted
    stt_engine: str
    clean_engine: str
    duration_s: float
    rms: float
    latency_ms: float
    strategies: list[str] = field(default_factory=list)
    fillers_removed: list[str] = field(default_factory=list)
    repetitions_collapsed: list[str] = field(default_factory=list)
    uncertain: bool = False
    audio_path: str = ""
    # Set later by reject(): the user undid this insertion, so the cleaned text
    # was wrong. The single most valuable label in the file.
    rejected: bool = False
    split: str = "train"


def _split_for(record_id: str) -> str:
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return "eval" if (int(digest[:8], 16) % 1000) / 1000 < EVAL_FRACTION else "train"


class Capture:
    """Append-only event log. Every method is a no-op when disabled."""

    def __init__(self, config) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._last_id: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config.capture, "enabled", False))

    def record(self, *, raw: str, cleaned: str, stt_engine: str, clean_engine: str,
               duration_s: float, rms: float, latency_ms: float,
               strategies: list[str], fillers_removed: list[str],
               repetitions_collapsed: list[str], uncertain: bool,
               audio=None, sample_rate: int = 16_000) -> str | None:
        """Append one dictation. Returns the record id, or None when disabled.

        Never raises: a failure to write a research artifact must not cost the
        user the text they just dictated.
        """
        if not self.enabled:
            return None
        try:
            at = time.time()
            record_id = hashlib.sha256(
                f"{at}{raw}{duration_s}".encode("utf-8")).hexdigest()[:16]

            audio_path = ""
            if audio is not None and getattr(self.config.capture, "audio", False):
                audio_path = self._write_audio(record_id, audio, sample_rate)

            event = Event(
                id=record_id, at=at, raw=raw, cleaned=cleaned,
                stt_engine=stt_engine, clean_engine=clean_engine,
                duration_s=round(duration_s, 3), rms=round(rms, 6),
                latency_ms=round(latency_ms, 1), strategies=strategies,
                fillers_removed=fillers_removed,
                repetitions_collapsed=repetitions_collapsed,
                uncertain=uncertain, audio_path=audio_path,
                split=_split_for(record_id),
            )
            with self._lock:
                EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with EVENTS_PATH.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
                self._last_id = record_id
            return record_id
        except Exception as exc:
            log.warning("capture failed: %s", exc)
            return None

    def reject(self, record_id: str | None = None) -> None:
        """Mark a record as undone by the user -- a labelled failure.

        Rewrites the line in place rather than appending a correction, so the
        file stays one-record-per-dictation and an export does not have to
        reconcile two entries.
        """
        if not self.enabled:
            return
        target = record_id or self._last_id
        if not target:
            return
        try:
            with self._lock:
                if not EVENTS_PATH.exists():
                    return
                lines = EVENTS_PATH.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(lines):
                    if f'"id": "{target}"' not in line:
                        continue
                    data = json.loads(line)
                    data["rejected"] = True
                    lines[i] = json.dumps(data, ensure_ascii=False)
                    break
                EVENTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            log.warning("could not mark capture %s rejected: %s", target, exc)

    def _write_audio(self, record_id: str, audio, sample_rate: int) -> str:
        import wave

        import numpy as np

        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        path = AUDIO_DIR / f"{record_id}.wav"
        samples = np.asarray(audio, dtype=np.float32)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
        return str(path.relative_to(CONFIG_DIR))

    @staticmethod
    def purge() -> int:
        """Delete every captured artifact. Returns how many files went."""
        removed = 0
        for path in (list(AUDIO_DIR.glob("*.wav")) if AUDIO_DIR.exists() else []):
            path.unlink(missing_ok=True)
            removed += 1
        if EVENTS_PATH.exists():
            EVENTS_PATH.unlink()
            removed += 1
        return removed

    @staticmethod
    def stats() -> dict:
        """Counts for the Settings page, so the user can see what they hold."""
        if not EVENTS_PATH.exists():
            return {"total": 0, "rejected": 0, "eval": 0, "audio": 0}
        total = rejected = held = 0
        for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            rejected += bool(data.get("rejected"))
            held += data.get("split") == "eval"
        audio = len(list(AUDIO_DIR.glob("*.wav"))) if AUDIO_DIR.exists() else 0
        return {"total": total, "rejected": rejected, "eval": held, "audio": audio}
