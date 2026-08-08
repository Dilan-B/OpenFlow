"""Microphone capture.

The stream is opened once at startup and left running, with capture gated by a
flag. Opening a PortAudio stream costs 100-300 ms on Windows, and paying that
on every hotkey press is the difference between "instant" and "laggy".
"""

from __future__ import annotations

import logging
import threading
import time

from ..config import AudioConfig

log = logging.getLogger(__name__)


class AudioUnavailable(RuntimeError):
    pass


class Recorder:
    def __init__(self, config: AudioConfig) -> None:
        self.cfg = config
        self._np = None
        self._sd = None
        self._stream = None
        self._chunks: list = []
        self._capturing = False
        self._lock = threading.Lock()
        self._level = 0.0
        self._started_at = 0.0

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise AudioUnavailable(f"install the audio extras: {exc}") from exc

        self._np, self._sd = np, sd
        blocksize = int(self.cfg.sample_rate * self.cfg.block_ms / 1000)
        self._stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype="float32",
            blocksize=blocksize,
            device=self.cfg.input_device,
            callback=self._on_block,
        )
        self._stream.start()
        log.info("audio stream open at %d Hz", self.cfg.sample_rate)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # -- capture -----------------------------------------------------------
    def _on_block(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("audio status: %s", status)
        block = indata.copy()
        rms = float(self._np.sqrt(self._np.mean(block**2))) if frames else 0.0
        # Fast attack, slow decay: the meter jumps the instant you speak and
        # falls away smoothly, so the pill reads as reactive rather than lagged.
        instant = min(1.0, rms * 16)
        self._level = instant if instant > self._level else self._level * 0.86
        with self._lock:
            if self._capturing:
                self._chunks.append(block)

    def start(self) -> None:
        with self._lock:
            self._chunks = []
            self._capturing = True
        self._started_at = time.monotonic()

    def stop(self):
        """Stop capture and return the recorded mono float32 array."""
        with self._lock:
            self._capturing = False
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return None
        audio = self._np.concatenate(chunks, axis=0)
        return audio[:, 0] if audio.ndim > 1 else audio

    def cancel(self) -> None:
        with self._lock:
            self._capturing = False
            self._chunks = []

    # -- introspection -----------------------------------------------------
    @property
    def level(self) -> float:
        """Smoothed 0..1 input level, for the overlay's waveform."""
        return self._level

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._started_at if self._capturing else 0.0

    @property
    def over_limit(self) -> bool:
        return self._capturing and self.elapsed_s > self.cfg.max_seconds
