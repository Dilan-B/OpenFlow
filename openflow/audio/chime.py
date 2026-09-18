"""The little sound that says "recording" when the dictation hotkey goes down.

Synthesized here rather than shipped as a file: two short, soft sine notes a
fifth apart (A5 then E6), each with a quick attack and an exponential decay,
plus a quiet octave partial for a bit of glassiness. About 160 ms in all and
deliberately quiet -- it should confirm, not startle, and the microphone opens
at the same moment, so a loud tone would end up in the recording.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

SAMPLE_RATE = 44_100
VOLUME = 0.16

# (frequency Hz, start s, length s)
_NOTES = ((880.0, 0.0, 0.11), (1318.5, 0.055, 0.105))

_samples = None
_lock = threading.Lock()


def _render():
    import numpy as np

    total = max(start + length for _, start, length in _NOTES)
    out = np.zeros(int(total * SAMPLE_RATE) + 1, dtype=np.float32)
    for freq, start, length in _NOTES:
        t = np.arange(int(length * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
        attack = np.minimum(1.0, t / 0.004)          # 4 ms: no click
        decay = np.exp(-t * 38.0)
        tone = np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(4 * np.pi * freq * t)
        begin = int(start * SAMPLE_RATE)
        out[begin:begin + len(t)] += (tone * attack * decay).astype(np.float32)
    # Short fade at the very end so the tail never ends on a nonzero sample.
    fade = min(len(out), int(0.01 * SAMPLE_RATE))
    out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out / np.abs(out).max() * VOLUME


def samples():
    global _samples
    with _lock:
        if _samples is None:
            _samples = _render()
        return _samples


def play() -> None:
    """Start the chime and return immediately. Never raises: a missing output
    device must not stop anyone dictating."""
    try:
        import sounddevice as sd

        sd.play(samples(), SAMPLE_RATE, blocking=False)
    except Exception as exc:
        log.debug("could not play the start chime: %s", exc)
