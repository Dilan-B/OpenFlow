"""The start and stop sounds: the little "pop-pop" Wispr Flow makes.

Synthesized here rather than shipped as a file. Both were fitted to a
recording of Wispr Flow (spectrogram error ~2.5 dB above the noise floor):

* start -- two quick wooden pops rising, ~290 Hz then ~420 Hz, 33 ms apart,
  with a faint echo of the second one;
* stop  -- the same pair falling, ~425 Hz then ~315 Hz, 48 ms apart.

Each pop is a sine that settles slightly downward in pitch over its first few
milliseconds (which is what makes it sound like a pop rather than a beep),
with a 2 ms attack, a ~5-10 ms exponential decay and a low "body" thump under
it. The whole sound is under 160 ms. The microphone opens at the same moment
as the start sound, so it stays well below speech level.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

SAMPLE_RATE = 48_000
VOLUME = 0.22

# One pop: (onset s, freq Hz, amp, attack s, decay s, 2nd harmonic,
#           body Hz, body amp, body decay s, initial pitch overshoot)
_START = (
    (0.0000, 274.0, 1.84, 0.0022, 0.0045, 0.067, 142.0, 0.28, 0.0049, 0.064),
    (0.0332, 411.0, 1.26, 0.0024, 0.0091, 0.000, 60.0, 0.08, 0.0035, 0.073),
    (0.0823, 427.0, 0.10, 0.0041, 0.0200, 0.082, 60.0, 0.10, 0.0100, 0.084),
)
_STOP = (
    (0.0000, 427.0, 1.53, 0.0017, 0.0062, 0.007, 194.0, 0.31, 0.0010, -0.007),
    (0.0482, 273.0, 0.72, 0.0019, 0.0099, 0.117, 60.0, 0.30, 0.0055, 0.158),
    (0.0961, 330.0, 0.11, 0.0031, 0.0200, 0.066, 108.0, 0.17, 0.0043, 0.112),
)
_TAIL = 0.06        # room for the last pop to decay

_cache: dict[str, object] = {}
_lock = threading.Lock()


def _render(pops):
    import numpy as np

    total = max(p[0] for p in pops) + _TAIL
    n = int(total * SAMPLE_RATE)
    out = np.zeros(n, dtype=np.float64)
    for onset, freq, amp, attack, decay, h2, body_f, body_amp, body_decay, glide in pops:
        begin = int(onset * SAMPLE_RATE)
        t = np.arange(n - begin) / SAMPLE_RATE
        rise = np.minimum(1.0, t / attack)
        pitch = freq * (1.0 + glide * np.exp(-t / 0.006))
        phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
        tone = (np.sin(phase) + h2 * np.sin(2 * phase)) * rise * np.exp(-t / decay)
        body = body_amp * np.sin(2 * np.pi * body_f * t) * rise * np.exp(-t / body_decay)
        out[begin:] += amp * (tone + body)
    # Short fade at the very end so the tail never ends on a nonzero sample.
    fade = int(0.01 * SAMPLE_RATE)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    out[0] = 0.0
    return (out / np.abs(out).max() * VOLUME).astype(np.float32)


def samples(kind: str = "start"):
    with _lock:
        if kind not in _cache:
            _cache[kind] = _render(_START if kind == "start" else _STOP)
        return _cache[kind]


def _play(kind: str) -> None:
    try:
        import sounddevice as sd

        sd.play(samples(kind), SAMPLE_RATE, blocking=False)
    except Exception as exc:
        log.debug("could not play the %s sound: %s", kind, exc)


def play() -> None:
    """Play the start sound and return immediately. Never raises: a missing
    output device must not stop anyone dictating."""
    _play("start")


def play_stop() -> None:
    """Play the stop sound (key released, recording over). Never raises."""
    _play("stop")
