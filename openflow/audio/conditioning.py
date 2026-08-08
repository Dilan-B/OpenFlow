"""Audio conditioning applied before transcription.

Three cheap fixes that measurably help every STT backend, in order:

1. **DC offset removal.** Many laptop and USB mics sit on a small DC bias.
   The bias contributes nothing acoustically but skews the mel spectrogram
   that Whisper and Parakeet both compute.
2. **Peak normalization.** Push-to-talk recordings are often quiet -- people
   lean back, or the mic gain is low. ASR models are trained on
   loudness-normalized corpora, so a quiet clip is genuinely out of
   distribution. We scale to a fixed peak rather than compress, which keeps
   the dynamics the model expects.
3. **Silence trimming.** The hotkey is pressed slightly before speech starts
   and released slightly after it ends. Leading/trailing silence gives the
   decoder room to hallucinate ("Thank you.", "Bye.") -- a well-documented
   Whisper failure mode -- and costs time on every request.

All of it is numpy, microseconds on a dictation-length clip, and every step
is a no-op when the input does not need it.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

TARGET_PEAK = 0.85       # leaves headroom; avoids clipping artifacts
NORMALIZE_PERCENTILE = 99  # gain reference: the speech body, not a lone click
CLIP_CEILING = 0.99      # a transient may reach this, never exceed it
KEEP_PAD_S = 0.12        # keep this much silence around speech
MIN_SPEECH_S = 0.20      # never trim a clip down below this

# Two thresholds, measured on two different signals. Conflating them is what
# broke dictation on a quiet microphone: the trim runs *after* peak
# normalization, where the loudest sample is always TARGET_PEAK, so 0.02 is a
# meaningful fraction. The discard gate runs on *raw* audio, whose absolute
# level depends entirely on the device's gain -- a real mic measured here sat
# at 0.000488 RMS ambient and still produced speech under 0.012.
TRIM_FLOOR = 0.02        # normalized-domain: envelope below this is silence

# Discard gate, raw domain. Deliberately close to digital silence: throwing
# away a real utterance makes the app look broken, while letting a silent clip
# through costs one request. Both conditions must hold, so a clip is only
# dropped when it is quiet by peak *and* by average.
SILENCE_RMS = 0.0015     # ~3x the measured ambient floor of a quiet mic
SILENCE_PEAK = 0.02      # speech transients clear this comfortably


def condition(audio, sample_rate: int):
    """Return ``audio`` DC-corrected, normalized, and silence-trimmed."""
    import numpy as np

    if audio is None or len(audio) == 0:
        return audio

    samples = np.asarray(audio, dtype=np.float32)

    # 1. DC offset.
    samples = samples - float(np.mean(samples))

    # 2. Normalize against the 99th percentile rather than the single loudest
    #    sample. A keyboard tap or mic pop is one sample; keying the gain off
    #    it pins the whole clip quiet, which is precisely the level ASR is
    #    worst at. The percentile tracks the speech body instead.
    #    Skip near-silent clips so we do not amplify room noise into something
    #    the decoder tries to transcribe.
    magnitude = np.abs(samples)
    reference = float(np.percentile(magnitude, NORMALIZE_PERCENTILE)) if samples.size else 0.0
    if reference > 0.002:
        #    Limit the outliers instead of letting them hold the gain down.
        #    Capping the gain so a lone click stays unclipped would leave every
        #    other sample as quiet as it started -- the click wins and the
        #    speech loses. Clipping a handful of samples is inaudible to a
        #    transcriber; a whisper-quiet clip is not.
        samples = np.clip(samples * (TARGET_PEAK / reference),
                          -CLIP_CEILING, CLIP_CEILING)

    # 3. Silence trim, on a 20 ms RMS envelope.
    window = max(1, int(sample_rate * 0.02))
    usable = (len(samples) // window) * window
    if usable >= window:
        frames = samples[:usable].reshape(-1, window)
        envelope = np.sqrt(np.mean(frames**2, axis=1))
        loud = np.flatnonzero(envelope > TRIM_FLOOR)
        if loud.size:
            pad = int(KEEP_PAD_S * sample_rate)
            start = max(0, loud[0] * window - pad)
            end = min(len(samples), (loud[-1] + 1) * window + pad)
            if end - start >= int(MIN_SPEECH_S * sample_rate):
                samples = samples[start:end]

    return samples


def measure(audio) -> tuple[float, float]:
    """Return ``(rms, peak)`` for a clip. Reported in the log on a discard, so
    a mis-tuned gate is one glance to diagnose instead of a mystery."""
    import numpy as np

    if audio is None or len(audio) == 0:
        return 0.0, 0.0
    samples = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(samples**2))), float(np.max(np.abs(samples)))


def is_silent(audio, sample_rate: int) -> bool:
    """True only when a clip holds no signal at all -- a muted or disconnected
    microphone, or the hotkey brushed by accident.

    This is deliberately not a "was there speech" test. Microphone gain spans
    more than an order of magnitude across devices, so any threshold high
    enough to judge *content* will silently swallow somebody's real voice.
    """
    if audio is None or len(audio) < sample_rate * 0.1:
        return True
    rms, peak = measure(audio)
    return rms < SILENCE_RMS and peak < SILENCE_PEAK
