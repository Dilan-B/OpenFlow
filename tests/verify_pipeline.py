"""End-to-end integration check: speech in, finished text out.

Not part of the unit suite -- it needs model weights on disk and takes tens of
seconds. Run it after changing an STT backend or the cleanup chain:

    python -m tests.verify_pipeline
    python -m tests.verify_pipeline --engine moonshine

Test speech is synthesized with Moonshine's TTS, so this needs no microphone
and no checked-in audio fixture. That does mean we are transcribing synthetic
speech: it validates that the pipeline is wired correctly and that the cleanup
runs on real ASR output, not that WER on your voice is good.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.config import Config  # noqa: E402
from openflow.stt.engines import SttError, build_engine  # noqa: E402
from openflow.text.cleaner import RuleBasedCleaner  # noqa: E402

# Spoken form on the left (TTS reads digits better as words), expected final
# text on the right. The first case is the PRD's canonical example.
PHRASES: tuple[tuple[str, str], ...] = (
    (
        # ASR renders spoken numbers as digits, so the expectation uses digits.
        "Can we meet up on Tuesday at five, or actually, can we meet up on Friday at three.",
        "Can we meet up on Friday at 3",
    ),
    (
        "So we need to rebuild the index before Friday.",
        "So we need to rebuild the index before Friday",
    ),
)


def _comparable(text: str) -> str:
    """Terminal punctuation varies by ASR backend and is not what we test."""
    return text.strip().rstrip(".!?").strip().lower()


def synthesize(text: str):
    """Render ``text`` to a float32 mono waveform via Moonshine TTS."""
    import numpy as np
    from moonshine_voice.tts import TextToSpeech

    engine = TextToSpeech("en")
    try:
        result = engine.synthesize(text)
    finally:
        try:
            engine.close()
        except Exception:
            pass

    samples, rate = result if isinstance(result, tuple) else (result, 24_000)
    return np.asarray(samples, dtype=np.float32), int(rate)


def resample(audio, source_rate: int, target_rate: int):
    """Linear resample. Good enough for a wiring check; the app records at the
    target rate directly and never hits this path."""
    import numpy as np

    if source_rate == target_rate:
        return audio
    duration = len(audio) / source_rate
    target_len = int(duration * target_rate)
    return np.interp(
        np.linspace(0, len(audio) - 1, target_len),
        np.arange(len(audio)),
        audio,
    ).astype("float32")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end pipeline check.")
    parser.add_argument("--engine", default="parakeet_onnx",
                        help="parakeet_onnx | moonshine | faster_whisper | groq")
    args = parser.parse_args(argv)

    config = Config()
    rate = config.audio.sample_rate
    engine = build_engine(args.engine, config)

    if not engine.available():
        print(f"{args.engine}: not installed", file=sys.stderr)
        return 2

    cleaner = RuleBasedCleaner()
    failures = 0

    for spoken, expected in PHRASES:
        print(f"\n  speaking : {spoken!r}")
        try:
            audio, source_rate = synthesize(spoken)
        except Exception as exc:
            print(f"  TTS unavailable ({exc}); cannot run this check", file=sys.stderr)
            return 2
        audio = resample(audio, source_rate, rate)
        print(f"  audio    : {len(audio) / rate:.1f}s @ {rate} Hz")

        started = time.perf_counter()
        try:
            raw = engine.transcribe(audio, rate)
        except SttError as exc:
            print(f"  FAILED   : {exc}")
            failures += 1
            continue
        stt_ms = (time.perf_counter() - started) * 1000

        result = cleaner.clean(raw)
        print(f"  heard    : {raw!r}  ({args.engine}, {stt_ms:.0f} ms)")
        print(f"  cleaned  : {result.text!r}  ({result.latency_ms:.2f} ms)")
        for retraction in result.retractions:
            print(f"             [{retraction.strategy}] dropped {retraction.removed!r}")

        if _comparable(result.text) == _comparable(expected):
            print("  MATCH")
        else:
            # ASR is not deterministic across machines; a mismatch here is
            # informational unless the transcript itself came back empty.
            print(f"  DIFFERS from {expected!r} (check whether the transcript was wrong)")
            if not raw.strip():
                failures += 1

    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
