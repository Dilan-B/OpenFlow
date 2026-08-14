"""Time local Parakeet inference on each available execution provider.

Run this after installing a GPU runtime to see whether it is actually worth it
on your machine:

    pip install onnxruntime-directml     # or onnxruntime-gpu
    python scripts/bench_providers.py

Timings are per-call after warm-up, on a fixed-length clip, so they compare
compute rather than model download or first-load cost.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLIP_SECONDS = 4.0
SAMPLE_RATE = 16_000
RUNS = 5


def make_clip(path: str | None = None):
    """Load a 16 kHz mono WAV, or fall back to speech-shaped noise.

    Prefer a real clip: a transducer's decode loop runs once per emitted
    token, so noise -- which the model happily turns into a torrent of
    tokens -- overstates decode cost badly and distorts any comparison.
    """
    import numpy as np

    if path:
        import wave

        with wave.open(path, "rb") as handle:
            frames = handle.readframes(handle.getnframes())
            rate = handle.getframerate()
            channels = handle.getnchannels()
        samples = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{path} is {rate} Hz; resample to {SAMPLE_RATE} first")
        return samples

    rng = np.random.default_rng(0)
    n = int(CLIP_SECONDS * SAMPLE_RATE)
    noise = rng.standard_normal(n).astype("float32")
    # Roll off the highs so it lands in roughly the band speech occupies.
    kernel = np.ones(32, dtype="float32") / 32
    shaped = np.convolve(noise, kernel, mode="same")
    envelope = (1 + np.sin(np.linspace(0, 12, n))) / 2
    return (shaped * envelope * 0.3).astype("float32")


def bench(provider: str, model_name: str, quantization: str | None, clip, seconds: float) -> None:
    import onnx_asr

    providers = [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
    try:
        started = time.perf_counter()
        model = onnx_asr.load_model(model_name, quantization=quantization, providers=providers)
        load_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:
        print(f"  {provider:<28} unavailable: {exc}")
        return

    try:
        model.recognize(clip, sample_rate=SAMPLE_RATE)   # warm-up
    except Exception as exc:
        print(f"  {provider:<28} failed on first call: {exc}")
        return

    times = []
    for _ in range(RUNS):
        started = time.perf_counter()
        model.recognize(clip, sample_rate=SAMPLE_RATE)
        times.append((time.perf_counter() - started) * 1000)
    times.sort()
    median = times[len(times) // 2]
    print(f"  {provider:<28} load {load_ms:7.0f} ms   transcribe {median:6.0f} ms "
          f"(min {times[0]:.0f}, max {times[-1]:.0f})   {seconds / (median / 1000):.0f}x realtime")


def main() -> int:
    import onnxruntime

    from openflow.config import Config
    from openflow.stt.providers import available_providers

    wav = sys.argv[1] if len(sys.argv) > 1 else None
    config = Config.load()
    model_name = config.stt.parakeet_model
    quantization = config.stt.parakeet_quantization or None

    candidates = [p for p in available_providers()]
    if not candidates:
        print("no execution providers available -- is onnxruntime installed?")
        return 1

    clip = make_clip(wav)
    seconds = len(clip) / SAMPLE_RATE
    print(f"onnxruntime {onnxruntime.__version__}")
    print(f"model {model_name} (quantization={quantization or 'fp32'}), "
          f"{seconds:.1f}s {'clip from ' + wav if wav else 'synthetic clip'}, "
          f"median of {RUNS}\n")
    for provider in candidates:
        bench(provider, model_name, quantization, clip, seconds)

    if not any(p != "CPUExecutionProvider" for p in candidates):
        print("\nOnly CPU is available. For GPU inference, replace the runtime:")
        print("    pip uninstall onnxruntime")
        print("    pip install onnxruntime-directml   # any DX12 GPU")
        print("    pip install onnxruntime-gpu        # NVIDIA, needs CUDA + cuDNN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
