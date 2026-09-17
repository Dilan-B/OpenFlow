"""Time and score the cleanup backends against each other.

Which model to send transcripts to is a latency/quality trade, and both halves
move: free tiers change their model lists, and a model that was fast last month
may not be. This script re-runs the comparison rather than trusting a number
committed to a README months ago.

    python scripts/bench_cleanup.py
    python scripts/bench_cleanup.py --models openai/gpt-oss-20b,qwen/qwen3.8-27b
    python scripts/bench_cleanup.py --runs 3

Latency is reported as the median of ``--runs`` passes over the sample set, so
one slow cold start does not decide the ranking. Correctness here is only a
smoke test -- the real scoring is ``python -m tests.harness --cleaner groq``
against the golden corpora, which is what actually picked the default:

    python -m tests.harness --corpus wispr --cleaner groq --delay 8
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.config import Config  # noqa: E402
from openflow.llm.base import ProviderError  # noqa: E402
from openflow.llm.providers import build_provider, system_prompt_for  # noqa: E402

# Short, and each one carries a different failure mode: fillers, a mid-sentence
# pivot, a stutter, and a sentence that must come back untouched.
SAMPLES = [
    "um so like i was thinking you know we should probably uh ship the thing on friday right",
    "can we meet up on tuesday at 5, or actually, can we meet up on friday at 3.",
    "so basically the issue is um the parser it doesn't it doesn't handle uh nested quotes",
    "I actually finished the migration last night.",
]

DEFAULT_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


def bench(provider, runs: int) -> tuple[float, list[str], int]:
    """Return (median ms per call, last outputs, failure count)."""
    timings: list[float] = []
    outputs: list[str] = []
    failures = 0
    system = system_prompt_for(provider)
    for _ in range(runs):
        outputs = []
        for sample in SAMPLES:
            started = time.perf_counter()
            try:
                outputs.append(provider.complete(system, sample))
            except ProviderError as exc:
                failures += 1
                outputs.append(f"<{exc}>")
            timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings), outputs, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="comma-separated Groq chat models to compare")
    parser.add_argument("--runs", type=int, default=3,
                        help="passes over the sample set (default 3)")
    parser.add_argument("--backend", default="groq", choices=["groq", "gemini", "ollama"])
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print each model's output for every sample")
    args = parser.parse_args()

    config = Config.load()
    results = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        if args.backend == "groq":
            config.llm.groq_model = model
        elif args.backend == "gemini":
            config.llm.gemini_model = model
        else:
            config.llm.ollama_model = model

        provider = build_provider(args.backend, config)
        if not provider.available():
            print(f"{model:28s}  unavailable (no key, or backend not running)")
            continue

        median_ms, outputs, failures = bench(provider, args.runs)
        results.append((median_ms, model, failures))
        note = f"  {failures} failed" if failures else ""
        print(f"{model:28s}  {median_ms:7.0f} ms/call{note}")
        if args.verbose:
            for sample, out in zip(SAMPLES, outputs):
                print(f"    in : {sample}")
                print(f"    out: {out}\n")

    if results:
        results.sort()
        print(f"\nfastest: {results[0][1]} at {results[0][0]:.0f} ms/call")
        print("Score quality separately: python -m tests.harness --cleaner "
              f"{args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
