"""Scoring harness for transcript cleanup.

Runs the golden corpus through any cleaner and reports exact-match and
normalized-match accuracy, per-tag breakdown, and per-case diffs. The same
harness scores the deterministic rules and the LLM backends, which is how we
tell whether a prompt change to an 8B local model actually helped.

    python -m tests.harness                     # rule-based cleaner
    python -m tests.harness --cleaner ollama    # local Llama 3.1 / Gemma 2
    python -m tests.harness --cleaner gemini    # Google AI Studio free tier
    python -m tests.harness --tag slot-patch -v
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.text.cleaner import Cleaner, RuleBasedCleaner  # noqa: E402

CORPUS_PATH = Path(__file__).parent / "corpus" / "stem_cases.json"

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


@dataclass(slots=True)
class Case:
    id: str
    input: str
    expected: str
    tags: list[str]
    note: str = ""


@dataclass(slots=True)
class Outcome:
    case: Case
    actual: str
    exact: bool
    normalized: bool
    latency_ms: float


def load_corpus(path: Path = CORPUS_PATH) -> list[Case]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Case(
            id=c["id"],
            input=c["input"],
            expected=c["expected"],
            tags=c.get("tags", []),
            note=c.get("note", ""),
        )
        for c in data["cases"]
    ]


def normalize(text: str) -> str:
    """Case- and punctuation-insensitive form, for grading LLM output whose
    cosmetic choices differ but whose semantic edit is correct."""
    return _WS_RE.sub(" ", _PUNCT_RE.sub("", text.lower())).strip()


def run(cleaner: Cleaner, cases: list[Case]) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for case in cases:
        started = time.perf_counter()
        actual = cleaner.clean(case.input).text
        elapsed = (time.perf_counter() - started) * 1000
        outcomes.append(
            Outcome(
                case=case,
                actual=actual,
                exact=actual == case.expected,
                normalized=normalize(actual) == normalize(case.expected),
                latency_ms=elapsed,
            )
        )
    return outcomes


def report(outcomes: list[Outcome], engine: str, verbose: bool) -> bool:
    total = len(outcomes)
    exact = sum(o.exact for o in outcomes)
    near = sum(o.normalized for o in outcomes)
    avg_ms = sum(o.latency_ms for o in outcomes) / total if total else 0.0

    print(f"\n  cleaner: {engine}   cases: {total}   avg latency: {avg_ms:.2f} ms\n")

    for o in outcomes:
        if o.exact:
            mark, color = "PASS", GREEN
        elif o.normalized:
            mark, color = "NEAR", YELLOW
        else:
            mark, color = "FAIL", RED
        print(f"  {color}{mark}{RESET}  {o.case.id}")
        if not o.exact or verbose:
            print(f"        in       {o.case.input!r}")
            print(f"        expected {o.case.expected!r}")
            print(f"        actual   {o.actual!r}")
            if o.case.note:
                print(f"        {DIM}{o.case.note}{RESET}")

    by_tag: dict[str, list[Outcome]] = {}
    for o in outcomes:
        for tag in o.case.tags:
            by_tag.setdefault(tag, []).append(o)

    print("\n  by tag:")
    for tag in sorted(by_tag):
        group = by_tag[tag]
        hits = sum(g.exact for g in group)
        print(f"    {tag:<12} {hits}/{len(group)}")

    pct = 100 * exact / total if total else 0.0
    print(f"\n  exact {exact}/{total} ({pct:.0f}%)   normalized {near}/{total}\n")
    return exact == total


def build_cleaner(name: str) -> Cleaner:
    if name == "rules":
        return RuleBasedCleaner()
    # Imported lazily: the LLM backends pull in optional dependencies and the
    # rule harness must stay runnable on a bare interpreter.
    from openflow.llm.cleaner import LLMCleaner

    return LLMCleaner(provider=name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a transcript cleaner.")
    parser.add_argument("--cleaner", default="rules",
                        help="rules | ollama | gemini | groq (default: rules)")
    parser.add_argument("--tag", action="append", default=[],
                        help="only run cases carrying this tag (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show input/expected/actual for passing cases too")
    args = parser.parse_args(argv)

    cases = load_corpus()
    if args.tag:
        wanted = set(args.tag)
        cases = [c for c in cases if wanted & set(c.tags)]
        if not cases:
            print(f"no cases match tags {sorted(wanted)}", file=sys.stderr)
            return 2

    cleaner = build_cleaner(args.cleaner)
    ok = report(run(cleaner, cases), args.cleaner, args.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
