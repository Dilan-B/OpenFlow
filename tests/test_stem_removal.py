"""Unit tests for the deterministic cleanup pipeline.

    python -m unittest discover -s tests

The corpus cases are pulled in as generated test methods so `unittest` failure
output names the exact case that regressed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.text.cleaner import RuleBasedCleaner  # noqa: E402
from openflow.text.fillers import strip_fillers  # noqa: E402
from openflow.text.punctuation import fix_punctuation, strip_wrapping_quotes  # noqa: E402
from openflow.text.stem_removal import remove_false_starts  # noqa: E402
from tests.harness import load_corpus  # noqa: E402


class GoldenCorpus(unittest.TestCase):
    """One test per corpus case, generated below."""


def _make_case_test(case):
    def test(self):
        actual = RuleBasedCleaner().clean(case.input).text
        self.assertEqual(case.expected, actual, msg=f"\n  case: {case.id}\n  {case.note}")

    test.__name__ = f"test_{case.id.replace('-', '_')}"
    return test


for _case in load_corpus():
    setattr(GoldenCorpus, f"test_{_case.id.replace('-', '_')}", _make_case_test(_case))


class StemRemovalInternals(unittest.TestCase):
    def test_reports_strategy_and_removed_span(self):
        _, retractions = remove_false_starts(
            "Can we meet up on tuesday at 5, or actually, can we meet up on friday at 3."
        )
        self.assertEqual(1, len(retractions))
        self.assertEqual("re-anchor", retractions[0].strategy)
        self.assertEqual("or actually", retractions[0].pivot)
        self.assertIn("tuesday", retractions[0].removed)

    def test_slot_patch_strategy(self):
        _, retractions = remove_false_starts("Let's meet Tuesday at 5, actually Friday at 3.")
        self.assertEqual("slot-patch", retractions[0].strategy)

    def test_no_pivot_is_untouched(self):
        text = "The workers pick up jobs after boot."
        out, retractions = remove_false_starts(text)
        self.assertEqual(text, out)
        self.assertEqual([], retractions)

    def test_retraction_does_not_cross_sentence_boundary(self):
        out, _ = remove_false_starts("Keep the logs. Ship on tuesday, or actually ship on friday.")
        self.assertIn("Keep the logs.", out)

    def test_bails_out_rather_than_guessing(self):
        _, retractions = remove_false_starts("The build passes locally, or actually in staging too.")
        self.assertEqual("pivot-only", retractions[0].strategy)
        self.assertEqual("", retractions[0].removed)


class Fillers(unittest.TestCase):
    def test_hard_filler_anywhere(self):
        out, removed = strip_fillers("we should um ship it")
        self.assertEqual("we should ship it", out)
        self.assertIn("um", removed)

    def test_soft_filler_only_in_discourse_position(self):
        self.assertEqual("I like it", strip_fillers("I like it")[0])
        self.assertEqual("it took three hours", strip_fillers("it took, like, three hours")[0])

    def test_multiword_filler(self):
        out, _ = strip_fillers("the index, you know, needs a rebuild")
        self.assertEqual("the index needs a rebuild", out)


class Punctuation(unittest.TestCase):
    def test_sentence_case_and_terminal_period(self):
        self.assertEqual("Ship it.", fix_punctuation("ship it"))

    def test_preserves_technical_capitalization(self):
        self.assertEqual("Run the PostgreSQL migration.", fix_punctuation("run the PostgreSQL migration"))

    def test_capitalizes_weekdays_and_pronoun_i(self):
        self.assertEqual("On Friday i'll ship.".replace("i'll", "I'll"),
                         fix_punctuation("on friday i'll ship"))

    def test_strips_llm_quote_wrapping(self):
        self.assertEqual("Ship it.", strip_wrapping_quotes('"Ship it."'))
        self.assertEqual('He said "go" now', strip_wrapping_quotes('He said "go" now'))


class Performance(unittest.TestCase):
    def test_cleanup_is_fast_enough_to_run_inline(self):
        import time

        text = ("Can we meet up on tuesday at 5, or actually, can we meet up on friday at 3. " * 20)
        cleaner = RuleBasedCleaner()
        started = time.perf_counter()
        cleaner.clean(text)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.assertLess(elapsed_ms, 50, "rule pass must stay far below perceptible latency")


if __name__ == "__main__":
    unittest.main(verbosity=2)
