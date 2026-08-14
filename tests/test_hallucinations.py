"""Tests for the silence-hallucination gate.

The gate has to be narrow. "Thank you" is a thing people genuinely dictate, so
discarding it on the strength of the phrase alone would make the app look
broken in a way that is very hard to diagnose from the user's side.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.text.hallucinations import (  # noqa: E402
    SUSPECT_DURATION_S, is_silence_hallucination, normalize,
)

LOUD = 0.05        # comfortably above the quiet threshold
QUIET = 0.001
LONG = SUSPECT_DURATION_S + 2.0
SHORT = 0.5


class Discards(unittest.TestCase):
    def test_stock_phrase_on_a_short_clip(self):
        self.assertTrue(is_silence_hallucination("Thank you.", SHORT, LOUD))

    def test_stock_phrase_on_a_quiet_clip(self):
        self.assertTrue(is_silence_hallucination("Thank you.", LONG, QUIET))

    def test_subtitle_credit(self):
        self.assertTrue(is_silence_hallucination(
            "Subtitles by the Amara.org community", SHORT, LOUD))

    def test_thanks_for_watching(self):
        self.assertTrue(is_silence_hallucination("Thanks for watching!", SHORT, LOUD))

    def test_punctuation_and_case_do_not_matter(self):
        self.assertTrue(is_silence_hallucination("  BYE!!  ", SHORT, LOUD))

    def test_bare_filler(self):
        self.assertTrue(is_silence_hallucination("Um.", SHORT, LOUD))


class Keeps(unittest.TestCase):
    def test_a_real_thank_you_that_was_long_and_loud_enough(self):
        # Somebody dictating "thank you" into an email. Both audio signals say
        # this was really spoken, so the phrase alone must not condemn it.
        self.assertFalse(is_silence_hallucination("Thank you.", LONG, LOUD))

    def test_a_stock_phrase_inside_a_real_sentence(self):
        self.assertFalse(is_silence_hallucination(
            "Thank you for sending the report over.", SHORT, LOUD))

    def test_ordinary_text_on_a_short_clip(self):
        self.assertFalse(is_silence_hallucination("Ship it.", SHORT, QUIET))

    def test_empty_transcript_is_not_a_hallucination(self):
        # Nothing to paste either way; the caller handles empty separately.
        self.assertFalse(is_silence_hallucination("", SHORT, QUIET))
        self.assertFalse(is_silence_hallucination("   ", SHORT, QUIET))

    def test_boundary_duration_is_inclusive(self):
        self.assertTrue(is_silence_hallucination("Bye.", SUSPECT_DURATION_S, LOUD))
        self.assertFalse(is_silence_hallucination(
            "Bye.", SUSPECT_DURATION_S + 0.01, LOUD))


class Normalization(unittest.TestCase):
    def test_folds_case_punctuation_and_whitespace(self):
        self.assertEqual(normalize("  Thank   YOU!!  "), "thank you")

    def test_handles_none_and_empty(self):
        self.assertEqual(normalize(""), "")
        self.assertEqual(normalize(None), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
