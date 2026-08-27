"""Tests for stutter / repeated-phrase collapse.

The risk in this pass is over-deletion: English really does double words, and
silently eating one changes meaning ("he had enough" is not "he had had
enough"). Most of these cases are therefore negatives.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.text.repetitions import collapse_repetitions  # noqa: E402


class Stutters(unittest.TestCase):
    def assert_clean(self, raw: str, expected: str):
        self.assertEqual(collapse_repetitions(raw)[0], expected)

    def test_doubled_word(self):
        self.assert_clean("I I think we should ship", "I think we should ship")

    def test_tripled_word(self):
        self.assert_clean("the the the deploy goes out at noon",
                          "the deploy goes out at noon")

    def test_repeated_phrase(self):
        self.assert_clean("can we can we meet on Friday", "can we meet on Friday")

    def test_longer_repeated_phrase(self):
        self.assert_clean("we should we should probably wait",
                          "we should probably wait")

    def test_the_surviving_copy_keeps_following_punctuation(self):
        self.assert_clean("send send the file, then wait", "send the file, then wait")

    def test_reports_what_it_collapsed(self):
        _, hits = collapse_repetitions("can we can we meet")
        self.assertEqual(hits, ["can we"])


class ThingsItMustNotTouch(unittest.TestCase):
    def assert_unchanged(self, raw: str):
        self.assertEqual(collapse_repetitions(raw)[0], raw)

    def test_non_adjacent_repeats_are_normal_english(self):
        self.assert_unchanged("the cat sat on the mat")

    def test_had_had(self):
        self.assert_unchanged("he had had enough of it")

    def test_that_that(self):
        self.assert_unchanged("I think that that is wrong")

    def test_intensifiers_are_emphasis_not_stutter(self):
        self.assert_unchanged("it was very very good")
        self.assert_unchanged("that is really really bad")

    def test_punctuation_between_copies_means_it_was_deliberate(self):
        self.assert_unchanged("no, no, keep it")

    def test_a_sentence_boundary_is_not_a_stutter(self):
        self.assert_unchanged("Go. Go now.")

    def test_empty_and_single_word(self):
        self.assert_unchanged("")
        self.assert_unchanged("hello")

    def test_repeat_across_a_full_stop_is_left_alone(self):
        self.assert_unchanged("Ship it. Ship it today.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
