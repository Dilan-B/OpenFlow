"""Tests for the dictionary repair, snippets, spoken punctuation, and streaks."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.history import Entry, History  # noqa: E402
from openflow.personalization import Personalization  # noqa: E402
from openflow.text.cleaner import RuleBasedCleaner  # noqa: E402
from openflow.text.spoken import apply_spoken_punctuation  # noqa: E402


def _personal(**kwargs) -> Personalization:
    tmp = tempfile.mkdtemp()
    p = Personalization.load(Path(tmp) / "p.json")
    for term in kwargs.get("terms", []):
        p.add_term(term)
    for trigger, body in kwargs.get("snippets", []):
        p.add_snippet(trigger, body)
    return p


class DictionaryRepair(unittest.TestCase):
    def test_fuzzy_fixes_misheard_term(self):
        p = _personal(terms=["OpenFlow"])
        self.assertEqual("I love OpenFlow so much",
                         p.apply_dictionary("I love openflo so much"))

    def test_joins_split_compound(self):
        p = _personal(terms=["OpenFlow"])
        self.assertEqual("try OpenFlow today", p.apply_dictionary("try open flow today"))

    def test_case_repair_for_short_terms(self):
        p = _personal(terms=["Groq"])
        self.assertEqual("use the Groq API", p.apply_dictionary("use the groq api".replace("api", "API")))

    def test_short_terms_never_fuzzy_match(self):
        p = _personal(terms=["Groq"])
        # "grow" is one edit from "groq" -- a real word must survive.
        self.assertEqual("plants grow fast", p.apply_dictionary("plants grow fast"))

    def test_multiword_term(self):
        p = _personal(terms=["Wispr Flow"])
        self.assertEqual("a Wispr Flow clone", p.apply_dictionary("a wispr flow clone"))

    def test_no_terms_is_identity(self):
        p = _personal()
        self.assertEqual("anything at all", p.apply_dictionary("anything at all"))

    def test_vocabulary_hint_budget(self):
        p = _personal(terms=["Alpha", "Beta", "Gamma"])
        hint = p.vocabulary_hint()
        self.assertIn("Alpha", hint)
        self.assertIn("Gamma", hint)


class Snippets(unittest.TestCase):
    def test_whole_utterance_trigger_expands(self):
        p = _personal(snippets=[("my email sig", "Best,\nDilan Bhimani")])
        self.assertEqual("Best,\nDilan Bhimani", p.apply_snippets("My email sig."))

    def test_partial_match_does_not_expand(self):
        p = _personal(snippets=[("my email sig", "Best,\nDilan")])
        text = "add my email sig to the doc"
        self.assertEqual(text, p.apply_snippets(text))


class SpokenPunctuation(unittest.TestCase):
    def test_trailing_period(self):
        self.assertEqual("send it now.", apply_spoken_punctuation("send it now period"))

    def test_question_mark_anywhere(self):
        self.assertEqual("did you ship it?",
                         apply_spoken_punctuation("did you ship it question mark"))

    def test_new_paragraph(self):
        out = apply_spoken_punctuation("first point new paragraph second point")
        self.assertEqual("first point\n\nsecond point", out)

    def test_mid_sentence_period_is_a_word(self):
        text = "that period of time was hard"
        self.assertEqual(text, apply_spoken_punctuation(text))

    def test_full_pipeline_keeps_paragraphs(self):
        cleaner = RuleBasedCleaner()
        out = cleaner.clean("um first thing new paragraph second thing period").text
        self.assertEqual("First thing.\n\nSecond thing.", out)

    def test_full_pipeline_question(self):
        out = RuleBasedCleaner().clean("can you review it question mark").text
        self.assertEqual("Can you review it?", out)


class Streaks(unittest.TestCase):
    def test_streak_counts_consecutive_days(self):
        tmp = tempfile.mkdtemp()
        history = History(Path(tmp) / "h.json")
        now = time.time()
        for days_ago in (0, 1, 2):
            history.add(Entry(at=now - days_ago * 86400, words=10, chars=50,
                              duration_s=4, latency_ms=900, stt_engine="x",
                              clean_engine="rules", retractions=0))
        self.assertEqual(3, history.stats.streak_days)
        self.assertEqual(7, len(history.stats.last_days(7)))
        self.assertEqual(30, history.stats.words)


if __name__ == "__main__":
    unittest.main(verbosity=2)
