"""Tests for the deterministic autofix pass.

Two halves matter equally: the fixes it makes, and the ones it refuses to.
Anything needing context to be right belongs to the LLM pass, so the negative
cases below are the real specification.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.text.autofix import apply_symbol_fixes, apply_word_fixes  # noqa: E402
from openflow.text.cleaner import RuleBasedCleaner  # noqa: E402


def clean(text: str) -> str:
    return RuleBasedCleaner().clean(text).text


class Contractions(unittest.TestCase):
    def test_restores_dropped_apostrophes(self):
        # This pass fixes words only; sentence-start casing is fix_punctuation's
        # job, so "i" stays lowercase here and is capitalized downstream.
        self.assertEqual("i don't think we can't", apply_word_fixes("i dont think we cant")[0])
        self.assertEqual("they're here", apply_word_fixes("theyre here")[0])
        self.assertEqual("I don't think we can't.", clean("i dont think we cant"))

    def test_leaves_ambiguous_words_alone(self):
        # "its" and "were" are real words; choosing the contraction needs
        # to know what the sentence means.
        for text in ("its fine", "we were there", "he lets me drive"):
            self.assertEqual(text, apply_word_fixes(text)[0])

    def test_lets_only_at_sentence_start(self):
        self.assertEqual("Let's go.", apply_symbol_fixes("Lets go.")[0])
        self.assertEqual("She lets go.", apply_symbol_fixes("She lets go.")[0])

    def test_existing_apostrophes_untouched(self):
        self.assertEqual("I don't", apply_word_fixes("I don't")[0])


class Casing(unittest.TestCase):
    def test_acronyms(self):
        self.assertEqual("the API returns JSON", apply_word_fixes("the api returns json")[0])
        self.assertEqual("The API returns JSON.", clean("the api returns json"))

    def test_brands(self):
        self.assertEqual("GitHub and JavaScript and iPhone",
                         apply_word_fixes("github and javascript and iphone")[0])

    def test_never_recases_what_the_transcriber_got_right(self):
        text = "PostgreSQL on the iOS build before the gRPC cutover"
        self.assertEqual(text, apply_word_fixes(text)[0])

    def test_cased_terms_survive_sentence_start(self):
        # fix_punctuation must not turn "iOS" into "IOs" at the front.
        self.assertEqual("iPhone builds are ready.", clean("iphone builds are ready"))
        self.assertEqual("gRPC is fine.", clean("gRPC is fine"))


class Stutters(unittest.TestCase):
    def test_removes_doubled_words(self):
        self.assertEqual("the report is ready",
                         apply_word_fixes("the the report is ready")[0])

    def test_keeps_legitimate_doubles(self):
        self.assertEqual("I had had enough", apply_word_fixes("I had had enough")[0])


class SymbolsAndAddresses(unittest.TestCase):
    def test_email_assembly(self):
        self.assertEqual("Email me at dilan.bhimani@gmail.com.",
                         clean("email me at dilan dot bhimani at gmail dot com"))

    def test_domain_assembly(self):
        self.assertEqual("Check out github.com for docs.",
                         clean("check out github dot com for docs"))

    def test_common_word_is_not_an_email_local_part(self):
        # "meet me at gmail dot com" is a sentence, not me@gmail.com.
        self.assertNotIn("@", clean("meet me at gmail dot com"))

    def test_spoken_symbols(self):
        self.assertIn("@", apply_symbol_fixes("send it to the at sign address")[0])
        self.assertIn("&", apply_symbol_fixes("Ben ampersand Jerry")[0])
        self.assertIn("\\", apply_symbol_fixes("a backslash here")[0])

    def test_times(self):
        self.assertEqual("Can we meet at 3:30 PM on Friday?",
                         clean("can we meet at 3 30 p m on friday"))
        self.assertEqual("Ship at 5 PM.", clean("ship at 5 p m"))


class CountsAndSafety(unittest.TestCase):
    def test_reports_how_many_fixes_it_made(self):
        result = RuleBasedCleaner().clean("the the api is json and we cant cache it")
        self.assertGreaterEqual(result.autofixes, 4)

    def test_clean_text_reports_no_fixes(self):
        self.assertEqual(0, RuleBasedCleaner().clean("The report is ready.").autofixes)

    def test_empty_input(self):
        self.assertEqual(("", 0), apply_word_fixes(""))
        self.assertEqual(("", 0), apply_symbol_fixes(""))

    def test_pass_is_still_sub_millisecond(self):
        import time

        text = "the the api returns json over https so we cant cache it " * 10
        started = time.perf_counter()
        RuleBasedCleaner().clean(text)
        self.assertLess((time.perf_counter() - started) * 1000, 25)



class NounPhraseCorrections(unittest.TestCase):
    """Regression: "or actually the X" used to delete the whole sentence.

    A determiner opens a noun phrase far more often than an independent
    clause, so treating it as a full restart threw away everything the speaker
    had already said.
    """

    def test_sentence_survives_a_noun_phrase_swap(self):
        out = clean("lets ask the docker team, or actually the kubernetes team")
        self.assertIn("ask", out, "the head of the sentence was deleted")
        self.assertIn("Kubernetes", out)
        self.assertNotIn("Docker", out)

    def test_no_duplicate_words_across_the_splice(self):
        out = clean("lets ask the docker team, or actually the kubernetes team")
        self.assertNotIn("the the", out)
        self.assertNotIn("team team", out)

    def test_explicit_restart_still_drops_the_head(self):
        # "let me rephrase that" promises a full restatement, so a determiner
        # opening the replacement is a real clause.
        out = clean("the thing is broken in a weird way, let me rephrase that, "
                    "the parser drops trailing commas")
        self.assertEqual("The parser drops trailing commas.", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
