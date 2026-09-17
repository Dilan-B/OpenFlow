"""Wispr Flow Smart Formatting and Flow Styles.

Cases marked "documented" reproduce the examples in Wispr's help center
("How do I use Smart Formatting & Backtrack"). The rest pin the documented
rules on other sentences, plus the guards that keep ordinary prose from being
mistaken for a list, a number or a name.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.config import Config  # noqa: E402
from openflow.formatting import (  # noqa: E402
    CASUAL, EXCITED, FORMAL, VERY_CASUAL, AppKind, CaretContext, classify,
    smart_format,
)
from openflow.profiles import PROFILES  # noqa: E402
from openflow.text.autofix import apply_symbol_fixes  # noqa: E402
from openflow.text.lists import format_lists  # noqa: E402
from openflow.text.numbers import format_numbers  # noqa: E402
from openflow.text.spoken import apply_spoken_punctuation  # noqa: E402

SLACK = AppKind("work", True)
WHATSAPP = AppKind("personal", True)
OUTLOOK = AppKind("email", False)


class SpokenPunctuation(unittest.TestCase):
    def test_documented_exclamation_and_period(self):
        self.assertEqual(
            apply_spoken_punctuation(
                "I can't wait to see you exclamation point Let's meet at seven period"),
            "I can't wait to see you! Let's meet at seven.")

    def test_every_documented_line_break(self):
        for spoken, mark in (("new line", "\n"), ("next line", "\n"),
                             ("line break", "\n"), ("skip a line", "\n\n"),
                             ("new paragraph", "\n\n"),
                             ("start a new paragraph", "\n\n")):
            self.assertEqual(apply_spoken_punctuation(f"one {spoken} two"),
                             f"one{mark}two", spoken)

    def test_em_dash_joins_words(self):
        self.assertEqual(apply_spoken_punctuation("it was fast em dash really fast"),
                         "it was fast—really fast")

    def test_quotation_mark_alternates_open_and_close(self):
        self.assertEqual(
            apply_spoken_punctuation("she said quotation mark hello quotation mark"),
            "she said “hello”")

    def test_semicolon_and_ellipsis(self):
        self.assertEqual(apply_spoken_punctuation("wait semicolon then ellipsis"),
                         "wait; then…")


class SpokenSymbols(unittest.TestCase):
    def fix(self, text: str) -> str:
        return apply_symbol_fixes(text)[0]

    def test_email_from_underscore_and_at_symbol(self):
        self.assertEqual(self.fix("john underscore smith at symbol gmail.com"),
                         "john_smith@gmail.com")

    def test_hashtag_and_handle_attach_forward(self):
        self.assertEqual(self.fix("launch hashtag product"), "launch #product")
        self.assertEqual(self.fix("ping at sign dilan"), "ping @dilan")

    def test_units_attach_backward(self):
        self.assertEqual(self.fix("it is 20 degrees celsius"), "it is 20°C")
        self.assertEqual(self.fix("Acme trademark symbol"), "Acme™")

    def test_parentheses_hug_their_contents(self):
        self.assertEqual(self.fix("see open paren below close paren"), "see (below)")


class Numbers(unittest.TestCase):
    def test_documented_time(self):
        self.assertEqual(format_numbers("Let's meet at seven."), "Let's meet at 7.")

    def test_clock_times(self):
        self.assertEqual(format_numbers("call me at seven thirty pm"), "call me at 7:30 PM")
        self.assertEqual(format_numbers("meet at five tomorrow"), "meet at 5 tomorrow")

    def test_percent_money_and_large_numbers(self):
        self.assertEqual(format_numbers("fifty percent"), "50%")
        self.assertEqual(format_numbers("twenty dollars"), "$20")
        self.assertEqual(format_numbers("twenty five people"), "25 people")
        self.assertEqual(format_numbers("the twenty first"), "the 21st")

    def test_small_numbers_and_idioms_stay_words(self):
        for text in ("one of the reasons", "no one came", "at one point we left",
                     "two options", "the fifth", "five six seven"):
            self.assertEqual(format_numbers(text), text)


class Lists(unittest.TestCase):
    def test_documented_list(self):
        self.assertEqual(
            format_lists("My top goals this week are one finish the report two send "
                         "the presentation"),
            "My top goals this week are:\n1. Finish the report\n2. Send the presentation")

    def test_punctuated_ordinals(self):
        self.assertEqual(format_lists("First, open the app. Second, log in."),
                         "1. Open the app\n2. Log in")

    def test_text_after_the_list_becomes_its_own_paragraph(self):
        self.assertEqual(
            format_lists("Number one we ship. Number two we celebrate. Thanks everyone."),
            "1. We ship\n2. We celebrate\n\nThanks everyone.")

    def test_sentences_that_merely_contain_numbers_are_left_alone(self):
        for text in ("I have one cat and two dogs.", "They are one of two companies.",
                     "At first I thought it was the second option.",
                     "Give me one minute and two seconds."):
            self.assertEqual(format_lists(text), text)


class MessagingApps(unittest.TestCase):
    def test_trailing_period_dropped_for_short_messages(self):
        self.assertEqual(smart_format("Sounds good.", kind=SLACK), "Sounds good")
        self.assertEqual(smart_format("Sounds good. See you soon.", kind=SLACK),
                         "Sounds good. See you soon")

    def test_kept_past_two_sentences(self):
        self.assertEqual(smart_format("One. Two. Three.", kind=SLACK), "One. Two. Three.")

    def test_question_and_exclamation_marks_always_survive(self):
        self.assertEqual(smart_format("Are we shipping?", kind=SLACK), "Are we shipping?")
        self.assertEqual(smart_format("We shipped!", kind=SLACK), "We shipped!")

    def test_kept_when_replacing_a_selection(self):
        self.assertEqual(
            smart_format("Sounds good.", kind=SLACK,
                         context=CaretContext(has_selection=True)),
            "Sounds good.")

    def test_not_applied_outside_messaging_apps(self):
        self.assertEqual(smart_format("Sounds good.", kind=OUTLOOK), "Sounds good.")

    def test_smart_formatting_off_keeps_the_old_chat_profile(self):
        self.assertEqual(smart_format("Sounds good.", profile=PROFILES["chat"], smart=False),
                         "Sounds good")


class FlowStyles(unittest.TestCase):
    def test_formal_changes_nothing(self):
        self.assertEqual(smart_format("We shipped it.", kind=OUTLOOK, style=FORMAL),
                         "We shipped it.")

    def test_casual_drops_the_trailing_period(self):
        self.assertEqual(smart_format("We shipped it. Thanks.", kind=OUTLOOK, style=CASUAL),
                         "We shipped it. Thanks")

    def test_very_casual_lowercases_but_keeps_names_and_i(self):
        self.assertEqual(
            smart_format("Sounds good. I'll see Sarah soon.", kind=WHATSAPP,
                         style=VERY_CASUAL),
            "sounds good. I'll see Sarah soon")

    def test_very_casual_never_lowercases_a_name(self):
        self.assertEqual(smart_format("Sarah is here.", kind=WHATSAPP, style=VERY_CASUAL),
                         "Sarah is here")

    def test_very_casual_respects_dictionary_terms(self):
        self.assertEqual(
            smart_format("Groq works.", kind=WHATSAPP, style=VERY_CASUAL,
                         protected_terms=("Groq",)),
            "Groq works")

    def test_very_casual_is_personal_only(self):
        self.assertEqual(smart_format("Sounds good.", kind=OUTLOOK, style=VERY_CASUAL),
                         "Sounds good.")

    def test_excited_adds_an_exclamation(self):
        self.assertEqual(smart_format("We shipped it.", kind=OUTLOOK, style=EXCITED),
                         "We shipped it!")

    def test_code_and_shell_ignore_styles(self):
        self.assertEqual(smart_format("Git status.", profile=PROFILES["shell"],
                                      style=CASUAL), "git status")


class CaretAwareness(unittest.TestCase):
    def test_mid_sentence_dictation_is_lowercased_and_spaced(self):
        self.assertEqual(
            smart_format("Ship it on Friday.",
                         context=CaretContext(before="I think we should")),
            " ship it on Friday.")

    def test_after_a_full_stop_it_stays_capitalised(self):
        self.assertEqual(smart_format("Ship it.", context=CaretContext(before="Done.")),
                         " Ship it.")

    def test_a_name_is_not_lowercased_mid_sentence(self):
        self.assertEqual(
            smart_format("Sarah will do it.", context=CaretContext(before="I told")),
            " Sarah will do it.")

    def test_inserting_before_text_adds_the_trailing_space(self):
        self.assertEqual(
            smart_format("Really", context=CaretContext(before="It's ", after="good.")),
            "really ")

    def test_existing_punctuation_after_the_caret_owns_the_ending(self):
        self.assertEqual(
            smart_format("Ship it on Friday.",
                         context=CaretContext(before="We ", after=". Then")),
            "ship it on Friday")

    def test_no_space_after_an_opening_bracket(self):
        # Inside an open parenthetical is mid-sentence, so it lowercases too.
        self.assertEqual(smart_format("See below.", context=CaretContext(before="(")),
                         "see below.")


class Classification(unittest.TestCase):
    def test_documented_default_categories(self):
        self.assertEqual(classify("whatsapp.exe"), AppKind("personal", True))
        self.assertEqual(classify("slack.exe"), AppKind("work", True))
        self.assertEqual(classify("outlook.exe"), AppKind("email", False))
        self.assertEqual(classify("winword.exe"), AppKind("other", False))

    def test_web_apps_by_tab_title(self):
        self.assertEqual(
            classify("chrome.exe", "Inbox (3) - me@gmail.com - Gmail - Google Chrome").category,
            "email")
        self.assertEqual(
            classify("msedge.exe", "(3) WhatsApp - Personal - Microsoft​ Edge"),
            AppKind("personal", True))
        self.assertEqual(classify("chrome.exe", "Home / X - Google Chrome"),
                         AppKind("other", True))

    def test_an_article_about_an_app_is_not_the_app(self):
        self.assertEqual(
            classify("chrome.exe", "Why Slack won the market - The Verge - Google Chrome"),
            AppKind())

    def test_user_override(self):
        self.assertEqual(classify("winword.exe", overrides={"winword.exe": "email"}).category,
                         "email")


class CaretReaderFailsSafe(unittest.TestCase):
    def test_any_accessibility_error_means_no_context(self):
        from openflow.input import caret

        with mock.patch.object(caret, "_read_windows", side_effect=OSError("denied")), \
                mock.patch.object(caret, "_read_macos", side_effect=OSError("denied")):
            self.assertIsNone(caret.read_caret_context())

    def test_a_slow_read_is_abandoned(self):
        import threading

        from openflow.input import caret

        release = threading.Event()
        with mock.patch.object(caret, "read_caret_context",
                               side_effect=lambda: release.wait(2) and None):
            reader = caret.CaretReader().start()
            self.assertIsNone(reader.result(wait_s=0.05))
        release.set()


class StyleMigration(unittest.TestCase):
    def test_legacy_casual_preset_carries_into_flow_styles(self):
        home = Path(tempfile.mkdtemp())
        (home / "personalization.json").write_text(json.dumps({"style": "casual"}))
        path = home / "config.json"
        path.write_text(json.dumps({"schema": 2}))
        with mock.patch("openflow.config.CONFIG_DIR", home):
            cfg = Config.load(path)
        self.assertEqual(set(cfg.formatting.styles.values()), {"casual"})

    def test_other_legacy_presets_stay_formal(self):
        home = Path(tempfile.mkdtemp())
        (home / "personalization.json").write_text(json.dumps({"style": "professional"}))
        path = home / "config.json"
        path.write_text(json.dumps({"schema": 2}))
        with mock.patch("openflow.config.CONFIG_DIR", home):
            cfg = Config.load(path)
        self.assertEqual(set(cfg.formatting.styles.values()), {"formal"})


if __name__ == "__main__":
    unittest.main()
