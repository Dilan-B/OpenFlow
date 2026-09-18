"""Texting mode, saved API keys, and the start chime."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow import keys  # noqa: E402
from openflow.config import Config, api_key  # noqa: E402
from openflow.formatting import (  # noqa: E402
    STYLES_FOR, TEXTING, AppKind, CaretContext, smart_format,
)
from openflow.profiles import PROFILES  # noqa: E402

WHATSAPP = AppKind("personal", True)
OUTLOOK = AppKind("email", False)


def texting(text: str, **kw) -> str:
    return smart_format(text, kind=kw.pop("kind", WHATSAPP), texting=True, **kw)


class TextingStyle(unittest.TestCase):
    def test_drops_caps_periods_and_commas(self):
        self.assertEqual(texting("Yeah, sounds good. Talk later."),
                         "yeah sounds good talk later")

    def test_keeps_questions_and_exclamations(self):
        self.assertEqual(texting("Wait, really? That's amazing!"),
                         "wait really? that's amazing!")

    def test_keeps_names_and_i(self):
        self.assertEqual(texting("Meet John at Starbucks. I'm late."),
                         "meet John at Starbucks I'm late")
        self.assertEqual(texting("Priya said hi. API is down."),
                         "priya said hi API is down")
        self.assertEqual(
            texting("Priya said hi.", protected_terms=("Priya",)), "Priya said hi")

    def test_keeps_numbers_initialisms_urls_and_ellipses(self):
        self.assertEqual(texting("It's 1,250 dollars, 3.5 each."),
                         "it's 1,250 dollars 3.5 each")
        self.assertEqual(texting("I moved to the U.S. last year."),
                         "I moved to the U.S. last year")
        self.assertEqual(texting("See google.com, it's good."),
                         "see google.com it's good")
        self.assertEqual(texting("Well... okay."), "well... okay")

    def test_applies_in_any_app_when_forced(self):
        self.assertEqual(texting("Thanks. See you.", kind=OUTLOOK), "thanks see you")

    def test_leaves_code_and_lists_alone(self):
        code = smart_format("return x.", kind=WHATSAPP, texting=True,
                            profile=PROFILES["code"])
        self.assertNotEqual(code, "")
        listed = "Groceries:\n1. Milk.\n2. Eggs."
        self.assertEqual(texting(listed), smart_format(listed, kind=WHATSAPP))

    def test_mid_sentence_still_fits_the_caret(self):
        self.assertEqual(
            texting("Sounds good.", context=CaretContext(before="ok")),
            " sounds good")

    def test_is_a_personal_style_and_off_by_default(self):
        self.assertIn(TEXTING, STYLES_FOR["personal"])
        self.assertNotIn(TEXTING, STYLES_FOR["email"])
        self.assertEqual(
            smart_format("Okay. Talk later.", kind=WHATSAPP, style=TEXTING),
            "okay talk later")
        self.assertEqual(
            smart_format("Okay. Talk later.", kind=OUTLOOK), "Okay. Talk later.")
        self.assertFalse(Config().formatting.texting)


class SavedKeys(unittest.TestCase):
    def setUp(self):
        keys._cache.clear()
        self.addCleanup(keys._cache.clear)

    def test_environment_wins_over_the_keychain(self):
        with mock.patch.dict("os.environ", {"GROQ_API_KEY": "from-env"}), \
                mock.patch.object(keys, "load", return_value="from-keychain"):
            self.assertEqual(api_key("GROQ_API_KEY"), "from-env")

    def test_falls_back_to_the_keychain(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.object(keys, "load", return_value="from-keychain"):
            self.assertEqual(api_key("GROQ_API_KEY"), "from-keychain")

    def test_rejects_malformed_keys_before_touching_the_keychain(self):
        with mock.patch.object(keys, "supported", return_value=True), \
                mock.patch("subprocess.run") as run:
            for bad in ("", "short", 'has"quote-in-it', "has space in it x"):
                with self.assertRaises(ValueError):
                    keys.save("GROQ_API_KEY", bad)
            with self.assertRaises(ValueError):
                keys.save("SOME_OTHER_KEY", "gsk_abcdefghijklmnop")
            run.assert_not_called()

    def test_key_goes_through_stdin_not_argv(self):
        done = mock.Mock(returncode=0, stderr="", stdout="")
        with mock.patch.object(keys, "supported", return_value=True), \
                mock.patch("subprocess.run", return_value=done) as run:
            keys.save("GROQ_API_KEY", "gsk_abcdefghijklmnop")
            args, kwargs = run.call_args
            self.assertEqual(args[0], ["security", "-i"])
            self.assertNotIn("gsk_abcdefghijklmnop", " ".join(args[0]))
            self.assertIn("gsk_abcdefghijklmnop", kwargs["input"])
            # Served from the cache the save filled, without another lookup.
            run.reset_mock()
            self.assertEqual(keys.load("GROQ_API_KEY"), "gsk_abcdefghijklmnop")
            run.assert_not_called()

    def test_unsupported_platform_reads_nothing(self):
        with mock.patch.object(keys, "supported", return_value=False), \
                mock.patch("subprocess.run") as run:
            self.assertIsNone(keys.load("GROQ_API_KEY"))
            run.assert_not_called()

    def test_groq_whisper_sees_a_key_saved_after_startup(self):
        from openflow.stt.engines import build_engine

        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.object(keys, "load", return_value=None):
            engine = build_engine("groq", Config())
            self.assertFalse(engine.available())
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.object(keys, "load", return_value="gsk_saved_later"):
            self.assertTrue(engine.available())


class Chime(unittest.TestCase):
    def test_is_short_quiet_and_clean(self):
        from openflow.audio import chime

        samples = chime.samples()
        seconds = len(samples) / chime.SAMPLE_RATE
        self.assertLess(seconds, 0.25)
        self.assertLessEqual(float(abs(samples).max()), chime.VOLUME + 1e-6)
        self.assertAlmostEqual(float(samples[0]), 0.0, places=3)
        self.assertAlmostEqual(float(samples[-1]), 0.0, places=3)

    def test_play_never_raises(self):
        from openflow.audio import chime

        with mock.patch.dict("sys.modules", {"sounddevice": None}):
            chime.play()


if __name__ == "__main__":
    unittest.main(verbosity=2)
