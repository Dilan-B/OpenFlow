"""Tests for the first-run name prompt.

The greeting used to read straight from %USERNAME%, so it showed whatever the
Windows account was called -- a login id, an asset tag, someone else's name on
a shared machine. These pin the replacement: ask once, remember the answer,
and never let a skipped prompt come back.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.config import Config  # noqa: E402

try:
    from PySide6.QtWidgets import QApplication

    from openflow.ui.main_window import _account_name, _first_name
    from openflow.ui.welcome import WelcomeDialog, ask_for_name, clean_name

    QT_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on install extras
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, "PySide6 not installed")
class NameCleaning(unittest.TestCase):
    def test_whitespace_is_collapsed(self):
        self.assertEqual(clean_name("  Ada   Lovelace \n"), "Ada Lovelace")

    def test_control_characters_are_dropped(self):
        # The value lands in a QLabel and a JSON file; newlines would wreck both.
        self.assertEqual(clean_name("Ada\r\nLovelace\t"), "Ada Lovelace")

    def test_overlong_input_is_truncated(self):
        self.assertLessEqual(len(clean_name("x" * 500)), 40)

    def test_empty_stays_empty(self):
        for junk in ("", "   ", "\n\t", None):
            self.assertEqual(clean_name(junk), "")


@unittest.skipUnless(QT_AVAILABLE, "PySide6 not installed")
class GreetingFallback(unittest.TestCase):
    def test_configured_name_wins(self):
        self.assertEqual(_first_name("Grace"), "Grace")

    def test_falls_back_to_the_account_name(self):
        self.assertEqual(_first_name(""), _account_name() or "there")

    def test_blank_account_name_falls_back_to_there(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"USERNAME": "", "USER": ""}, clear=False):
            self.assertEqual(_first_name(""), "there")

    def test_whitespace_only_config_is_not_a_name(self):
        self.assertEqual(_first_name("   "), _account_name() or "there")


@unittest.skipUnless(QT_AVAILABLE, "PySide6 not installed")
class FirstRunPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.json"

    def _answer(self, typed: str | None) -> Config:
        """Drive the dialog without a human: typed=None means Skip."""
        config = Config()
        dialog = WelcomeDialog(suggestion="Accountname")
        if typed is None:
            dialog.reject()
        else:
            dialog._field.setText(typed)
            dialog._accept()
        config.ui.display_name = dialog.name
        config.onboarded = True
        config.save(self.path)
        return config

    def test_a_typed_name_is_stored(self):
        self._answer("Grace")
        self.assertEqual(Config.load(self.path).ui.display_name, "Grace")

    def test_a_typed_name_is_cleaned_on_the_way_in(self):
        self._answer("  grace   hopper  ")
        self.assertEqual(Config.load(self.path).ui.display_name, "grace hopper")

    def test_skipping_stores_no_name_but_still_onboards(self):
        self._answer(None)
        reloaded = Config.load(self.path)
        self.assertEqual(reloaded.ui.display_name, "")
        self.assertTrue(reloaded.onboarded,
                        "a skipped prompt must not reappear on the next launch")

    def test_a_fresh_config_asks(self):
        self.assertFalse(Config().onboarded)

    def test_ask_for_name_persists_and_onboards(self):
        # The real entry point, with exec() stubbed to stand in for the user
        # typing a name, and CONFIG_PATH redirected away from the real one.
        from unittest import mock

        import openflow.config as config_module

        def types_a_name(dialog):
            dialog._field.setText("Grace")
            dialog._accept()

        config = Config()
        with mock.patch.object(WelcomeDialog, "exec", types_a_name), \
                mock.patch.object(config_module, "CONFIG_PATH", self.path):
            returned = ask_for_name(config, suggestion="Accountname")

        self.assertEqual(returned, "Grace")
        self.assertEqual(config.ui.display_name, "Grace")
        self.assertTrue(config.onboarded)
        self.assertEqual(Config.load(self.path).ui.display_name, "Grace",
                         "the answer has to reach disk, not just memory")

    def test_the_suggestion_is_prefilled_and_selected(self):
        dialog = WelcomeDialog(suggestion="Accountname")
        self.assertEqual(dialog._field.text(), "Accountname")
        self.assertEqual(dialog._field.selectedText(), "Accountname",
                         "typing should replace the guess, not append to it")


if __name__ == "__main__":
    unittest.main(verbosity=2)
