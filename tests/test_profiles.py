"""Tests for per-app dictation profiles.

The failure mode to guard against is a profile firing in the wrong place: a
missing full stop in a Word document is a typo the user has to fix by hand,
every single time, and they will not know why it is happening.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.profiles import (  # noqa: E402
    DEFAULT, PROFILES, apply_profile, profile_for,
)


class Matching(unittest.TestCase):
    def test_terminals_get_the_shell_profile(self):
        for exe in ("WindowsTerminal.exe", "cmd.exe", "pwsh.exe"):
            self.assertEqual(profile_for(exe).name, "shell", exe)

    def test_editors_get_the_code_profile(self):
        for exe in ("Code.exe", "idea64.exe", "sublime_text.exe"):
            self.assertEqual(profile_for(exe).name, "code", exe)

    def test_chat_apps(self):
        self.assertEqual(profile_for("slack.exe").name, "chat")

    def test_matching_is_case_insensitive(self):
        self.assertEqual(profile_for("SLACK.EXE").name, "chat")

    def test_unknown_apps_get_prose(self):
        self.assertIs(profile_for("winword.exe"), DEFAULT)
        self.assertIs(profile_for("some-random-app.exe"), DEFAULT)

    def test_no_foreground_process_is_not_an_error(self):
        self.assertIs(profile_for(""), DEFAULT)

    def test_matching_is_exact_not_substring(self):
        # "vscode.exe" contains "code.exe"; substring matching would claim it,
        # and would also claim anything else with those letters in the name.
        self.assertIs(profile_for("notvscode.exe"), DEFAULT)

    def test_user_overrides_win(self):
        self.assertEqual(
            profile_for("winword.exe", {"winword.exe": "chat"}).name, "chat")

    def test_an_unknown_override_target_falls_back_to_prose(self):
        self.assertIs(profile_for("winword.exe", {"winword.exe": "nope"}), DEFAULT)


class Shaping(unittest.TestCase):
    def shape(self, profile_name: str, text: str) -> str:
        return apply_profile(text, PROFILES[profile_name])

    def test_shell_drops_the_full_stop_and_the_capital(self):
        self.assertEqual(self.shape("shell", "Git status."), "git status")

    def test_code_uses_straight_quotes(self):
        self.assertEqual(self.shape("code", 'It returns “ok”.'), 'it returns "ok"')

    def test_code_straightens_dashes_and_ellipses(self):
        self.assertEqual(self.shape("code", "a — b …"), "a -- b ...")

    def test_chat_keeps_the_capital_but_drops_the_stop(self):
        self.assertEqual(self.shape("chat", "Sounds good."), "Sounds good")

    def test_question_marks_survive(self):
        # A question mark is a choice the speaker made, not stray punctuation.
        self.assertEqual(self.shape("chat", "Are we shipping?"), "Are we shipping?")

    def test_acronyms_keep_their_capitals(self):
        self.assertEqual(self.shape("shell", "API is down."), "API is down")

    def test_a_lone_i_is_not_lowercased(self):
        self.assertEqual(self.shape("shell", "I ran it."), "I ran it")

    def test_ellipsis_is_not_mistaken_for_a_full_stop(self):
        self.assertEqual(self.shape("chat", "well..."), "well...")

    def test_prose_changes_nothing(self):
        for text in ("Sounds good.", 'She said “yes”.', "Are we shipping?"):
            self.assertEqual(apply_profile(text, DEFAULT), text)

    def test_empty_text(self):
        self.assertEqual(self.shape("shell", ""), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
