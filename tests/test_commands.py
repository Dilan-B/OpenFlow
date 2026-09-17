"""Command Mode: parsing spoken instructions, and the hotkey that starts them.

The rules that decide whether your words leave the machine live in
openflow/commands.py, so they are pinned hard here: nothing is a search unless
the speaker named an engine, and nothing is an edit unless they said the wake
phrase.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.commands import SEARCH_URLS, parse  # noqa: E402
from openflow.config import HotkeyConfig  # noqa: E402
from openflow.input.hotkeys import COMMAND_TAKEOVER_S, HotkeyListener  # noqa: E402


class EditCommands(unittest.TestCase):
    def test_wake_phrase_variants(self):
        for spoken in ("hey flow make this shorter", "Hey Flow, make this shorter",
                       "hi flow make this shorter", "ok flow make this shorter",
                       "flow, make this shorter"):
            command = parse(spoken)
            self.assertEqual(command.kind, "edit", spoken)
            self.assertEqual(command.instruction, "make this shorter")

    def test_without_the_wake_phrase_it_is_not_a_command(self):
        for spoken in ("make this shorter", "can we meet on Friday",
                       "rewrite this as a list"):
            self.assertEqual(parse(spoken).kind, "none", spoken)

    def test_empty_speech(self):
        self.assertEqual(parse("").kind, "none")
        self.assertEqual(parse("hey flow").kind, "none")


class SearchCommands(unittest.TestCase):
    def test_documented_engines(self):
        for spoken, engine in (
            ("search google for flight times", "google"),
            ("ask perplexity about tide tables", "perplexity"),
            ("hey chatgpt what is the capital of Peru", "chatgpt"),
            ("ask claude to explain recursion", "claude"),
        ):
            command = parse(spoken)
            self.assertEqual(command.kind, "search", spoken)
            self.assertEqual(command.engine, engine)
            self.assertTrue(command.url.startswith(SEARCH_URLS[engine]))

    def test_question_words_stay_in_the_query(self):
        self.assertEqual(parse("hey chatgpt what is the capital of Peru").query,
                         "what is the capital of Peru")

    def test_selection_is_appended_to_the_query(self):
        command = parse("search google for", selection="best pizza in Chicago")
        self.assertEqual(command.query, "best pizza in Chicago")

    def test_a_search_inside_the_wake_phrase_still_searches(self):
        self.assertEqual(parse("hey flow ask claude about this").kind, "search")

    def test_an_unnamed_engine_is_never_a_search(self):
        """Nothing leaves the machine unless the speaker named where to send it."""
        for spoken in ("search for flight times", "look up the tide tables",
                       "google-ish thing", "ask someone about this"):
            self.assertNotEqual(parse(spoken).kind, "search", spoken)

    def test_the_query_cannot_choose_the_destination(self):
        command = parse("search google for evil.example.com/steal")
        self.assertTrue(command.url.startswith("https://www.google.com/search?q="))
        self.assertNotIn(" ", command.url)


class ClipboardFallback(unittest.TestCase):
    """Reading a selection by copying it, for apps accessibility cannot see.

    The user's clipboard is borrowed, so the contract is: return the selection,
    and put back exactly what was there -- including when nothing was selected,
    and when the copy raises.
    """

    def _fake_clipboard(self, copied: str | None):
        """A clipboard whose copy shortcut yields ``copied`` (None = nothing
        selected, so the app leaves the clipboard untouched)."""
        from unittest import mock

        store = {"value": "PRIOR"}
        clipboard = mock.Mock()
        clipboard.paste.side_effect = lambda: store["value"]
        clipboard.copy.side_effect = lambda text: store.update(value=text)

        keyboard = mock.MagicMock()

        def press(_key):
            if copied is not None:
                store["value"] = copied

        keyboard.press.side_effect = press
        return clipboard, keyboard, store

    def _run(self, clipboard, keyboard):
        from unittest import mock

        from openflow.input import caret

        modules = {"pyperclip": clipboard,
                   "pynput.keyboard": mock.Mock(Controller=lambda: keyboard,
                                                Key=mock.Mock())}
        with mock.patch.dict(sys.modules, modules):
            return caret.copy_selection(timeout_s=0.1)

    def test_returns_the_selection_and_restores_the_clipboard(self):
        clipboard, keyboard, store = self._fake_clipboard("the selected words")
        self.assertEqual(self._run(clipboard, keyboard), "the selected words")
        self.assertEqual(store["value"], "PRIOR")

    def test_nothing_selected_returns_empty_and_restores(self):
        clipboard, keyboard, store = self._fake_clipboard(None)
        self.assertEqual(self._run(clipboard, keyboard), "")
        self.assertEqual(store["value"], "PRIOR")

    def test_a_failure_still_restores_the_clipboard(self):
        clipboard, keyboard, store = self._fake_clipboard("ignored")
        keyboard.pressed.side_effect = RuntimeError("no input permission")
        self.assertEqual(self._run(clipboard, keyboard), "")
        self.assertEqual(store["value"], "PRIOR")


class _FakeKey:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.name}>"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, _FakeKey) and other.name == self.name


CTRL, WIN, ALT = _FakeKey("ctrl"), _FakeKey("cmd"), _FakeKey("alt")


class _Listener(HotkeyListener):
    """A listener wired to fake keys, so the combo logic is testable without
    pynput's platform hooks."""

    def __init__(self, **kwargs):
        super().__init__(HotkeyConfig(), **kwargs)
        self._combo = frozenset({CTRL, WIN})
        self._command_combo = frozenset({CTRL, WIN, ALT})

    def _canonical(self, key):
        return key


class CommandHotkey(unittest.TestCase):
    """The command combo contains the dictation combo, so pressing it always
    starts a dictation first. It has to hand over cleanly."""

    def setUp(self) -> None:
        self.events: list[str] = []
        self.listener = _Listener(
            on_start=lambda: self.events.append("start"),
            on_stop=lambda: self.events.append("stop"),
            on_cancel=lambda: self.events.append("cancel"),
            on_command_start=lambda: self.events.append("command-start"),
            on_command_stop=lambda: self.events.append("command-stop"),
        )

    def test_the_third_key_takes_the_dictation_over(self):
        for key in (CTRL, WIN, ALT):
            self.listener._on_press(key)
        self.assertEqual(self.events, ["start", "cancel", "command-start"])

    def test_releasing_runs_the_command_not_a_dictation(self):
        for key in (CTRL, WIN, ALT):
            self.listener._on_press(key)
        self.listener._on_release(ALT)
        self.listener._on_release(WIN)
        self.listener._on_release(CTRL)
        self.assertEqual(self.events,
                         ["start", "cancel", "command-start", "command-stop"])
        self.assertNotIn("stop", self.events)

    def test_a_dictation_already_under_way_is_not_stolen(self):
        """Past the takeover window the user is mid-sentence; Alt must not
        throw away what they have already said."""
        self.listener._on_press(CTRL)
        self.listener._on_press(WIN)
        self.listener._recording_since -= COMMAND_TAKEOVER_S + 1
        self.listener._on_press(ALT)
        self.assertEqual(self.events, ["start"])
        self.listener._on_release(WIN)
        self.assertEqual(self.events, ["start", "stop"])

    def test_plain_dictation_is_unaffected(self):
        self.listener._on_press(CTRL)
        self.listener._on_press(WIN)
        self.listener._on_release(WIN)
        self.assertEqual(self.events, ["start", "stop"])

    def test_escape_cancels_a_command(self):
        self.listener._cancel_key = _FakeKey("esc")
        for key in (CTRL, WIN, ALT):
            self.listener._on_press(key)
        self.listener._on_press(_FakeKey("esc"))
        self.assertEqual(self.events[-1], "cancel")
        self.listener._on_release(ALT)
        self.assertNotIn("command-stop", self.events)

    def test_command_mode_off_leaves_dictation_alone(self):
        listener = _Listener(
            on_start=lambda: self.events.append("start"),
            on_stop=lambda: self.events.append("stop"),
            on_cancel=lambda: self.events.append("cancel"),
        )
        listener._command_combo = frozenset()
        for key in (CTRL, WIN, ALT):
            listener._on_press(key)
        self.assertEqual(self.events, ["start"])


if __name__ == "__main__":
    unittest.main()
