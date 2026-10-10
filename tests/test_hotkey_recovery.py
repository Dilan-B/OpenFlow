"""The hotkey must come back on its own when it goes deaf.

macOS switches off an event tap whose callback is slow, and pynput's thread
can die; either way the app used to keep running with no hotkey until it was
restarted by hand.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.config import HotkeyConfig  # noqa: E402
from openflow.input import hotkeys  # noqa: E402
from openflow.input.hotkeys import HotkeyListener  # noqa: E402


class _FakeListener:
    def __init__(self, running: bool, thread_alive: bool | None = None) -> None:
        self.running = running
        self._thread_alive = running if thread_alive is None else thread_alive

    def is_alive(self) -> bool:
        return self._thread_alive


def _listener() -> HotkeyListener:
    return HotkeyListener(HotkeyConfig(), on_start=lambda: None,
                          on_stop=lambda: None, on_cancel=lambda: None)


class Heal(unittest.TestCase):
    def test_restarts_a_dead_listener(self):
        h = _listener()
        h._listener = _FakeListener(running=False)
        h._pressed.add("stale")
        h._active = True
        with mock.patch.object(h, "start") as start, mock.patch.object(h, "stop") as stop:
            self.assertTrue(h.heal())
        stop.assert_called_once()
        start.assert_called_once()
        self.assertEqual(h._pressed, set())
        self.assertFalse(h._active)

    def test_restarts_when_the_tap_was_never_created(self):
        # pynput leaves ``running`` True when macOS refuses the event tap; only
        # the thread having exited gives it away.
        h = _listener()
        h._listener = _FakeListener(running=True, thread_alive=False)
        self.assertFalse(h.alive)
        with mock.patch.object(h, "start") as start, mock.patch.object(h, "stop"):
            self.assertTrue(h.heal())
        start.assert_called_once()

    def test_leaves_a_healthy_listener_alone(self):
        h = _listener()
        h._listener = _FakeListener(running=True)
        with mock.patch.object(h, "start") as start:
            self.assertFalse(h.heal())
        start.assert_not_called()

    def test_nothing_to_heal_before_start(self):
        self.assertFalse(_listener().heal())


@unittest.skipUnless(sys.platform == "darwin", "macOS event tap")
class DisabledTap(unittest.TestCase):
    def _make(self):
        from pynput import keyboard
        return hotkeys._make_listener(keyboard, on_press=lambda k: None,
                                      on_release=lambda k: None)

    def test_reenables_the_tap_when_macos_turns_it_off(self):
        import Quartz
        listener = self._make()
        listener._openflow_tap = object()
        for kind in (hotkeys._TAP_DISABLED_BY_TIMEOUT,
                     hotkeys._TAP_DISABLED_BY_USER_INPUT):
            with mock.patch.object(Quartz, "CGEventTapEnable") as enable:
                listener._handler(None, kind, None, None)
            enable.assert_called_once_with(listener._openflow_tap, True)

    def test_heal_reenables_a_disabled_tap(self):
        import Quartz
        h = _listener()
        h._listener = _FakeListener(running=True)
        h._listener._openflow_tap = object()
        with mock.patch.object(Quartz, "CGEventTapIsEnabled", return_value=False), \
                mock.patch.object(Quartz, "CGEventTapEnable") as enable:
            self.assertTrue(h.heal())
        enable.assert_called_once_with(h._listener._openflow_tap, True)


if __name__ == "__main__":
    unittest.main()
