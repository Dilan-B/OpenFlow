"""Keystroke injection into whatever app had focus (PRD section 1).

Two modes:

  paste -- put the text on the clipboard, send Ctrl/Cmd+V, restore the previous
           clipboard. Constant time regardless of length; the default.
  type  -- synthesize each character. Slower, but works in apps that block
           clipboard paste (some terminals, some remote-desktop clients).

On Windows we also capture the foreground window before the overlay appears and
restore it before injecting, because even a non-activating overlay can shuffle
focus on some window managers.
"""

from __future__ import annotations

import logging
import sys
import time

from ..config import InjectionConfig

log = logging.getLogger(__name__)
IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"


class Injector:
    def __init__(self, config: InjectionConfig) -> None:
        self.cfg = config
        self._controller = None
        self._saved_window = None
        # What we last put into another app, so undo knows how much to remove.
        self._last_injected = ""

    def _keyboard(self):
        if self._controller is None:
            from pynput.keyboard import Controller

            self._controller = Controller()
        return self._controller

    # -- focus -------------------------------------------------------------
    def remember_focus(self) -> None:
        """Snapshot the currently focused window before the overlay shows."""
        if not IS_WINDOWS:
            return
        try:
            import ctypes

            self._saved_window = ctypes.windll.user32.GetForegroundWindow()
        except Exception:  # pragma: no cover - platform specific
            self._saved_window = None

    def restore_focus(self) -> None:
        if not IS_WINDOWS or not self._saved_window:
            return
        try:
            import ctypes

            ctypes.windll.user32.SetForegroundWindow(self._saved_window)
            time.sleep(self.cfg.focus_restore_delay_s)
        except Exception:  # pragma: no cover - platform specific
            log.debug("could not restore foreground window")

    # -- injection ---------------------------------------------------------
    def inject(self, text: str) -> None:
        if not text:
            return
        self.restore_focus()
        self._last_injected = text
        if self.cfg.method == "paste":
            try:
                self._paste(text)
                return
            except Exception as exc:
                log.warning("paste injection failed (%s); typing instead", exc)
        self._type(text)

    def replace_last(self, text: str) -> bool:
        """Swap the previous insertion for ``text``.

        Deletes exactly as many characters as we inserted and types the
        replacement. This is only correct while the caret is still where we
        left it, which is why the caller gates it on a short time window --
        if the user has typed, clicked, or switched apps since, the backspaces
        would eat their work instead of ours.
        """
        previous = getattr(self, "_last_injected", "")
        if not previous:
            return False

        from pynput.keyboard import Key

        self.restore_focus()
        keyboard = self._keyboard()
        for _ in range(len(previous)):
            keyboard.press(Key.backspace)
            keyboard.release(Key.backspace)
        self._last_injected = text
        if text:
            if self.cfg.method == "paste":
                try:
                    self._paste(text)
                    return True
                except Exception as exc:
                    log.warning("paste failed during undo (%s); typing instead", exc)
            self._type(text)
        return True

    def _paste(self, text: str) -> None:
        import pyperclip
        from pynput.keyboard import Key

        previous = None
        if self.cfg.restore_clipboard:
            try:
                previous = pyperclip.paste()
            except Exception:
                previous = None

        pyperclip.copy(text)
        keyboard = self._keyboard()
        modifier = Key.cmd if IS_MAC else Key.ctrl
        with keyboard.pressed(modifier):
            keyboard.press("v")
            keyboard.release("v")

        if previous is not None:
            # Let the target app read the clipboard before we put it back.
            time.sleep(0.12)
            try:
                pyperclip.copy(previous)
            except Exception:
                pass

    def _type(self, text: str) -> None:
        keyboard = self._keyboard()
        interval = self.cfg.type_interval_s
        for char in text:
            keyboard.type(char)
            if interval:
                time.sleep(interval)
