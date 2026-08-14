"""Global hotkey listener (PRD section 1).

Supports both trigger styles:

  push_to_talk -- record while the combo is held, transcribe on release
  toggle       -- press to start, press again to stop

pynput delivers key events on its own thread. Callbacks fire there, so they
must stay non-blocking; the app posts real work onto a worker thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from ..config import HotkeyConfig

log = logging.getLogger(__name__)


class HotkeyUnavailable(RuntimeError):
    pass


class HotkeyListener:
    def __init__(
        self,
        config: HotkeyConfig,
        *,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_cancel: Callable[[], None],
        on_undo: Callable[[], None] | None = None,
    ) -> None:
        self.cfg = config
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_cancel = on_cancel
        self.on_undo = on_undo

        self._listener = None
        self._active = False          # combo currently satisfied
        self._recording = False
        self._pressed: set = set()
        self._combo: frozenset = frozenset()
        self._undo_combo: frozenset = frozenset()
        self._undo_active = False
        self._cancel_key = None
        self._last_edge = 0.0
        self._lock = threading.Lock()
        self.paused = False
        self._capture_cb: Callable[[str], None] | None = None
        self._capture_keys: list = []

    # -- setup -------------------------------------------------------------
    def start(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise HotkeyUnavailable(f"install pynput: {exc}") from exc

        self._keyboard = keyboard
        self._combo = frozenset(
            keyboard.HotKey.parse(self._normalize(self.cfg.trigger))
        )
        if self.cfg.cancel:
            parsed = keyboard.HotKey.parse(self._normalize(self.cfg.cancel))
            self._cancel_key = parsed[0] if len(parsed) == 1 else None
        if self.cfg.undo:
            try:
                self._undo_combo = frozenset(
                    keyboard.HotKey.parse(self._normalize(self.cfg.undo))
                )
            except ValueError as exc:
                # A bad combo in the config must not take the hotkey down with
                # it -- dictation matters more than undo.
                log.warning("could not parse undo combo %r: %s", self.cfg.undo, exc)
                self._undo_combo = frozenset()

        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release, suppress=False
        )
        self._listener.daemon = True
        self._listener.start()
        log.info("hotkey listening: %s (%s)", self.cfg.trigger, self.cfg.mode)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def rebind(self, combo: str) -> None:
        """Swap the trigger combo without restarting the listener."""
        from pynput import keyboard

        parsed = frozenset(keyboard.HotKey.parse(self._normalize(combo)))
        with self._lock:
            self.cfg.trigger = combo
            self._combo = parsed
            self._pressed.clear()
            self._active = False
            self._recording = False
        log.info("hotkey rebound to %s", combo)

    def capture(self, callback: Callable[[str], None]) -> None:
        """Record the next chord the user presses and hand it back as a combo
        string. Dictation is suspended until the capture completes."""
        self._capture_keys = []
        self._capture_cb = callback

    def _describe(self, key) -> str | None:
        """pynput key -> the canonical spelling used in config."""
        from pynput import keyboard

        for name in ("ctrl", "alt", "shift", "cmd"):
            base = getattr(keyboard.Key, name)
            variants = {base}
            for side in ("_l", "_r"):
                sided = getattr(keyboard.Key, name + side, None)
                if sided is not None:
                    variants.add(sided)
            if key in variants:
                return f"<{name}>"
        if isinstance(key, keyboard.Key):
            return f"<{key.name}>"
        if getattr(key, "char", None):
            return key.char.lower()
        return None

    @staticmethod
    def _normalize(combo: str) -> str:
        """Accept friendly spellings as well as pynput's canonical form."""
        aliases = {
            "option": "<alt>", "alt": "<alt>", "ctrl": "<ctrl>", "control": "<ctrl>",
            "shift": "<shift>", "cmd": "<cmd>", "win": "<cmd>", "super": "<cmd>",
            "space": "<space>", "capslock": "<caps_lock>", "caps": "<caps_lock>",
            "esc": "<esc>", "escape": "<esc>",
        }
        parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
        return "+".join(aliases.get(p, p) for p in parts)

    # -- event handling ----------------------------------------------------
    def _canonical(self, key):
        return self._listener.canonical(key) if self._listener else key

    def _on_press(self, key) -> None:
        key = self._canonical(key)

        if self._capture_cb is not None:
            name = self._describe(key)
            if name and name not in self._capture_keys:
                self._capture_keys.append(name)
            return

        if self.paused:
            return

        if self._cancel_key is not None and key == self._cancel_key and self._recording:
            self._recording = self._active = False
            self._safely(self.on_cancel)
            return

        self._pressed.add(key)

        # Undo is checked before the dictation trigger and only when we are not
        # recording, so a combo that shares modifiers cannot steal a dictation.
        if (self.on_undo is not None and self._undo_combo
                and not self._recording
                and self._undo_combo.issubset(self._pressed)):
            if not self._undo_active and self._debounced():
                self._undo_active = True
                self._safely(self.on_undo)
            return

        if not self._combo.issubset(self._pressed) or self._active:
            return
        if not self._debounced():
            return

        self._active = True
        with self._lock:
            if self.cfg.mode == "toggle":
                self._recording = not self._recording
                self._safely(self.on_start if self._recording else self.on_stop)
            elif not self._recording:
                self._recording = True
                self._safely(self.on_start)

    def _on_release(self, key) -> None:
        key = self._canonical(key)

        if self._capture_cb is not None:
            # The chord is complete when the user lets go of the first key.
            callback, keys = self._capture_cb, list(self._capture_keys)
            self._capture_cb, self._capture_keys = None, []
            if keys:
                # Modifiers first, so "<ctrl>+<cmd>" not "<cmd>+<ctrl>".
                order = {"<ctrl>": 0, "<alt>": 1, "<shift>": 2, "<cmd>": 3}
                keys.sort(key=lambda k: order.get(k, 9))
                self._safely(lambda: callback("+".join(keys)))
            return

        self._pressed.discard(key)
        if self._undo_combo and not self._undo_combo.issubset(self._pressed):
            self._undo_active = False
        if not self._combo.issubset(self._pressed):
            self._active = False
            if self.cfg.mode == "push_to_talk" and self._recording:
                self._recording = False
                self._safely(self.on_stop)

    def _debounced(self) -> bool:
        now = time.monotonic()
        if (now - self._last_edge) * 1000 < self.cfg.debounce_ms:
            return False
        self._last_edge = now
        return True

    @staticmethod
    def _safely(callback: Callable[[], None]) -> None:
        # An exception here would kill the pynput thread and silently disable
        # the hotkey for the rest of the session.
        try:
            callback()
        except Exception:
            log.exception("hotkey callback failed")
