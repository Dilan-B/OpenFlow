"""Read the text around the caret in whatever app has focus.

Wispr Flow formats a dictation against where it lands: mid-sentence text is
lowercased, spaces are added only where missing, a trailing period is dropped
when punctuation already follows. All of that needs the characters on either
side of the insertion point, which only the platform accessibility layer can
provide -- UI Automation's TextPattern on Windows, the AX API on macOS.

Every failure here returns None and the formatter falls back to formatting as
if the text stood alone, which is what OpenFlow did before this existed. Many
apps expose no text pattern at all; that is normal, not an error.

Privacy: this reads at most a few hundred characters around the caret, only at
the moment the dictation hotkey goes down, never from password fields, and
nothing read is stored or sent anywhere -- it is used for one formatting
decision and dropped.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

from ..formatting import CaretContext

log = logging.getLogger(__name__)

BEFORE_CHARS = 200
AFTER_CHARS = 40
# A selection is the thing Command Mode edits, so it gets a real budget --
# enough for a long paragraph, short of a whole document.
SELECTION_CHARS = 8000
# UI Automation calls cross into the target process. A hung app must not hang
# dictation, so every call is bounded, and the whole read runs off the hotkey
# thread (see CaretReader).
UIA_TIMEOUT_MS = 250


def _clean(text: str | None) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def read_caret_context() -> CaretContext | None:
    try:
        if sys.platform == "win32":
            return _read_windows()
        if sys.platform == "darwin":
            return _read_macos()
    except Exception as exc:  # accessibility APIs fail in endless app-specific ways
        log.debug("caret context unavailable: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Windows: UI Automation TextPattern
# ---------------------------------------------------------------------------
_UIA_TEXT_PATTERN_ID = 10014
_ENDPOINT_START, _ENDPOINT_END = 0, 1
_UNIT_CHARACTER = 0

_uia_lock = threading.Lock()


def _uia_module():
    """Generate (once) and import the UIAutomationClient wrapper.

    comtypes builds it from the system type library on first use and caches
    it; in a frozen build the cache lives under %APPDATA%, not inside the app.
    """
    import comtypes.client

    with _uia_lock:
        comtypes.client.GetModule("UIAutomationCore.dll")
        from comtypes.gen import UIAutomationClient

    return UIAutomationClient


def _read_windows() -> CaretContext | None:
    import comtypes
    import comtypes.client

    comtypes.CoInitialize()
    try:
        uia = _uia_module()
        automation = comtypes.client.CreateObject(
            uia.CUIAutomation, interface=uia.IUIAutomation)
        try:
            bounded = automation.QueryInterface(uia.IUIAutomation2)
            bounded.ConnectionTimeout = UIA_TIMEOUT_MS
            bounded.TransactionTimeout = UIA_TIMEOUT_MS
        except Exception:
            pass  # pre-Windows 10 UIA: no timeouts, still worth trying

        element = automation.GetFocusedElement()
        if element is None or element.CurrentIsPassword:
            return None
        pattern = element.GetCurrentPattern(_UIA_TEXT_PATTERN_ID)
        if pattern is None:
            return None
        text = pattern.QueryInterface(uia.IUIAutomationTextPattern)
        selection = text.GetSelection()
        if selection is None or selection.Length < 1:
            return None
        caret = selection.GetElement(0)
        has_selection = caret.CompareEndpoints(
            _ENDPOINT_START, caret, _ENDPOINT_END) != 0

        before = caret.Clone()
        before.MoveEndpointByRange(_ENDPOINT_END, caret, _ENDPOINT_START)
        before.MoveEndpointByUnit(_ENDPOINT_START, _UNIT_CHARACTER, -BEFORE_CHARS)
        after = caret.Clone()
        after.MoveEndpointByRange(_ENDPOINT_START, caret, _ENDPOINT_END)
        after.MoveEndpointByUnit(_ENDPOINT_END, _UNIT_CHARACTER, AFTER_CHARS)

        return CaretContext(
            before=_clean(before.GetText(BEFORE_CHARS)),
            after=_clean(after.GetText(AFTER_CHARS)),
            has_selection=has_selection,
            selected=_clean(caret.GetText(SELECTION_CHARS)) if has_selection else "",
        )
    finally:
        comtypes.CoUninitialize()


# ---------------------------------------------------------------------------
# macOS: Accessibility (AX) API
# ---------------------------------------------------------------------------
def _read_macos() -> CaretContext | None:
    from ApplicationServices import (  # type: ignore[import-not-found]
        AXUIElementCopyAttributeValue,
        AXUIElementCreateSystemWide,
        AXValueGetValue,
        kAXFocusedUIElementAttribute,
        kAXSelectedTextRangeAttribute,
        kAXSubroleAttribute,
        kAXValueAttribute,
        kAXValueCFRangeType,
    )

    system = AXUIElementCreateSystemWide()
    err, element = AXUIElementCopyAttributeValue(system, kAXFocusedUIElementAttribute, None)
    if err or element is None:
        return None
    err, subrole = AXUIElementCopyAttributeValue(element, kAXSubroleAttribute, None)
    if not err and subrole == "AXSecureTextField":
        return None
    err, value = AXUIElementCopyAttributeValue(element, kAXValueAttribute, None)
    if err or not isinstance(value, str):
        return None
    err, range_value = AXUIElementCopyAttributeValue(
        element, kAXSelectedTextRangeAttribute, None)
    if err or range_value is None:
        return None
    ok, selected = AXValueGetValue(range_value, kAXValueCFRangeType, None)
    if not ok:
        return None
    start, length = int(selected.location), int(selected.length)
    return CaretContext(
        before=value[max(0, start - BEFORE_CHARS):start],
        after=value[start + length:start + length + AFTER_CHARS],
        has_selection=length > 0,
        selected=value[start:start + min(length, SELECTION_CHARS)],
    )


# ---------------------------------------------------------------------------
# Off-thread reader
# ---------------------------------------------------------------------------
def copy_selection(timeout_s: float = 0.35) -> str:
    """Read the selection by copying it, for apps accessibility cannot see.

    Many Electron apps (Slack, Discord, Notion) expose no text pattern at all,
    and Command Mode is useless without the words it is meant to edit. So this
    is the fallback: send the copy shortcut, read the clipboard, then put back
    whatever was on it.

    Only ever called for Command Mode, never for ordinary dictation -- taking
    over the clipboard is a reasonable cost for an explicit "edit this", and an
    unreasonable one for every sentence a user speaks.
    """
    try:
        import pyperclip
        from pynput.keyboard import Controller, Key
    except ImportError:
        return ""

    keyboard = Controller()
    modifier = Key.cmd if sys.platform == "darwin" else Key.ctrl
    try:
        previous = pyperclip.paste()
    except Exception:
        previous = ""
    sentinel = "\x00openflow-no-selection\x00"
    try:
        pyperclip.copy(sentinel)
        with keyboard.pressed(modifier):
            keyboard.press("c")
            keyboard.release("c")
        deadline = time.monotonic() + timeout_s
        text = sentinel
        while time.monotonic() < deadline:
            time.sleep(0.02)
            text = pyperclip.paste()
            if text != sentinel:
                break
        return "" if text == sentinel else text
    except Exception as exc:
        log.debug("could not copy the selection: %s", exc)
        return ""
    finally:
        try:
            pyperclip.copy(previous)
        except Exception:
            log.debug("could not restore the clipboard")


class CaretReader:
    """Read the caret context in the background as a dictation starts.

    Started when the hotkey goes down -- the only moment the target app is
    guaranteed to still have focus -- and collected once transcription is
    done, which always takes far longer than the read.
    """

    def __init__(self) -> None:
        self._result: CaretContext | None = None
        self._done = threading.Event()

    def start(self) -> "CaretReader":
        threading.Thread(target=self._run, name="openflow-caret", daemon=True).start()
        return self

    def _run(self) -> None:
        try:
            self._result = read_caret_context()
        finally:
            self._done.set()

    def result(self, wait_s: float = 0.3) -> CaretContext | None:
        self._done.wait(wait_s)
        return self._result if self._done.is_set() else None
