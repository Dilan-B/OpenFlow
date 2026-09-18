"""Keep pynput's keyboard-layout lookups on the main thread (macOS only).

pynput reads the current keyboard layout through the Carbon Text Input Sources
API (``TISCopyCurrentKeyboardInputSource`` / ``TISGetInputSourceProperty``)
from whichever thread builds a ``Controller`` or runs a ``Listener`` -- for us,
the worker thread and pynput's own listener thread. Recent macOS asserts that
API runs on the main dispatch queue and kills the process (SIGTRAP in
``dispatch_assert_queue``) otherwise, which crashed OpenFlow on the first
dictation.

The layout context pynput wants is plain data -- a keyboard type and the
``uchr`` layout bytes -- so we read it once on the main thread and hand every
later caller that snapshot instead of letting it touch Carbon.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading

log = logging.getLogger(__name__)

_installed = False


def install() -> None:
    """Snapshot the layout now and route pynput's lookups to the snapshot.

    Must be called on the main thread, before any pynput ``Controller`` or
    ``Listener`` is created. A no-op everywhere but macOS.
    """
    global _installed
    if sys.platform != "darwin" or _installed:
        return
    if threading.current_thread() is not threading.main_thread():
        log.warning("keyboard layout snapshot skipped: not on the main thread")
        return
    try:
        from pynput._util import darwin as util
        from pynput.keyboard import _darwin as keyboard
    except Exception as exc:  # pragma: no cover - pynput missing or changed
        log.warning("could not patch pynput keyboard layout lookup: %s", exc)
        return

    with util.keycode_context() as snapshot:
        context = snapshot

    @contextlib.contextmanager
    def cached_keycode_context():
        yield context

    # get_unicode_to_keycode_map (Controller) looks the name up in _util.darwin;
    # Listener._run uses the copy imported into keyboard._darwin. Patch both.
    util.keycode_context = cached_keycode_context
    keyboard.keycode_context = cached_keycode_context
    _installed = True
    log.debug("keyboard layout cached on the main thread")
