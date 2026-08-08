"""Mute other apps while you dictate; restore them when you let go.

Uses the same per-application WASAPI sessions the Windows volume mixer shows,
via pycaw. On duck we record each session's mute state and mute it; on restore
we unmute exactly the sessions *we* muted -- an app you had muted yourself
stays muted.

Both calls happen on the pynput hotkey thread, which keeps every COM object in
the apartment it was created in. Failures are logged and swallowed: audio
ducking must never break a dictation.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

log = logging.getLogger(__name__)


class AudioDucker:
    def __init__(self) -> None:
        self._muted_sessions: list = []
        self._lock = threading.Lock()
        self._com_ready = threading.local()

    @staticmethod
    def available() -> bool:
        if sys.platform != "win32":
            return False
        try:
            import pycaw.pycaw  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_com(self) -> None:
        if getattr(self._com_ready, "done", False):
            return
        import comtypes

        comtypes.CoInitialize()
        self._com_ready.done = True

    def duck(self) -> None:
        """Mute every other app that could be playing audio."""
        if not self.available():
            return
        try:
            self._ensure_com()
            from pycaw.pycaw import AudioUtilities

            me = os.getpid()
            muted = []
            for session in AudioUtilities.GetAllSessions():
                try:
                    if session.Process is not None and session.Process.pid == me:
                        continue
                    volume = session.SimpleAudioVolume
                    if volume is None or bool(volume.GetMute()):
                        continue  # already muted by the user -- leave it alone
                    volume.SetMute(1, None)
                    muted.append(session)
                except Exception:
                    continue
            with self._lock:
                # A cancel/stop race could leave a previous set behind; fold
                # it in so restore() always unmutes everything we ever muted.
                self._muted_sessions.extend(muted)
            if muted:
                log.debug("ducked %d audio sessions", len(muted))
        except Exception as exc:
            log.warning("audio duck failed: %s", exc)

    def restore(self) -> None:
        """Unmute exactly what duck() muted."""
        with self._lock:
            sessions, self._muted_sessions = self._muted_sessions, []
        if not sessions:
            return
        try:
            self._ensure_com()
        except Exception:
            pass
        restored = 0
        for session in sessions:
            try:
                session.SimpleAudioVolume.SetMute(0, None)
                restored += 1
            except Exception:
                continue
        log.debug("restored %d audio sessions", restored)
