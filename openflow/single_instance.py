"""Single-instance guard.

Two copies of a dictation app is not a cosmetic problem: each registers its own
global hotkey listener, so one keypress starts two recordings, two
transcriptions race, and both try to paste into the same cursor. There are also
two tray icons and two writers on the same history file.

Rather than refuse the second launch silently -- which looks broken when you
click a Desktop shortcut and nothing happens -- the second process hands its
"activate" request to the first and exits. Clicking the shortcut therefore
surfaces the running app, which is what every other Windows application does.

Implemented over a named local socket (a named pipe on Windows) because it
carries that message; a mutex could only say "someone else is here".
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

log = logging.getLogger(__name__)

SERVER_NAME = "OpenFlow.SingleInstance.v1"
CONNECT_TIMEOUT_MS = 300


class SingleInstance(QObject):
    """Owns the lock, and signals when another launch asks to be surfaced."""

    activate_requested = Signal()

    def __init__(self, name: str = SERVER_NAME) -> None:
        super().__init__()
        self._name = name
        self._server: QLocalServer | None = None

    def acquire(self) -> bool:
        """True if this process is the primary instance.

        False means a live instance answered and has been asked to show itself;
        the caller should exit without touching the microphone or the hotkey.
        """
        if self._ping_existing():
            return False

        # Nothing answered. A name can still linger from a process that was
        # killed rather than closed, so clear it before claiming it -- this is
        # safe precisely because the ping above proved no one is listening.
        QLocalServer.removeServer(self._name)

        self._server = QLocalServer(self)
        self._server.newConnection.connect(self._on_connection)
        if not self._server.listen(self._name):
            # Losing a startup race is the realistic cause. Carry on as a
            # normal instance rather than refusing to run at all.
            log.warning("could not claim the single-instance lock: %s",
                        self._server.errorString())
            self._server = None
        return True

    def _ping_existing(self) -> bool:
        socket = QLocalSocket()
        socket.connectToServer(self._name)
        if not socket.waitForConnected(CONNECT_TIMEOUT_MS):
            return False
        socket.write(b"activate")
        socket.flush()
        socket.waitForBytesWritten(CONNECT_TIMEOUT_MS)
        socket.disconnectFromServer()
        log.info("another instance is already running; asked it to show")
        return True

    def _on_connection(self) -> None:
        connection = self._server.nextPendingConnection() if self._server else None
        if connection is None:
            return
        connection.disconnected.connect(connection.deleteLater)
        self.activate_requested.emit()

    def release(self) -> None:
        if self._server is not None:
            self._server.close()
            QLocalServer.removeServer(self._name)
            self._server = None
