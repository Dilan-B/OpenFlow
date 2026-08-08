"""System tray icon -- QSystemTrayIcon edition (pystray retired)."""

from __future__ import annotations

import io
import logging
from collections.abc import Callable

from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from . import theme

log = logging.getLogger(__name__)


def _icon(color: str | None = None) -> QIcon:
    from .icon import mic_image

    buffer = io.BytesIO()
    mic_image(64, color=color, tile=True).save(buffer, format="PNG")
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue())
    return QIcon(pixmap)


class Tray:
    def __init__(
        self,
        *,
        on_show: Callable[[], None],
        on_toggle_pause: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        self.on_show = on_show
        self.on_toggle_pause = on_toggle_pause
        self.on_quit = on_quit
        self._tray: QSystemTrayIcon | None = None
        self._pause_action: QAction | None = None
        self._paused = False

    @staticmethod
    def available() -> bool:
        return QSystemTrayIcon.isSystemTrayAvailable()

    def start(self) -> None:
        if not self.available():
            log.info("system tray unavailable")
            return
        self._tray = QSystemTrayIcon(_icon())
        self._tray.setToolTip("OpenFlow — ready")

        menu = QMenu()
        open_action = QAction("Open OpenFlow", menu)
        open_action.triggered.connect(self.on_show)
        menu.addAction(open_action)
        self._pause_action = QAction("Pause dictation", menu)
        self._pause_action.triggered.connect(self.on_toggle_pause)
        menu.addAction(self._pause_action)
        menu.addSeparator()
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self.on_quit)
        menu.addAction(quit_action)
        self._menu = menu   # keep a reference; the tray does not own it
        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_activated)
        self._tray.show()

    def _on_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.on_show()

    def set_state(self, state: str, paused: bool = False) -> None:
        self._paused = paused
        if self._tray is None:
            return
        self._tray.setIcon(_icon(theme.STATE_COLOR.get(state)))
        self._tray.setToolTip(f"OpenFlow — {theme.STATE_LABEL.get(state, state)}")
        if self._pause_action is not None:
            self._pause_action.setText(
                "Resume dictation" if paused else "Pause dictation")

    def stop(self) -> None:
        if self._tray is not None:
            self._tray.hide()
            self._tray = None
