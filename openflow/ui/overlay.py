"""The recording pill (PRD section 1) -- Qt edition.

A 124x32 frameless capsule with a live waveform. Three fixes over the old
Tk pill:

* QPainter antialiasing -- bars are smooth rounded strokes, not the jagged
  circle-capped lines Tk canvas produced.
* Fluid motion -- each bar eases toward its target height every frame
  (``h += (target - h) * 0.35``) instead of snapping, and targets come from a
  short level history so the wave travels.
* Multi-monitor -- shown bottom-center of whichever screen currently contains
  the mouse cursor, not always the primary display.

Never steals focus: Tool window, WA_ShowWithoutActivating.
"""

from __future__ import annotations

import logging
import math
import time

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QPainter
from PySide6.QtWidgets import QWidget

from ..config import UiConfig
from . import theme

log = logging.getLogger(__name__)

BARS = 8
BAR_WIDTH = 2.4
BAR_GAP = 3.0
EASE_UP = 0.6      # jump fast when the level rises...
EASE_DOWN = 0.28   # ...settle slowly when it falls


class Overlay(QWidget):
    def __init__(self, config: UiConfig) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.cfg = config
        self.state = "idle"
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(config.overlay_width, config.overlay_height)
        self.setWindowOpacity(config.opacity)

        self._targets = [0.0] * BARS
        self._heights = [0.0] * BARS
        self._level_history = [0.0] * BARS

    # -- state -------------------------------------------------------------
    def show_pill(self, state: str = "recording") -> None:
        self.state = state
        self._place_on_active_screen()
        if not self.isVisible():
            self.show()

    def hide_pill(self) -> None:
        self.hide()
        self.state = "idle"
        self._targets = [0.0] * BARS
        self._heights = [0.0] * BARS
        self._level_history = [0.0] * BARS

    def set_state(self, state: str, detail: str = "") -> None:
        self.state = state

    def _place_on_active_screen(self) -> None:
        """Bottom-center of the screen the cursor is on -- dictation happens
        where you're working, which is where the mouse is."""
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + geo.height() - self.height() - self.cfg.bottom_margin
        self.move(x, y)

    # -- animation ---------------------------------------------------------
    def push_level(self, level: float) -> None:
        self._level_history.pop(0)
        self._level_history.append(max(0.0, min(1.0, level)))

    def tick(self) -> None:
        """Advance the animation one frame. Called at ~60 Hz by the app."""
        if not self.isVisible():
            return

        if self.state == "recording":
            # The level history *is* the waveform: newest sample in the
            # middle, older samples flowing outward -- reads as a ripple.
            mid = BARS // 2
            for i in range(BARS):
                age = abs(i - mid)
                sample = self._level_history[-1 - age * 2] if age * 2 < BARS else 0.0
                envelope = 0.45 + 0.55 * math.cos(age / mid * math.pi / 2)
                # Square-root response lifts quiet speech into the visible
                # range; the floor keeps silence as short stubs, never dots.
                loud = min(1.0, math.sqrt(max(0.0, sample)) * 1.35)
                self._targets[i] = max(0.24, loud) * envelope
        else:
            # Processing: a soft traveling wave so the pill stays alive.
            phase = time.monotonic() * 5.0
            for i in range(BARS):
                self._targets[i] = 0.18 + 0.30 * (0.5 + 0.5 * math.sin(phase + i * 0.8))

        for i in range(BARS):
            delta = self._targets[i] - self._heights[i]
            self._heights[i] += delta * (EASE_UP if delta > 0 else EASE_DOWN)
        self.update()

    # -- painting ----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        radius = rect.height() / 2
        painter.setPen(QColor(theme.PILL_BORDER))
        painter.setBrush(QColor(theme.PILL_BG))
        painter.drawRoundedRect(rect, radius, radius)

        color = QColor(theme.PILL_BAR_COLOR.get(self.state, theme.PILL_BAR))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)

        # Bars occupy the middle band only -- they should never crowd the
        # capsule's rounded top and bottom.
        span = self.height() * 0.5
        total = BARS * BAR_WIDTH + (BARS - 1) * BAR_GAP
        left = (self.width() - total) / 2
        mid_y = self.height() / 2
        for i in range(BARS):
            bar = max(3.5, self._heights[i] * span)
            x = left + i * (BAR_WIDTH + BAR_GAP)
            painter.drawRoundedRect(
                QRectF(x, mid_y - bar / 2, BAR_WIDTH, bar),
                BAR_WIDTH / 2, BAR_WIDTH / 2,
            )
        painter.end()
