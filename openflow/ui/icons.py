"""Line icons drawn with QPainter, for when Segoe Fluent Icons is missing.

The window names its icons by Segoe Fluent / MDL2 codepoint. Windows ships
that font; macOS does not, and there every glyph fell back to the same box of
lines. These stand in for the handful the window uses, drawn on a 24-unit
grid in the same outline style.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPen


def _mic(p: QPainterPath) -> None:
    p.addRoundedRect(QRectF(9, 2.5, 6, 12), 3, 3)
    p.moveTo(5, 11)
    p.arcTo(QRectF(5, 4, 14, 14), 180, 180)
    p.moveTo(12, 18)
    p.lineTo(12, 21.5)


def _chart(p: QPainterPath) -> None:
    p.moveTo(3.5, 3.5)
    p.lineTo(3.5, 20.5)
    p.lineTo(20.5, 20.5)
    for x, top in ((8.5, 12), (13, 7), (17.5, 10.5)):
        p.moveTo(x, 16.5)
        p.lineTo(x, top)


def _book(p: QPainterPath) -> None:
    for side in (1, -1):
        def pt(x, y, s=side):
            return QPointF(12 + s * (x - 12), y)
        p.moveTo(pt(12, 7.5))
        p.quadTo(pt(12, 4.5), pt(8.5, 4.5))
        p.lineTo(pt(2.5, 4.5))
        p.lineTo(pt(2.5, 17.5))
        p.lineTo(pt(9, 17.5))
        p.quadTo(pt(12, 17.5), pt(12, 20.5))
    p.moveTo(12, 7.5)
    p.lineTo(12, 20.5)


def _scissors(p: QPainterPath) -> None:
    p.addEllipse(QPointF(6, 6), 3, 3)
    p.addEllipse(QPointF(6, 18), 3, 3)
    p.moveTo(20, 4)
    p.lineTo(8.1, 15.9)
    p.moveTo(14.5, 14.5)
    p.lineTo(20, 20)
    p.moveTo(8.1, 8.1)
    p.lineTo(12, 12)


def _letter(p: QPainterPath) -> None:
    p.moveTo(4.5, 20)
    p.lineTo(12, 3.5)
    p.lineTo(19.5, 20)
    p.moveTo(7.6, 13.5)
    p.lineTo(16.4, 13.5)


def _sync(p: QPainterPath) -> None:
    p.moveTo(3, 12)
    p.arcTo(QRectF(3, 3, 18, 18), 180, -90)
    p.cubicTo(14.5, 3, 16.8, 3.9, 18.7, 5.7)
    p.lineTo(21, 8)
    p.moveTo(21, 3)
    p.lineTo(21, 8)
    p.lineTo(16, 8)
    p.moveTo(21, 12)
    p.arcTo(QRectF(3, 3, 18, 18), 0, -90)
    p.cubicTo(9.5, 21, 7.2, 20.1, 5.3, 18.3)
    p.lineTo(3, 16)
    p.moveTo(8, 16)
    p.lineTo(3, 16)
    p.lineTo(3, 21)


def _note(p: QPainterPath) -> None:
    p.addRoundedRect(QRectF(4, 2.5, 16, 19), 2.5, 2.5)
    for y, end in ((8, 16), (12, 16), (16, 13)):
        p.moveTo(8, y)
        p.lineTo(end, y)


def _gear(p: QPainterPath) -> None:
    p.addEllipse(QPointF(12, 12), 3, 3)
    p.addEllipse(QPointF(12, 12), 7, 7)
    for i in range(8):
        a = math.radians(i * 45)
        p.moveTo(12 + 7 * math.cos(a), 12 + 7 * math.sin(a))
        p.lineTo(12 + 9.5 * math.cos(a), 12 + 9.5 * math.sin(a))


def _copy(p: QPainterPath) -> None:
    p.addRoundedRect(QRectF(8, 8, 13.5, 13.5), 2, 2)
    p.moveTo(4.5, 16)
    p.quadTo(2.5, 16, 2.5, 14)
    p.lineTo(2.5, 4.5)
    p.quadTo(2.5, 2.5, 4.5, 2.5)
    p.lineTo(14, 2.5)
    p.quadTo(16, 2.5, 16, 4.5)


# Segoe Fluent codepoint -> drawing.
DRAWINGS = {
    "": _mic,        # Microphone
    "": _chart,      # AreaChart
    "": _book,       # Dictionary
    "": _scissors,   # Cut
    "": _letter,     # Font
    "": _sync,       # Sync
    "": _note,       # QuickNote
    "": _gear,       # Setting
    "": _copy,       # Copy
}


def draw(painter: QPainter, glyph: str, px: int) -> bool:
    """Draw ``glyph`` into the ``px`` square at the origin with the painter's
    current pen colour. False if there is no drawing for it."""
    build = DRAWINGS.get(glyph)
    if build is None:
        return False
    path = QPainterPath()
    build(path)
    painter.save()
    painter.scale(px / 24, px / 24)
    pen = QPen(painter.pen().color(), 1.9)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(path)
    painter.restore()
    return True
