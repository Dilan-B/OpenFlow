"""The OpenFlow application window -- Wispr-styled.

Warm cream canvas; the content lives on a white rounded panel to the right of
a transparent icon sidebar. Serif for editorial moments (banner headlines,
stat numbers), sans for chrome. Color is scarce: teal for data, purple for
selection, black for primary buttons.

All widget access happens on the Qt main thread.
"""

from __future__ import annotations

import io
import logging
import math
import os
import sys
import time

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QPushButton, QScrollArea, QSizePolicy,
    QStackedWidget, QTextEdit, QVBoxLayout, QWidget, QGridLayout,
)

from ..config import Config
from ..history import History
from ..personalization import STYLES, Personalization
from . import theme

log = logging.getLogger(__name__)

# dwmapi.h -- recolour the native caption so it matches the app canvas.
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36

# Segoe Fluent / MDL2 glyphs (shipped with Windows 10/11).
NAV = [
    ("dictation", "Dictation", ""),
    ("insights", "Insights", ""),
    ("dictionary", "Dictionary", ""),
    ("snippets", "Snippets", ""),
    ("style", "Style", ""),
    ("transforms", "Transforms", ""),
    ("scratchpad", "Scratchpad", ""),
]

STYLE_PREVIEWS = {
    "default": ("Your words, untouched",
                "Can we push the demo to Friday? I want the auth flow in."),
    "professional": ("Caps + punctuation",
                     "Could we move the demo to Friday? I would like the auth flow included."),
    "casual": ("Keeps your tone",
               "Hey, can we push the demo to Friday? Want the auth flow in."),
    "concise": ("Trimmed down",
                "Push demo to Friday — auth flow should be in."),
    "email": ("Paragraphs for email",
              "Can we push the demo to Friday?\n\nI want the auth flow included."),
}

TRANSFORMS = [
    ("formal", "Polish", "Improve clarity and register"),
    ("shorten", "Shorten", "Cut to the essentials"),
    ("bullets", "Bullets", "Restructure as a list"),
]


def _first_name() -> str:
    raw = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    return raw.split(".")[0].split("_")[0].capitalize() or "there"


def logo_pixmap(size: int = 26) -> QPixmap:
    from .icon import mic_image

    buffer = io.BytesIO()
    mic_image(size * 4, tile=False).save(buffer, format="PNG")
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue())
    return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)


def fluent_icon(glyph: str, color: str, px: int = 15) -> QIcon:
    """Render a Segoe Fluent Icons glyph into a QIcon."""
    pixmap = QPixmap(px * 2, px * 2)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont("Segoe Fluent Icons")
    if not font.exactMatch():
        font = QFont("Segoe MDL2 Assets")
    font.setPixelSize(px)
    painter.setFont(font)
    painter.setPen(QColor(color))
    painter.drawText(QRectF(0, 0, px, px), Qt.AlignmentFlag.AlignCenter, glyph)
    painter.end()
    return QIcon(pixmap)


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())


def _row(*widgets, stretch_index: int | None = None,
         margins=(0, 0, 0, 0)) -> QFrame:
    """A QWidget-backed row. Dynamic content must live in real widgets:
    children added through bare nested layouts after the parent is visible
    never get shown by Qt."""
    frame = QFrame()
    frame.setObjectName("Row")
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(*margins)
    for i, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if i == stretch_index else 0)
    if stretch_index is None:
        layout.addStretch()
    return frame


def _hairline() -> QFrame:
    line = QFrame()
    line.setObjectName("Hairline")
    line.setFixedHeight(1)
    return line


# ---------------------------------------------------------------------------
# Painted widgets
# ---------------------------------------------------------------------------

class BarsChart(QWidget):
    """Words per day, teal."""

    def __init__(self, history: History) -> None:
        super().__init__()
        self.history = history
        self.setMinimumHeight(140)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        days = self.history.stats.last_days(7)
        peak = max((w for _, w in days), default=0) or 1
        width, height = self.width(), self.height()
        slot = width / 7
        bar_w = min(24.0, slot * 0.3)
        base = height - 24.0
        painter.setPen(Qt.PenStyle.NoPen)
        for i, (day, words) in enumerate(days):
            bar = 4 + (base - 14) * words / peak
            x = i * slot + (slot - bar_w) / 2
            colour = theme.CHART_COLORS[i % len(theme.CHART_COLORS)]
            painter.setBrush(QColor(colour if words else theme.BORDER_SOFT))
            painter.drawRoundedRect(QRectF(x, base - bar, bar_w, bar), 4, 4)
            painter.setPen(QColor(theme.TEXT_FAINT))
            painter.drawText(QRectF(i * slot, base + 4, slot, 16),
                             Qt.AlignmentFlag.AlignCenter, day[5:].replace("-", "/"))
            painter.setPen(Qt.PenStyle.NoPen)
        painter.end()


class Gauge(QWidget):
    """Wispr's semicircular WPM gauge."""

    def __init__(self, history: History, ceiling: float = 220.0) -> None:
        super().__init__()
        self.history = history
        self.ceiling = ceiling
        self.setMinimumHeight(84)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        stroke = 10.0
        radius = min(width / 2 - stroke, height - stroke - 4)
        if radius <= 10:
            painter.end()
            return
        cx, cy = width / 2, height - 4
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)

        from PySide6.QtGui import QPen

        pen = QPen(QColor(theme.BORDER_SOFT), stroke, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, 0, 180 * 16)

        share = max(0.03, min(1.0, self.history.stats.speaking_wpm / self.ceiling))
        pen.setColor(QColor(theme.TEAL_A))
        painter.setPen(pen)
        painter.drawArc(rect, 180 * 16, -int(180 * 16 * share))

        # end dot
        angle = math.pi * (1 - share)
        dot_x = cx + radius * math.cos(angle)
        dot_y = cy - radius * math.sin(angle)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawEllipse(QRectF(dot_x - 6, dot_y - 6, 12, 12))
        painter.setBrush(QColor(theme.TEAL_A))
        painter.drawEllipse(QRectF(dot_x - 4, dot_y - 4, 8, 8))
        painter.end()


class Heatmap(QWidget):
    """Wispr's streak calendar: weeks x weekdays, teal ramp."""

    WEEKS = 18

    def __init__(self, history: History) -> None:
        super().__init__()
        self.history = history
        self.setMinimumHeight(150)

    def paintEvent(self, event) -> None:  # noqa: N802
        import datetime as dt

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        daily = self.history.stats.daily
        peak = max((int(v) for v in daily.values()), default=0) or 1

        left_gutter = 30.0
        top_gutter = 16.0
        cols, rows = self.WEEKS, 7
        cell = min((self.width() - left_gutter - 4) / cols,
                   (self.height() - top_gutter - 4) / rows)
        gap = max(2.0, cell * 0.18)
        size = cell - gap

        today = dt.date.today()
        # Column 0 is the oldest week; align so the last column ends today.
        start = today - dt.timedelta(days=today.weekday() + 1 + (cols - 1) * 7)
        # start on a Sunday
        painter.setPen(QColor(theme.TEXT_FAINT))
        small = painter.font()
        small.setPixelSize(10)
        painter.setFont(small)
        for r, label in enumerate(("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")):
            if r % 2 == 0:
                painter.drawText(
                    QRectF(0, top_gutter + r * cell, left_gutter - 6, cell),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)

        month_drawn: set[str] = set()
        painter.setPen(Qt.PenStyle.NoPen)
        for c in range(cols):
            for r in range(rows):
                day = start + dt.timedelta(days=c * 7 + r)
                if day > today:
                    continue
                words = int(daily.get(day.isoformat(), 0))
                if words == 0:
                    color = theme.BORDER_SOFT
                elif words < peak * 0.34:
                    color = "#F6D3C6"
                elif words < peak * 0.67:
                    color = "#EE9A78"
                else:
                    color = theme.CORAL
                painter.setBrush(QColor(color))
                painter.drawRoundedRect(
                    QRectF(left_gutter + c * cell, top_gutter + r * cell, size, size),
                    2.5, 2.5)
                month = day.strftime("%b")
                if day.day <= 7 and r == 0 and month not in month_drawn:
                    month_drawn.add(month)
                    painter.setPen(QColor(theme.TEXT_FAINT))
                    painter.drawText(
                        QRectF(left_gutter + c * cell, 0, cell * 5, 14),
                        Qt.AlignmentFlag.AlignLeft, month)
                    painter.setPen(Qt.PenStyle.NoPen)
        painter.end()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, config: Config, history: History, personal: Personalization,
                 *, callbacks: dict) -> None:
        super().__init__()
        self.config = config
        self.history = history
        self.personal = personal
        self.cb = callbacks
        self.state = "ready"
        self.scratch_mode = False
        self._w: dict = {}
        self._built_pages: set[str] = set()
        self._title_bar_blended = False

        self.setWindowTitle("OpenFlow")
        self.resize(1140, 740)
        self.setMinimumSize(980, 640)
        self.setStyleSheet(theme.QSS)
        try:
            from .icon import ensure_ico

            ico = ensure_ico()
            if ico:
                self.setWindowIcon(QIcon(str(ico)))
        except Exception:
            pass

        canvas = QWidget()
        canvas.setObjectName("Canvas")
        canvas_layout = QHBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 12, 12)
        canvas_layout.setSpacing(0)
        self.setCentralWidget(canvas)

        canvas_layout.addWidget(self._build_sidebar())

        panel = QFrame()
        panel.setObjectName("Panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(panel, 1)

        self.stack = QStackedWidget()
        self.stack.setObjectName("Panel")
        panel_layout.addWidget(self.stack)

        self._page_index: dict[str, int] = {}
        for key in [k for k, _l, _g in NAV] + ["settings"]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            holder = QWidget()
            holder.setObjectName("PageHolder")
            scroll.setWidget(holder)
            self._page_index[key] = self.stack.addWidget(scroll)
        self.show_page("dictation")

    # ---------------------------------------------------------------- sidebar
    def _build_sidebar(self) -> QWidget:
        rail = QFrame()
        rail.setObjectName("Sidebar")
        rail.setFixedWidth(208)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(14, 16, 10, 16)
        layout.setSpacing(2)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(6, 0, 0, 16)
        logo = QLabel()
        logo.setPixmap(logo_pixmap(22))
        logo.setStyleSheet("background: transparent;")
        brand_row.addWidget(logo)
        name = QLabel("OpenFlow")
        name.setObjectName("Brand")
        brand_row.addWidget(name)
        brand_row.addStretch()
        layout.addLayout(brand_row)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        for key, label, glyph in NAV:
            colour = theme.NAV_COLORS.get(key, theme.INDIGO)
            button = QPushButton(f"  {label}")
            button.setObjectName("Nav")
            button.setIcon(fluent_icon(glyph, colour))
            # Selected state wears the destination's colour, so the sidebar
            # reads as a legend for the rest of the window.
            button.setStyleSheet(
                f"QPushButton#Nav:checked {{ background: {theme.tint(colour)};"
                f" color: {colour}; font-weight: 600; }}")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self._nav_group.addButton(button)
            layout.addWidget(button)
            self._w[f"nav_{key}"] = button

        layout.addStretch()

        settings = QPushButton("  Settings")
        settings.setObjectName("NavSmall")
        settings.setIcon(fluent_icon("", theme.BLUE))
        settings.setStyleSheet(
            f"QPushButton#NavSmall:checked {{ background: {theme.tint(theme.BLUE)};"
            f" color: {theme.BLUE}; font-weight: 600; }}")
        settings.setCheckable(True)
        settings.setCursor(Qt.CursorShape.PointingHandCursor)
        settings.clicked.connect(lambda: self.show_page("settings"))
        self._nav_group.addButton(settings)
        layout.addWidget(settings)
        self._w["nav_settings"] = settings

        chip = QFrame()
        chip.setObjectName("StatusChip")
        chip_layout = QHBoxLayout(chip)
        chip_layout.setContentsMargins(12, 8, 12, 8)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {theme.GREEN}; background: transparent; font-size: 10px;")
        chip_layout.addWidget(dot)
        status = QLabel("Ready")
        status.setStyleSheet("background: transparent; font-weight: 600; font-size: 12px;")
        chip_layout.addWidget(status)
        chip_layout.addStretch()
        hotkey = QLabel(theme.pretty_hotkey(self.config.hotkey.trigger))
        hotkey.setObjectName("Faint")
        chip_layout.addWidget(hotkey)
        layout.addSpacing(8)
        layout.addWidget(chip)
        self._w["status_dot"], self._w["status_label"] = dot, status
        self._w["status_hotkey"] = hotkey
        return rail

    # ------------------------------------------------------------------ pages
    def show_page(self, key: str) -> None:
        if key not in self._built_pages:
            self._built_pages.add(key)
            holder = self.stack.widget(self._page_index[key]).widget()
            getattr(self, f"_page_{key}")(holder)
        self.stack.setCurrentIndex(self._page_index[key])
        self._w[f"nav_{key}"].setChecked(True)
        if key == "dictation":
            self.refresh()
        elif key == "insights":
            self._render_insights()

    def _page_layout(self, holder: QWidget) -> QVBoxLayout:
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(14)
        return layout

    def _banner(self, layout, headline: str, sub: str, button: str | None = None,
                on_click=None, variant: str = "Banner") -> None:
        banner = QFrame()
        banner.setObjectName(variant)
        inner = QVBoxLayout(banner)
        inner.setContentsMargins(26, 20, 26, 20)
        inner.setSpacing(4)
        head = QLabel(headline)
        head.setObjectName("BannerHead")
        inner.addWidget(head)
        subtitle = QLabel(sub)
        subtitle.setObjectName("BannerSub")
        subtitle.setWordWrap(True)
        inner.addWidget(subtitle)
        if button:
            row = QHBoxLayout()
            action = QPushButton(button)
            action.setObjectName("BannerBtn")
            if on_click:
                action.clicked.connect(on_click)
            row.addWidget(action)
            row.addStretch()
            inner.addSpacing(8)
            inner.addLayout(row)
        layout.addWidget(banner)

    def _card(self, layout, margins=(20, 16, 20, 16), stretch=0) -> QVBoxLayout:
        card = QFrame()
        card.setObjectName("Card")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(*margins)
        inner.setSpacing(8)
        layout.addWidget(card, stretch)
        return inner

    # -------------------------------------------------------------- dictation
    def _page_dictation(self, holder: QWidget) -> None:
        layout = self._page_layout(holder)
        title = QLabel(f"Welcome back, {_first_name()}")
        title.setObjectName("H1")
        layout.addWidget(title)

        columns = QHBoxLayout()
        columns.setSpacing(18)
        layout.addLayout(columns, 1)

        main = QVBoxLayout()
        main.setSpacing(12)
        columns.addLayout(main, 1)

        self._banner(
            main,
            "Your voice works anywhere you write",
            "Hold the shortcut, speak, release — clean text lands at your cursor.\n"
            "Teach it names in Dictionary; expand phrases with Snippets.",
            "Open scratchpad", lambda: self.show_page("scratchpad"),
        )

        history_head = QHBoxLayout()
        kicker = QLabel("TODAY")
        kicker.setObjectName("Kicker")
        history_head.addWidget(kicker)
        history_head.addStretch()
        clear = QPushButton("Clear")
        clear.setObjectName("Ghost")
        clear.clicked.connect(lambda: (self.cb["clear_history"](), self.refresh()))
        history_head.addWidget(clear)
        main.addSpacing(6)
        main.addLayout(history_head)
        self._w["history_kicker"] = kicker

        recent = QVBoxLayout()
        recent.setSpacing(0)
        main.addLayout(recent)
        main.addStretch()
        self._w["recent_layout"] = recent

        # Right rail: one card, Wispr-style.
        rail_card = QFrame()
        rail_card.setObjectName("Card")
        rail_card.setFixedWidth(232)
        rail = QVBoxLayout(rail_card)
        rail.setContentsMargins(20, 18, 20, 18)
        rail.setSpacing(10)
        for key, caption, colour in (("words", "total words", theme.INDIGO),
                                     ("wpm", "wpm", theme.TEAL_A),
                                     ("streak", "day streak", theme.CORAL)):
            row = QHBoxLayout()
            value = QLabel("0")
            value.setObjectName("StatNum")
            value.setStyleSheet(f"color: {colour}; background: transparent;")
            row.addWidget(value)
            cap = QLabel(caption)
            cap.setObjectName("Sub")
            row.addWidget(cap)
            row.addStretch()
            rail.addLayout(row)
            self._w[f"stat_{key}"] = value
        rail.addWidget(_hairline())
        engines_title = QLabel("Engines")
        engines_title.setObjectName("H2")
        rail.addWidget(engines_title)
        engine_rows = QVBoxLayout()
        engine_rows.setSpacing(3)
        rail.addLayout(engine_rows)
        rail.addStretch()
        self._w["engine_layout"] = engine_rows

        rail_holder = QVBoxLayout()
        rail_holder.addWidget(rail_card)
        rail_holder.addStretch()
        columns.addLayout(rail_holder)

    def _render_recent(self) -> None:
        recent = self._w.get("recent_layout")
        if recent is None:
            return
        _clear(recent)
        entries = self.history.recent(12)
        if not entries:
            empty = QLabel("Nothing yet — hold the shortcut and say something.")
            empty.setObjectName("Faint")
            recent.addWidget(empty)
            return

        today = time.strftime("%Y-%m-%d")
        yesterday = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
        current_day = None
        first_group = True
        for entry in entries:
            day = time.strftime("%Y-%m-%d", time.localtime(entry.at))
            if day != current_day:
                if day == today and first_group:
                    pass  # the TODAY kicker is already in the header row
                else:
                    label = ("YESTERDAY" if day == yesterday
                             else time.strftime("%B %d", time.localtime(entry.at)).upper())
                    kicker = QLabel(label)
                    kicker.setObjectName("Kicker")
                    kicker.setContentsMargins(0, 14, 0, 4)
                    recent.addWidget(kicker)
                current_day = day
                first_group = False
            elif recent.count():
                recent.addWidget(_hairline())

            row_frame = QFrame()
            row_frame.setObjectName("Row")
            row = QHBoxLayout(row_frame)
            row.setContentsMargins(6, 9, 6, 9)
            when = QLabel(entry.when)
            when.setObjectName("Faint")
            when.setFixedWidth(52)
            row.addWidget(when)
            body = entry.text or f"{entry.words} words dictated"
            if len(body) > 88:
                body = body[:85] + "…"
            text = QLabel(body)
            text.setStyleSheet("background: transparent;" if entry.text else
                               f"background: transparent; color: {theme.TEXT_DIM};")
            text.setMinimumWidth(0)
            text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            row.addWidget(text, 1)
            if entry.text:
                copy = QPushButton()
                copy.setObjectName("Ghost")
                copy.setIcon(fluent_icon("", theme.TEXT_FAINT, 13))
                copy.setFixedWidth(30)
                copy.setToolTip("Copy")
                copy.clicked.connect(lambda _=False, t=entry.text: self._copy(t))
                row.addWidget(copy)
            recent.addWidget(row_frame)

    # --------------------------------------------------------------- insights
    def _page_insights(self, holder: QWidget) -> None:
        layout = self._page_layout(holder)
        title = QLabel("Insights")
        title.setObjectName("H1")
        layout.addWidget(title)

        top = QHBoxLayout()
        top.setSpacing(14)
        layout.addLayout(top)

        # WPM gauge card.
        wpm_card = QFrame()
        wpm_card.setObjectName("Card")
        wpm_card.setStyleSheet(
            f"#Card {{ background: {theme.tint(theme.TEAL_A)};"
            f" border: 1px solid {theme.TEAL_A}22; border-radius: 12px; }}")
        wpm_inner = QVBoxLayout(wpm_card)
        wpm_inner.setContentsMargins(20, 16, 20, 12)
        wpm_num = QLabel("0")
        wpm_num.setObjectName("StatNum")
        wpm_num.setStyleSheet(f"color: {theme.TEAL_A}; background: transparent;")
        wpm_inner.addWidget(wpm_num)
        wpm_cap = QLabel("WORDS PER MINUTE")
        wpm_cap.setObjectName("StatCap")
        wpm_inner.addWidget(wpm_cap)
        gauge = Gauge(self.history)
        wpm_inner.addWidget(gauge)
        top.addWidget(wpm_card, 1)
        self._w["i_wpm"], self._w["gauge"] = wpm_num, gauge

        # Fixes card.
        fixes_card = QFrame()
        fixes_card.setObjectName("Card")
        fixes_card.setStyleSheet(
            f"#Card {{ background: {theme.tint(theme.VIOLET_A)};"
            f" border: 1px solid {theme.VIOLET_A}22; border-radius: 12px; }}")
        fixes_inner = QVBoxLayout(fixes_card)
        fixes_inner.setContentsMargins(20, 16, 20, 16)
        fixes_num = QLabel("0")
        fixes_num.setObjectName("StatNum")
        fixes_num.setStyleSheet(f"color: {theme.VIOLET_A}; background: transparent;")
        fixes_inner.addWidget(fixes_num)
        fixes_cap = QLabel("FIXES MADE BY OPENFLOW")
        fixes_cap.setObjectName("StatCap")
        fixes_inner.addWidget(fixes_cap)
        fixes_inner.addWidget(_hairline())
        fixes_rows = QVBoxLayout()
        fixes_rows.setSpacing(4)
        fixes_inner.addLayout(fixes_rows)
        fixes_inner.addStretch()
        top.addWidget(fixes_card, 1)
        self._w["i_fixes"], self._w["fixes_rows"] = fixes_num, fixes_rows

        # Total words card.
        words_card = QFrame()
        words_card.setObjectName("Card")
        words_card.setStyleSheet(
            f"#Card {{ background: {theme.tint(theme.INDIGO)};"
            f" border: 1px solid {theme.INDIGO}22; border-radius: 12px; }}")
        words_inner = QVBoxLayout(words_card)
        words_inner.setContentsMargins(20, 16, 20, 16)
        words_num = QLabel("0")
        words_num.setObjectName("StatNum")
        words_num.setStyleSheet(f"color: {theme.INDIGO}; background: transparent;")
        words_inner.addWidget(words_num)
        words_cap = QLabel("TOTAL WORDS DICTATED")
        words_cap.setObjectName("StatCap")
        words_inner.addWidget(words_cap)
        words_inner.addWidget(_hairline())
        saved = QLabel("")
        saved.setObjectName("Sub")
        words_inner.addWidget(saved)
        latency = QLabel("")
        latency.setObjectName("Sub")
        words_inner.addWidget(latency)
        words_inner.addStretch()
        top.addWidget(words_card, 1)
        self._w["i_words"], self._w["i_saved"], self._w["i_latency"] = (
            words_num, saved, latency)

        bottom = QHBoxLayout()
        bottom.setSpacing(14)
        layout.addLayout(bottom)

        chart_card = QFrame()
        chart_card.setObjectName("Card")
        chart_inner = QVBoxLayout(chart_card)
        chart_inner.setContentsMargins(20, 16, 20, 12)
        chart_title = QLabel("Words per day")
        chart_title.setObjectName("H2")
        chart_inner.addWidget(chart_title)
        bars = BarsChart(self.history)
        chart_inner.addWidget(bars)
        bottom.addWidget(chart_card, 1)
        self._w["chart"] = bars

        streak_card = QFrame()
        streak_card.setObjectName("Card")
        streak_inner = QVBoxLayout(streak_card)
        streak_inner.setContentsMargins(20, 16, 20, 12)
        streak_head = QHBoxLayout()
        streak_title = QLabel("0 day streak")
        streak_title.setObjectName("SerifH")
        streak_title.setStyleSheet(f"color: {theme.CORAL}; background: transparent;")
        streak_head.addWidget(streak_title)
        streak_head.addStretch()
        longest = QLabel("LONGEST STREAK | 0 DAYS")
        longest.setObjectName("StatCap")
        streak_head.addWidget(longest)
        streak_inner.addLayout(streak_head)
        heatmap = Heatmap(self.history)
        streak_inner.addWidget(heatmap)
        bottom.addWidget(streak_card, 1)
        self._w["streak_title"], self._w["longest"] = streak_title, longest
        self._w["heatmap"] = heatmap
        layout.addStretch()

    def _render_insights(self) -> None:
        stats = self.history.stats
        if "i_wpm" not in self._w:
            return
        self._w["i_wpm"].setText(f"{stats.speaking_wpm:.0f}")
        total_fixes = stats.retractions + stats.dict_fixes
        self._w["i_fixes"].setText(f"{total_fixes:,}")
        fixes_rows = self._w["fixes_rows"]
        _clear(fixes_rows)
        for caption, value in (("false starts removed", stats.retractions),
                               ("dictionary fixes", stats.dict_fixes),
                               ("dictations", stats.dictations)):
            val = QLabel(f"{value:,}")
            val.setStyleSheet("background: transparent; font-weight: 600;")
            cap = QLabel(caption)
            cap.setObjectName("Sub")
            row = _row(val, cap)
            fixes_rows.addWidget(row)
            row.show()
        self._w["i_words"].setText(f"{stats.words:,}")
        self._w["i_saved"].setText(f"≈ {stats.minutes_saved:.0f} minutes saved vs typing")
        self._w["i_latency"].setText(
            f"{stats.avg_latency_ms / 1000:.1f} s average latency"
            if stats.dictations else "")
        self._w["streak_title"].setText(f"{stats.streak_days} day streak")
        self._w["longest"].setText(f"LONGEST STREAK | {stats.longest_streak} DAYS")
        self._w["chart"].update()
        self._w["heatmap"].update()

    # ------------------------------------------------------------- dictionary
    def _page_dictionary(self, holder: QWidget) -> None:
        layout = self._page_layout(holder)
        head = QHBoxLayout()
        title = QLabel("Dictionary")
        title.setObjectName("H1")
        head.addWidget(title)
        head.addStretch()
        layout.addLayout(head)

        self._banner(
            layout, "OpenFlow spells the way you do.",
            "Add personal terms, company jargon, client names, or industry lingo — "
            "recognition is biased toward them and near-misses get repaired.",
            variant="BannerWarm"
        )

        add_card = self._card(layout, margins=(14, 12, 14, 12))
        add_row = QHBoxLayout()
        entry = QLineEdit()
        entry.setPlaceholderText("Groq, OpenFlow, kubectl")
        add_row.addWidget(entry, 1)
        add = QPushButton("Add new")
        add.setObjectName("Primary")

        def do_add() -> None:
            if self.personal.add_term(entry.text()):
                entry.clear()
                self._render_terms()

        add.clicked.connect(do_add)
        entry.returnPressed.connect(do_add)
        add_row.addWidget(add)
        add_card.addLayout(add_row)

        list_card = self._card(layout, margins=(6, 4, 6, 4))
        terms = QVBoxLayout()
        terms.setSpacing(0)
        list_card.addLayout(terms)
        self._w["dict_layout"] = terms
        layout.addStretch()
        self._render_terms()

    def _render_terms(self) -> None:
        terms = self._w.get("dict_layout")
        if terms is None:
            return
        _clear(terms)
        if not self.personal.dictionary:
            empty = QLabel("No terms yet.")
            empty.setObjectName("Faint")
            empty.setContentsMargins(12, 10, 12, 10)
            terms.addWidget(empty)
            return
        for i, term in enumerate(self.personal.dictionary):
            if i:
                terms.addWidget(_hairline())
            row_frame = QFrame()
            row_frame.setObjectName("Row")
            row = QHBoxLayout(row_frame)
            row.setContentsMargins(12, 9, 8, 9)
            label = QLabel(term)
            label.setStyleSheet("background: transparent; font-size: 13px;")
            row.addWidget(label)
            row.addStretch()
            remove = QPushButton("Remove")
            remove.setObjectName("Ghost")
            remove.clicked.connect(
                lambda _=False, t=term: (self.personal.remove_term(t),
                                         self._render_terms()))
            row.addWidget(remove)
            terms.addWidget(row_frame)

    # --------------------------------------------------------------- snippets
    def _page_snippets(self, holder: QWidget) -> None:
        layout = self._page_layout(holder)
        title = QLabel("Snippets")
        title.setObjectName("H1")
        layout.addWidget(title)

        self._banner(
            layout, "The stuff you shouldn't have to re-type.",
            "Save text you use often — an address, an intro, a prompt — then say "
            "the trigger to drop it in instantly.",
            variant="BannerTeal"
        )

        add_card = self._card(layout, margins=(14, 12, 14, 12))
        trigger = QLineEdit()
        trigger.setPlaceholderText("my email sig")
        add_card.addWidget(trigger)
        body = QTextEdit()
        body.setPlaceholderText("Best,\nDilan")
        body.setFixedHeight(78)
        add_card.addWidget(body)
        save_row = QHBoxLayout()
        save_row.addStretch()
        save = QPushButton("Save snippet")
        save.setObjectName("Primary")

        def do_save() -> None:
            if self.personal.add_snippet(trigger.text(), body.toPlainText()):
                trigger.clear()
                body.clear()
                self._render_snippets()

        save.clicked.connect(do_save)
        save_row.addWidget(save)
        add_card.addLayout(save_row)

        list_card = self._card(layout, margins=(6, 4, 6, 4))
        snippets = QVBoxLayout()
        snippets.setSpacing(0)
        list_card.addLayout(snippets)
        self._w["snip_layout"] = snippets
        layout.addStretch()
        self._render_snippets()

    def _render_snippets(self) -> None:
        snippets = self._w.get("snip_layout")
        if snippets is None:
            return
        _clear(snippets)
        if not self.personal.snippets:
            empty = QLabel("No snippets yet.")
            empty.setObjectName("Faint")
            empty.setContentsMargins(12, 10, 12, 10)
            snippets.addWidget(empty)
            return
        for i, snippet in enumerate(self.personal.snippets):
            if i:
                snippets.addWidget(_hairline())
            row_frame = QFrame()
            row_frame.setObjectName("Row")
            row = QHBoxLayout(row_frame)
            row.setContentsMargins(12, 9, 8, 9)
            preview = snippet.text.replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:57] + "…"
            label = QLabel(f"{snippet.trigger}  →  {preview}")
            label.setStyleSheet("background: transparent; font-size: 13px;")
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            row.addWidget(label, 1)
            remove = QPushButton("Remove")
            remove.setObjectName("Ghost")
            remove.clicked.connect(
                lambda _=False, t=snippet.trigger: (self.personal.remove_snippet(t),
                                                    self._render_snippets()))
            row.addWidget(remove)
            snippets.addWidget(row_frame)

    # ------------------------------------------------------------------ style
    def _page_style(self, holder: QWidget) -> None:
        layout = self._page_layout(holder)
        title = QLabel("Style")
        title.setObjectName("H1")
        layout.addWidget(title)
        sub = QLabel("How cleanup shapes your words. Applied by the AI engine when "
                     "one is available; the deterministic pass never rewrites.")
        sub.setObjectName("Sub")
        sub.setWordWrap(True)
        layout.addWidget(sub)

        grid = QGridLayout()
        grid.setSpacing(14)
        layout.addLayout(grid)
        self._w["style_cards"] = {}
        palette = (theme.PINK, theme.INDIGO, theme.AMBER_A, theme.TEAL_A, theme.BLUE)
        for i, name in enumerate(STYLES):
            caption, example = STYLE_PREVIEWS[name]
            colour = palette[i % len(palette)]
            card = QFrame()
            card.setObjectName("StyleCard")
            self._w.setdefault("style_colours", {})[name] = colour
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            inner = QVBoxLayout(card)
            inner.setContentsMargins(16, 14, 16, 14)
            inner.setSpacing(6)
            head = QLabel(name.capitalize())
            head.setStyleSheet(
                f"background: transparent; font-weight: 600; font-size: 14px;"
                f" color: {colour};")
            inner.addWidget(head)
            cap = QLabel(caption)
            cap.setObjectName("Faint")
            inner.addWidget(cap)
            bubble = QFrame()
            bubble.setObjectName("Bubble")
            bubble.setStyleSheet(
                f"#Bubble {{ background: {theme.tint(colour)}; border-radius: 10px; }}")
            bubble_layout = QVBoxLayout(bubble)
            bubble_layout.setContentsMargins(12, 10, 12, 10)
            text = QLabel(example)
            text.setWordWrap(True)
            text.setStyleSheet("background: transparent; font-size: 12px;")
            bubble_layout.addWidget(text)
            inner.addWidget(bubble)
            inner.addStretch()

            card.mousePressEvent = (
                lambda _event, n=name: self._pick_style(n))  # type: ignore[assignment]
            grid.addWidget(card, i // 3, i % 3)
            self._w["style_cards"][name] = card
        layout.addStretch()
        self._sync_style()

    def _pick_style(self, name: str) -> None:
        self.personal.set_style(name)
        self._sync_style()

    def _sync_style(self) -> None:
        colours = self._w.get("style_colours", {})
        for name, card in self._w.get("style_cards", {}).items():
            active = name == self.personal.style
            colour = colours.get(name, theme.INDIGO)
            border = f"2px solid {colour}" if active else f"1px solid {theme.BORDER}"
            card.setStyleSheet(
                f"#StyleCard {{ background: {theme.CARD}; border: {border};"
                f" border-radius: 12px; }}")

    # -------------------------------------------------------------- transforms
    def _page_transforms(self, holder: QWidget) -> None:
        layout = self._page_layout(holder)
        title = QLabel("Transforms")
        title.setObjectName("H1")
        layout.addWidget(title)

        self._banner(
            layout, "Transform works on your Scratchpad",
            "Apply a transform to rewrite, clean up, or restructure text after "
            "you dictate. Needs an AI engine — a Gemini key or local Ollama.",
            variant="Banner"
        )

        serif = QLabel("My Transforms")
        serif.setObjectName("SerifH")
        layout.addWidget(serif)

        row = QHBoxLayout()
        row.setSpacing(14)
        layout.addLayout(row)
        self._w["transform_buttons"] = []
        transform_colours = (theme.VIOLET_A, theme.CORAL, theme.GREEN_A)
        for i, (key, label, blurb) in enumerate(TRANSFORMS):
            colour = transform_colours[i % len(transform_colours)]
            card = QFrame()
            card.setObjectName("Card")
            card.setStyleSheet(
                f"#Card {{ background: {theme.tint(colour)};"
                f" border: 1px solid {colour}33; border-radius: 12px; }}")
            inner = QVBoxLayout(card)
            inner.setContentsMargins(16, 14, 16, 14)
            inner.setSpacing(4)
            head = QLabel(label)
            head.setStyleSheet(
                f"background: transparent; font-weight: 600; font-size: 14px;"
                f" color: {colour};")
            inner.addWidget(head)
            cap = QLabel(blurb)
            cap.setObjectName("Sub")
            inner.addWidget(cap)
            inner.addSpacing(6)
            run = QPushButton("Run")
            run.setFixedWidth(74)
            run.clicked.connect(lambda _=False, k=key: self.cb["transform"](k))
            inner.addWidget(run)
            row.addWidget(card, 1)
            self._w["transform_buttons"].append(run)

        status = QLabel("")
        status.setObjectName("Faint")
        layout.addWidget(status)
        self._w["transform_status"] = status
        layout.addStretch()

    def set_transform_status(self, message: str, busy: bool = False) -> None:
        status = self._w.get("transform_status")
        if status is not None:
            status.setText(message)
        for button in self._w.get("transform_buttons", []):
            button.setEnabled(not busy)

    # -------------------------------------------------------------- scratchpad
    def _page_scratchpad(self, holder: QWidget) -> None:
        layout = self._page_layout(holder)
        title = QLabel("Scratchpad")
        title.setObjectName("H1")
        layout.addWidget(title)

        self._banner(
            layout, "For quick thoughts you want to come back to",
            "Drop a to-do list, polish a message before you send it, brain-dump "
            "an idea. Turn on capture and dictations land here.",
            variant="BannerTeal"
        )

        bar = QHBoxLayout()
        toggle = QCheckBox("Dictate into the scratchpad")
        toggle.toggled.connect(lambda on: setattr(self, "scratch_mode", on))
        bar.addWidget(toggle)
        bar.addStretch()
        copy_all = QPushButton("Copy all")
        copy_all.clicked.connect(lambda: self._copy(self.scratch_text()))
        bar.addWidget(copy_all)
        clear = QPushButton("Clear")
        clear.setObjectName("Ghost")
        bar.addWidget(clear)
        layout.addLayout(bar)

        pad = QTextEdit()
        pad.setPlaceholderText("Hold the shortcut and start talking…")
        pad.setStyleSheet("QTextEdit { font-size: 15px; padding: 12px; }")
        layout.addWidget(pad, 1)
        clear.clicked.connect(pad.clear)
        self._w["scratch"] = pad

    def scratch_text(self) -> str:
        pad = self._w.get("scratch")
        return pad.toPlainText().strip() if pad else ""

    def append_scratch(self, text: str) -> None:
        if "scratch" not in self._w:
            self.show_page("scratchpad")
        pad = self._w["scratch"]
        existing = pad.toPlainText().strip()
        pad.insertPlainText((" " if existing else "") + text)
        pad.moveCursor(pad.textCursor().MoveOperation.End)

    def replace_scratch(self, text: str) -> None:
        pad = self._w.get("scratch")
        if pad is not None:
            pad.setPlainText(text)

    # ---------------------------------------------------------------- settings
    def _page_settings(self, holder: QWidget) -> None:
        layout = self._page_layout(holder)
        title = QLabel("Settings")
        title.setObjectName("H1")
        layout.addWidget(title)

        # Shortcuts row, Wispr-style: description + Change + mode segment.
        shortcut_card = self._card(layout, margins=(20, 14, 20, 14))
        shortcut_row = QHBoxLayout()
        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        head = QLabel("Shortcut")
        head.setStyleSheet("background: transparent; font-weight: 600;")
        text_col.addWidget(head)
        desc = QLabel(f"Hold {theme.pretty_hotkey(self.config.hotkey.trigger)} and speak.")
        desc.setObjectName("Faint")
        text_col.addWidget(desc)
        shortcut_row.addLayout(text_col, 1)
        self._w["hotkey_desc"] = desc

        seg_wrap = QFrame()
        seg_wrap.setObjectName("SegWrap")
        seg_layout = QHBoxLayout(seg_wrap)
        seg_layout.setContentsMargins(3, 3, 3, 3)
        seg_layout.setSpacing(0)
        self._seg_group = QButtonGroup(self)
        for value, label in (("push_to_talk", "Hold"), ("toggle", "Toggle")):
            seg = QPushButton(label)
            seg.setObjectName("Seg")
            seg.setCheckable(True)
            seg.setChecked(self.config.hotkey.mode == value)
            seg.clicked.connect(lambda _=False, v=value: self.cb["mode"](v))
            self._seg_group.addButton(seg)
            seg_layout.addWidget(seg)
        shortcut_row.addWidget(seg_wrap)

        rebind = QPushButton("Change")
        rebind.clicked.connect(self._begin_capture)
        shortcut_row.addWidget(rebind)
        self._w["rebind_button"] = rebind
        shortcut_card.addLayout(shortcut_row)

        card = self._card(layout, margins=(20, 14, 20, 14))
        try:
            from ..shortcuts import launch_at_login_enabled

            at_login = launch_at_login_enabled()
        except Exception:
            at_login = False
        self._switch(card, "Start with Windows",
                     "Launch OpenFlow automatically at sign-in.",
                     "launch_at_login", at_login)
        card.addWidget(_hairline())
        self._switch(card, "Keep running when window is closed",
                     "The shortcut keeps working from the tray.",
                     "close_to_tray", self.config.ui.close_to_tray)
        card.addWidget(_hairline())
        self._switch(card, "AI cleanup",
                     "Sends transcripts to Gemini or Ollama for a second pass. "
                     "Slower — the built-in pass is instant and usually matches it.",
                     "llm_enabled", self.config.llm.enabled)
        card.addWidget(_hairline())
        self._switch(card, "Mute other apps while dictating",
                     "Spotify, videos, and calls mute on press and come back on release.",
                     "duck_others", self.config.audio.duck_others)
        card.addWidget(_hairline())
        self._switch(card, "Save dictation text",
                     "Shows your words in History. Off keeps only counts.",
                     "log_transcripts", self.config.log_transcripts)
        card.addWidget(_hairline())

        method_row = QHBoxLayout()
        method_label = QLabel("Insert text by")
        method_row.addWidget(method_label)
        method_row.addStretch()
        method = QComboBox()
        method.addItems(["paste", "type"])
        method.setCurrentText(self.config.injection.method)
        method.currentTextChanged.connect(
            lambda v: self.cb["setting"]("injection.method", v))
        method_row.addWidget(method)
        card.addLayout(method_row)

        footer = QHBoxLayout()
        pause = QPushButton("Pause dictation")
        pause.clicked.connect(self.cb["pause"])
        footer.addWidget(pause)
        self._w["pause_button"] = pause
        open_config = QPushButton("Open config folder")
        open_config.clicked.connect(self._open_config_dir)
        footer.addWidget(open_config)
        footer.addStretch()
        quit_button = QPushButton("Quit OpenFlow")
        quit_button.setObjectName("Danger")
        quit_button.clicked.connect(self.cb["quit"])
        footer.addWidget(quit_button)
        layout.addSpacing(4)
        layout.addLayout(footer)
        layout.addStretch()

    def _switch(self, layout, title: str, subtitle: str, key: str, initial: bool) -> None:
        row = QHBoxLayout()
        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        label = QLabel(title)
        label.setStyleSheet("background: transparent; font-weight: 600;")
        text_col.addWidget(label)
        sub = QLabel(subtitle)
        sub.setObjectName("Faint")
        text_col.addWidget(sub)
        row.addLayout(text_col, 1)
        switch = QCheckBox()
        switch.setChecked(initial)
        switch.toggled.connect(lambda on, k=key: self.cb["setting"](k, on))
        row.addWidget(switch)
        layout.addLayout(row)

    # ---------------------------------------------------------------- actions
    def _begin_capture(self) -> None:
        self._w["rebind_button"].setText("Press keys…")
        self._w["rebind_button"].setEnabled(False)
        self.cb["rebind"]()

    def capture_finished(self, combo: str | None) -> None:
        button = self._w.get("rebind_button")
        if button is not None:
            button.setText("Change")
            button.setEnabled(True)
        self.set_hotkey(combo or self.config.hotkey.trigger)

    def _copy(self, text: str) -> None:
        from PySide6.QtGui import QGuiApplication as Gui

        Gui.clipboard().setText(text)

    def _open_config_dir(self) -> None:
        from ..config import CONFIG_DIR

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(CONFIG_DIR)  # noqa: S606

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._blend_title_bar()

    def _blend_title_bar(self) -> None:
        """Paint the native caption in the app's own colour.

        Windows draws the title bar itself, in its own grey, which reads as a
        separate strip bolted to the top of the window. Windows 11 exposes the
        caption colour through DWM, so the bar can match the canvas while
        keeping every native behaviour -- snap layouts, drag, the real
        minimise/maximise/close buttons. Going frameless would mean
        reimplementing all of that badly.
        """
        if sys.platform != "win32" or self._title_bar_blended:
            return
        try:
            import ctypes

            def colorref(value: str) -> ctypes.c_int:
                red, green, blue = (int(value[i:i + 2], 16) for i in (1, 3, 5))
                return ctypes.c_int((blue << 16) | (green << 8) | red)

            handle = int(self.winId())
            dwm = ctypes.windll.dwmapi
            for attribute, colour in ((DWMWA_CAPTION_COLOR, theme.BG),
                                      (DWMWA_TEXT_COLOR, theme.TEXT),
                                      (DWMWA_BORDER_COLOR, theme.BG)):
                value = colorref(colour)
                dwm.DwmSetWindowAttribute(
                    handle, attribute, ctypes.byref(value), ctypes.sizeof(value))
            self._title_bar_blended = True
        except Exception:
            # Older Windows ignores these attributes; a grey caption is a
            # cosmetic loss, never a reason to fail startup.
            log.debug("could not recolour the title bar", exc_info=True)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.config.ui.close_to_tray:
            event.ignore()
            self.cb["hide"]()
        else:
            self.cb["quit"]()
            event.accept()

    # ----------------------------------------------------------------- update
    def set_state(self, state: str) -> None:
        self.state = state
        dot = self._w.get("status_dot")
        if dot is not None:
            dot.setStyleSheet(
                f"color: {theme.STATE_COLOR.get(state, theme.TEXT_DIM)}; "
                f"background: transparent; font-size: 10px;")
        label = self._w.get("status_label")
        if label is not None:
            label.setText(theme.STATE_LABEL.get(state, state.title()))
        pause = self._w.get("pause_button")
        if pause is not None:
            pause.setText("Resume dictation" if state == "paused" else "Pause dictation")

    def set_hotkey(self, combo: str) -> None:
        pretty = theme.pretty_hotkey(combo)
        chip = self._w.get("status_hotkey")
        if chip is not None:
            chip.setText(pretty)
        desc = self._w.get("hotkey_desc")
        if desc is not None:
            desc.setText(f"Hold {pretty} and speak.")

    def set_engines(self, rows: list[tuple[str, str, bool, str]]) -> None:
        engine_layout = self._w.get("engine_layout")
        if engine_layout is None:
            return
        _clear(engine_layout)
        for _group, name, ready, _detail in rows:
            dot = QLabel("●")
            dot.setStyleSheet(
                f"color: {theme.GREEN if ready else '#D9D5CC'}; "
                f"background: transparent; font-size: 9px;")
            dot.setFixedWidth(14)
            label = QLabel(name)
            label.setStyleSheet(
                f"background: transparent; font-size: 12px; "
                f"color: {theme.TEXT if ready else theme.TEXT_FAINT};")
            widget = _row(dot, label)
            engine_layout.addWidget(widget)
            widget.show()

    def refresh(self) -> None:
        stats = self.history.stats
        words = stats.words
        values = {
            "words": f"{words / 1000:.1f}K" if words >= 10_000 else f"{words:,}",
            "wpm": f"{stats.speaking_wpm:.0f}",
            "streak": str(stats.streak_days),
        }
        for key, value in values.items():
            widget = self._w.get(f"stat_{key}")
            if widget is not None:
                widget.setText(value)
        self._render_recent()
        if self.stack.currentIndex() == self._page_index.get("insights"):
            self._render_insights()
