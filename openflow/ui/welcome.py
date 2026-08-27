"""First-run welcome: ask what to call the person using the app.

The dictation page greets you by name. Without this it fell back to the
Windows account name, which is frequently not a name at all -- "Admin",
"dilan.b", the shop's asset tag. Ask once, store the answer, never ask again.

Skipping is a first-class outcome: the greeting falls back exactly as before,
and ``config.onboarded`` still flips so nobody gets nagged at every launch.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
)

from . import theme

MAX_NAME = 40


class WelcomeDialog(QDialog):
    """Modal, one time. ``name`` holds the answer once exec() returns."""

    def __init__(self, suggestion: str = "", parent=None) -> None:
        super().__init__(parent)
        self.name = ""

        self.setWindowTitle("Welcome to OpenFlow")
        self.setModal(True)
        self.setMinimumWidth(420)
        # QSS paints QMainWindow and #Canvas; a bare QDialog matches neither
        # and would come up in the default system grey.
        self.setObjectName("Canvas")
        self.setStyleSheet(theme.QSS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 22)
        layout.setSpacing(10)

        title = QLabel("Welcome to OpenFlow")
        title.setObjectName("H1")
        layout.addWidget(title)

        blurb = QLabel("What should we call you? This only shows up on your "
                       "own dictation page.")
        blurb.setObjectName("Faint")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)
        layout.addSpacing(6)

        self._field = QLineEdit()
        self._field.setPlaceholderText("Your first name")
        self._field.setMaxLength(MAX_NAME)
        self._field.setText(suggestion)
        self._field.selectAll()
        # Enter is the fast path out of a one-field dialog.
        self._field.returnPressed.connect(self._accept)
        layout.addWidget(self._field)
        layout.addSpacing(8)

        buttons = QHBoxLayout()
        skip = QPushButton("Skip")
        skip.setCursor(Qt.CursorShape.PointingHandCursor)
        skip.clicked.connect(self.reject)
        buttons.addWidget(skip)
        buttons.addStretch()

        start = QPushButton("Get started")
        start.setObjectName("Primary")
        start.setCursor(Qt.CursorShape.PointingHandCursor)
        start.setDefault(True)
        start.clicked.connect(self._accept)
        buttons.addWidget(start)
        layout.addLayout(buttons)

    def _accept(self) -> None:
        self.name = clean_name(self._field.text())
        self.accept()


def clean_name(raw: str) -> str:
    """Trim to something safe to render in a greeting.

    Collapses whitespace and drops control characters -- the value is written
    into a QLabel and a JSON config, so a newline or a stray tab would only
    ever make a mess.
    """
    collapsed = " ".join(str(raw or "").split())
    printable = "".join(ch for ch in collapsed if ch.isprintable())
    return printable[:MAX_NAME].strip()


def ask_for_name(config, suggestion: str = "", parent=None) -> str:
    """Run the dialog and persist the outcome. Returns the stored name.

    Marks the config onboarded either way, so a skip is remembered.
    """
    dialog = WelcomeDialog(suggestion, parent)
    dialog.exec()
    config.ui.display_name = dialog.name
    config.onboarded = True
    config.save()
    return dialog.name
