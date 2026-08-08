"""OpenFlow's visual language -- Wispr-derived warm light theme.

Warm cream canvas, white rounded content panel, hairline borders, serif
editorial moments, black primary buttons.

Colour is used generously but never arbitrarily: each destination and each
metric owns a hue and keeps it wherever it appears, so the palette reads as
a system instead of confetti. Text and chrome stay neutral; colour carries
identity (which section), quantity (charts, stat numbers) and state.
"""

from __future__ import annotations

# Canvas + surfaces
BG = "#F4F2ED"            # warm cream window canvas (sidebar sits on this)
PANEL = "#FFFFFF"         # the big rounded content panel
CARD = "#FFFFFF"
CARD_TINT = "#F7F6F2"     # nested / hover
BORDER = "#E8E5DE"
BORDER_SOFT = "#EFEDE7"
NAV_SELECTED = "#E7E4DD"

# Text
TEXT = "#191817"
TEXT_DIM = "#6F6C64"
TEXT_FAINT = "#A09C92"

# The one dark: primary buttons and banner surfaces.
INK = "#1A1A18"
INK_HOVER = "#33312D"
BANNER_A = "#3B372F"      # banner gradient, warm dark
BANNER_B = "#262420"

# Data + selection
TEAL = "#0F6E56"
TEAL_MID = "#1D9E75"
TEAL_SOFT = "#9FE1CB"
TEAL_FAINT = "#E1F5EE"
VIOLET = "#7F77DD"
VIOLET_SOFT = "#CECBF6"

GREEN = "#1D9E75"
AMBER = "#BA7517"
RED = "#C24141"

# ---------------------------------------------------------------------------
# The accent set. Colour carries meaning here -- each destination and each
# metric keeps its own hue everywhere it appears, so the sidebar, the stat
# numbers and the charts all agree on what "Insights" or "streak" looks like.
# ---------------------------------------------------------------------------
INDIGO = "#5B5BD6"
VIOLET_A = "#8B5CF6"
PINK = "#DB2777"
CORAL = "#E0623D"
AMBER_A = "#E09B2D"
GREEN_A = "#4C9A2A"
TEAL_A = "#0F9E82"
BLUE = "#2E7DD1"

# Matching pale washes for card fills and selected rows.
TINTS = {
    INDIGO: "#EDEDFB", VIOLET_A: "#F2ECFD", PINK: "#FCEAF3",
    CORAL: "#FCEDE8", AMBER_A: "#FDF3E2", GREEN_A: "#EDF5E5",
    TEAL_A: "#E4F5F1", BLUE: "#E8F1FB",
}

# One hue per destination.
NAV_COLORS = {
    "dictation": INDIGO,
    "insights": TEAL_A,
    "dictionary": CORAL,
    "snippets": AMBER_A,
    "style": PINK,
    "transforms": VIOLET_A,
    "scratchpad": GREEN_A,
    "settings": BLUE,
}

# Chart series, in order.
CHART_COLORS = (INDIGO, BLUE, TEAL_A, GREEN_A, AMBER_A, CORAL, PINK)


def tint(color: str) -> str:
    return TINTS.get(color, CARD_TINT)

# The pill floats over arbitrary apps; it stays dark.
PILL_BG = "#1C1E24"
PILL_BORDER = "#2A2D36"
PILL_BAR = "#F2F3F5"

STATE_COLOR = {
    "idle": TEXT_FAINT,
    "ready": GREEN,
    "recording": TEAL_MID,
    "transcribing": AMBER,
    "injecting": GREEN,
    "paused": TEXT_FAINT,
    "error": RED,
}

STATE_LABEL = {
    "idle": "Ready",
    "ready": "Ready",
    "recording": "Listening",
    "transcribing": "Transcribing",
    "injecting": "Inserting",
    "paused": "Paused",
    "error": "Error",
}

PILL_BAR_COLOR = {
    "recording": PILL_BAR,
    "transcribing": "#F2B441",
    "injecting": "#3DDC97",
    "error": "#EF5B5B",
}

SANS = "Segoe UI"
SERIF = "Georgia"


def pretty_hotkey(combo: str) -> str:
    names = {
        "<ctrl>": "Ctrl", "<alt>": "Alt", "<shift>": "Shift", "<cmd>": "Win",
        "<space>": "Space", "<caps_lock>": "Caps Lock", "<esc>": "Esc",
        "<tab>": "Tab", "<enter>": "Enter", "<backspace>": "Backspace",
    }
    parts = []
    for raw in combo.split("+"):
        token = raw.strip()
        parts.append(names.get(token.lower(), token.strip("<>").upper()))
    return " + ".join(parts)


QSS = f"""
* {{
    font-family: "Segoe UI Variable Text", "{SANS}";
    font-size: 13px;
    color: {TEXT};
}}
QMainWindow, #Canvas {{ background: {BG}; }}
#Panel {{ background: {PANEL}; border-radius: 16px; }}
#PageHolder {{ background: {PANEL}; }}
#Sidebar {{ background: transparent; }}
#Brand {{ font-size: 16px; font-weight: 700; background: transparent; }}

QPushButton#Nav {{
    text-align: left; padding: 7px 10px; border: none; border-radius: 8px;
    color: {TEXT}; background: transparent; font-size: 13px;
}}
QPushButton#Nav:hover {{ background: {BORDER_SOFT}; }}
QPushButton#Nav:checked {{ background: {NAV_SELECTED}; font-weight: 600; }}
QPushButton#NavSmall {{
    text-align: left; padding: 5px 10px; border: none; border-radius: 8px;
    color: {TEXT_DIM}; background: transparent; font-size: 12px;
}}
QPushButton#NavSmall:hover {{ background: {BORDER_SOFT}; color: {TEXT}; }}

#Card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; }}
#Row {{ background: transparent; }}
#Hairline {{ background: {BORDER_SOFT}; border: none; max-height: 1px; min-height: 1px; }}
#Banner {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #4B3FBF, stop:0.45 #7C4BD0, stop:1 #C0417A);
    border-radius: 14px;
}}
#BannerTeal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0E7F6B, stop:0.5 #1D9E75, stop:1 #5BA83C);
    border-radius: 14px;
}}
#BannerWarm {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #C2542A, stop:0.5 #DE8A2C, stop:1 #D6A93A);
    border-radius: 14px;
}}
#BannerHead {{
    font-family: "{SERIF}"; font-size: 21px; color: #F3EFE7;
    background: transparent;
}}
#BannerSub {{ color: #C9C4B8; background: transparent; font-size: 12px; }}
QPushButton#BannerBtn {{
    background: #F3EFE7; color: {INK}; border: none; border-radius: 8px;
    padding: 7px 16px; font-weight: 600;
}}
QPushButton#BannerBtn:hover {{ background: #FFFFFF; }}

#H1 {{ font-size: 22px; font-weight: 700; background: transparent; }}
#H2 {{ font-size: 14px; font-weight: 600; background: transparent; }}
#SerifH {{ font-family: "{SERIF}"; font-size: 19px; background: transparent; }}
#Sub {{ color: {TEXT_DIM}; background: transparent; }}
#Faint {{ color: {TEXT_FAINT}; font-size: 12px; background: transparent; }}
#Kicker {{ color: {TEXT_FAINT}; font-size: 11px; font-weight: 600;
          letter-spacing: 1px; background: transparent; }}
#StatNum {{ font-family: "{SERIF}"; font-size: 26px; background: transparent; }}
#StatCap {{ color: {TEXT_FAINT}; font-size: 11px; font-weight: 600;
           letter-spacing: 0.5px; background: transparent; }}

QPushButton {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 6px 14px; color: {TEXT};
}}
QPushButton:hover {{ background: {CARD_TINT}; border-color: #D9D5CC; }}
QPushButton:pressed {{ background: {BORDER_SOFT}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: {CARD_TINT}; }}
QPushButton#Primary {{
    background: {INK}; color: #FFFFFF; border: none; font-weight: 600;
    padding: 7px 16px;
}}
QPushButton#Primary:hover {{ background: {INK_HOVER}; }}
QPushButton#Ghost {{ background: transparent; border: none; color: {TEXT_FAINT}; }}
QPushButton#Ghost:hover {{ background: {BORDER_SOFT}; color: {TEXT}; }}
QPushButton#Danger:hover {{ background: #FBEDED; border-color: {RED}; color: {RED}; }}
QPushButton#Seg {{
    background: transparent; border: none; border-radius: 7px;
    padding: 5px 14px; color: {TEXT_DIM};
}}
QPushButton#Seg:checked {{ background: {CARD}; color: {TEXT};
                           border: 1px solid {BORDER}; font-weight: 600; }}
#SegWrap {{ background: {BORDER_SOFT}; border-radius: 9px; }}

#StyleCard {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; }}
#StyleCardActive {{ background: {CARD}; border: 2px solid {VIOLET}; border-radius: 12px; }}
#Bubble {{ background: #F6EDF7; border-radius: 10px; }}

QLineEdit, QTextEdit, QPlainTextEdit {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 7px 10px; selection-background-color: {VIOLET_SOFT};
    selection-color: {TEXT};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {TEXT_FAINT}; }}

QComboBox {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 5px 10px;
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    selection-background-color: {BORDER_SOFT}; selection-color: {TEXT};
}}

QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{
    width: 34px; height: 20px; border-radius: 10px; background: #D9D5CC;
}}
QCheckBox::indicator:checked {{ background: {INK}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #D9D5CC; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ height: 0; }}

QMenu {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {BORDER_SOFT}; }}
#StatusChip {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; }}
"""
