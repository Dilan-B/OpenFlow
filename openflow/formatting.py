"""Wispr Flow-style Smart Formatting and Flow Styles.

Cleanup decides *which words* survive; this decides how they are laid out for
the place they land. Everything here comes from Wispr Flow's help center
("How do I use Smart Formatting & Backtrack", "How to setup Flow Styles"):

Smart Formatting
  * spoken numbered lists  "are one X two Y" -> "are:\\n1. X\\n2. Y"
  * numbers                "meet at seven" -> "meet at 7"
  * messaging apps         trailing period dropped for up to two sentences;
                           question and exclamation marks always kept
  * context                dictating mid-sentence lowercases the first word to
                           match the surrounding text; spaces are added before
                           and after as needed

Flow Styles, chosen per app category
  * Formal       caps + punctuation
  * Casual       caps + less punctuation (trailing period dropped up to ~10
                 sentences)
  * Very Casual  no caps + less punctuation (personal messages only)
  * Excited      more exclamations (work, email and other only)

Flow Styles change capitalization and punctuation only, never words -- the
same promise cleanup makes, which is why this runs after it and can never undo
an edit it made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .profiles import DEFAULT as DEFAULT_PROFILE
from .profiles import Profile, apply_profile
from .text.casing import is_sentence_case_only, lowercase_words_in
from .text.lists import format_lists
from .text.numbers import format_numbers

# ---------------------------------------------------------------------------
# Categories and styles
# ---------------------------------------------------------------------------
PERSONAL, WORK, EMAIL, OTHER = "personal", "work", "email", "other"
CATEGORIES = (PERSONAL, WORK, EMAIL, OTHER)

CATEGORY_LABELS = {
    PERSONAL: "Personal messages",
    WORK: "Work messages",
    EMAIL: "Email",
    OTHER: "Everything else",
}

FORMAL, CASUAL, VERY_CASUAL, EXCITED = "formal", "casual", "very_casual", "excited"

STYLE_LABELS = {
    FORMAL: "Formal",
    CASUAL: "Casual",
    VERY_CASUAL: "Very casual",
    EXCITED: "Excited!",
}

STYLE_CAPTIONS = {
    FORMAL: "Caps + punctuation",
    CASUAL: "Caps + less punctuation",
    VERY_CASUAL: "No caps + less punctuation",
    EXCITED: "More exclamations",
}

# Wispr offers Very Casual only for personal messages, and Excited only
# outside them.
STYLES_FOR = {
    PERSONAL: (FORMAL, CASUAL, VERY_CASUAL),
    WORK: (FORMAL, CASUAL, EXCITED),
    EMAIL: (FORMAL, CASUAL, EXCITED),
    OTHER: (FORMAL, CASUAL, EXCITED),
}

# Sentences a trailing period is dropped for. None means no limit.
_PERIOD_LIMIT = {CASUAL: 10, VERY_CASUAL: None}
_MESSAGING_PERIOD_LIMIT = 2


@dataclass(slots=True, frozen=True)
class AppKind:
    category: str = OTHER
    messaging: bool = False


# Desktop apps, by lowercased executable (Windows) or app name (macOS).
# Wispr's documented defaults -- Personal: WhatsApp, Telegram, Discord,
# Instagram; Work: Slack, Teams, LinkedIn; Email: Gmail, Superhuman, Outlook,
# Apple Mail -- plus the other messengers on its messaging-app list, placed in
# the category they plainly belong to.
_APPS: dict[str, AppKind] = {}


def _register(kind: AppKind, *names: str) -> None:
    for name in names:
        _APPS[name] = kind


_register(AppKind(PERSONAL, True),
          "whatsapp.exe", "whatsapp.root.exe", "whatsapp", "telegram.exe",
          "telegram", "discord.exe", "discord", "signal.exe", "signal",
          "wechat.exe", "weixin.exe", "wechat", "line.exe", "line",
          "beeper.exe", "beeper", "texts.exe", "texts", "messages",
          "messenger.exe", "messenger")
_register(AppKind(WORK, True),
          "slack.exe", "slack", "ms-teams.exe", "teams.exe", "microsoft teams",
          "lark.exe", "feishu.exe", "lark", "google chat")
_register(AppKind(WORK, False), "linkedin")
_register(AppKind(EMAIL, False),
          "outlook.exe", "olk.exe", "hxoutlook.exe", "microsoft outlook",
          "superhuman.exe", "superhuman", "mail", "thunderbird.exe",
          "thunderbird", "mailspring.exe", "mailspring")

BROWSERS = frozenset({
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "arc.exe",
    "opera.exe", "vivaldi.exe", "google chrome", "safari", "firefox",
    "microsoft edge", "brave browser", "arc", "opera", "vivaldi",
})

# Web apps, recognised by a whole segment of the browser tab title, so an
# article *about* Slack does not count as Slack.
_WEB_APPS: dict[str, AppKind] = {
    "gmail": AppKind(EMAIL, False),
    "outlook": AppKind(EMAIL, False),
    "superhuman": AppKind(EMAIL, False),
    "slack": AppKind(WORK, True),
    "microsoft teams": AppKind(WORK, True),
    "google chat": AppKind(WORK, True),
    "linkedin": AppKind(WORK, False),
    "whatsapp": AppKind(PERSONAL, True),
    "telegram": AppKind(PERSONAL, True),
    "telegram web": AppKind(PERSONAL, True),
    "discord": AppKind(PERSONAL, True),
    "instagram": AppKind(PERSONAL, True),
    "messenger": AppKind(PERSONAL, True),
    "facebook": AppKind(OTHER, True),
    "x": AppKind(OTHER, True),
    "twitter": AppKind(OTHER, True),
    "reddit": AppKind(OTHER, True),
}

_TITLE_SPLIT = re.compile(r"\s+[-—–|/•]\s+|\u200b")
_UNREAD_PREFIX = re.compile(r"^\(\d+\+?\)\s*")


def classify(process: str, title: str = "",
             overrides: dict[str, str] | None = None) -> AppKind:
    """Which Wispr category an app belongs to, and whether it is a messenger."""
    key = (process or "").lower()
    if overrides and key in overrides and overrides[key] in CATEGORIES:
        known = _APPS.get(key, AppKind())
        return AppKind(overrides[key], known.messaging)
    if key in _APPS:
        return _APPS[key]
    if key in BROWSERS and title:
        for segment in _TITLE_SPLIT.split(title):
            name = _UNREAD_PREFIX.sub("", segment).strip().lower()
            if name in _WEB_APPS:
                return _WEB_APPS[name]
    return AppKind()


# ---------------------------------------------------------------------------
# Caret context
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class CaretContext:
    """Text immediately around the insertion point in the target app."""

    before: str = ""
    after: str = ""
    has_selection: bool = False
    # The highlighted text itself, when the app exposes it. Command Mode edits
    # this; formatting only cares whether a selection exists.
    selected: str = ""


# ---------------------------------------------------------------------------
# The passes
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=\S)")
_FIRST_WORD = re.compile(r"^(\W*)([A-Za-z][A-Za-z']*)")


def _sentence_count(text: str) -> int:
    return len([s for s in _SENTENCE_SPLIT.split(text.strip()) if s])


def _drop_trailing_period(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith(".") and not stripped.endswith("..") \
            and not re.search(r"\b[A-Za-z]\.[A-Za-z]\.$", stripped):
        return stripped[:-1]
    return text


def _lower_first_word(text: str, protected: set[str], seen: set[str]) -> str:
    match = _FIRST_WORD.match(text)
    if match and is_sentence_case_only(match.group(2), protected=protected,
                                       seen_lowercase=seen):
        word = match.group(2)
        return match.group(1) + word[0].lower() + word[1:] + text[match.end():]
    return text


def _lowercase_sentence_starts(text: str, protected: set[str]) -> str:
    seen = lowercase_words_in(text)
    lines = []
    for line in text.split("\n"):
        parts = _SENTENCE_SPLIT.split(line)
        lines.append(" ".join(_lower_first_word(p, protected, seen) for p in parts))
    return "\n".join(lines)


def _excite(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith(".") and not stripped.endswith(".."):
        return stripped[:-1] + "!"
    return text


def _fit_to_context(text: str, context: CaretContext, protected: set[str]) -> str:
    before = context.before.rstrip(" \t")
    after = context.after

    # Mid-sentence: the text before the caret has not ended its sentence.
    if before and not before.endswith(("\n", ".", "!", "?", ":")) and \
            not context.before.endswith("\n"):
        seen = lowercase_words_in(text) | lowercase_words_in(context.before)
        text = _lower_first_word(text, protected, seen)

    # Existing punctuation right after the caret owns the sentence ending.
    next_char = after.lstrip(" \t")[:1]
    if next_char and next_char in ".,;:!?":
        text = text.rstrip().rstrip(".,;:")

    if context.before and not context.before[-1].isspace() \
            and context.before[-1] not in "([{“\"'/@#\n" and text[:1] not in ".,;:!?)]}”":
        text = " " + text
    if after and not after[0].isspace() and after[0] not in ".,;:!?)]}”\"'":
        text = text + " "
    return text


def smart_format(
    text: str,
    *,
    kind: AppKind = AppKind(),
    style: str = FORMAL,
    profile: Profile = DEFAULT_PROFILE,
    context: CaretContext | None = None,
    smart: bool = True,
    protected_terms: tuple[str, ...] = (),
) -> str:
    """Lay finished text out for the app it is about to land in."""
    if not text or not text.strip():
        return text
    protected = {t.lower() for t in protected_terms}
    out = text

    is_shell = profile.name == "shell"
    if smart and not is_shell:
        out = format_lists(out)
        out = format_numbers(out)

    # Code and shell profiles already fix casing, periods and quotes for
    # syntax; a style on top would fight them.
    # With Smart Formatting off, the chat profile keeps its old job of
    # dropping the full stop; with it on, the messaging rule below does that.
    if profile.name in ("code", "shell") or (not smart and profile.name == "chat"):
        out = apply_profile(out, profile)
    else:
        if style not in STYLES_FOR.get(kind.category, STYLES_FOR[OTHER]):
            style = FORMAL
        is_list = "\n1. " in out or out.startswith("1. ")
        if style == VERY_CASUAL and not is_list:
            out = _lowercase_sentence_starts(out, protected)
        if style == EXCITED:
            out = _excite(out)

        sentences = _sentence_count(out)
        limit = _PERIOD_LIMIT.get(style, 0)
        drop = style in _PERIOD_LIMIT and (limit is None or sentences <= limit)
        if smart and kind.messaging and sentences <= _MESSAGING_PERIOD_LIMIT \
                and not (context and context.has_selection):
            drop = True
        if drop and not is_list:
            out = _drop_trailing_period(out)

    if smart and context is not None:
        out = _fit_to_context(out, context, protected)
    return out
