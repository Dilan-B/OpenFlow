"""Command Mode: speak an instruction instead of dictating text.

Wispr Flow's Command Mode, from its help center ("How to use Command Mode"):
hold a second shortcut, say what you want, release. Two kinds of command:

* **Text editing** -- must open with a "Hey Flow" variant, and rewrites the
  text you have selected: "hey flow, make this shorter".
* **Web search** -- opens Google, Perplexity, ChatGPT or Claude with what you
  said, plus any selected text: "search google for flight times to Denver".

Anything that opens with neither is not a command. Wispr does nothing in that
case; OpenFlow says so on the pill instead, because a command that silently
evaporates is indistinguishable from a broken hotkey.

Parsing is deliberately literal. This module turns speech into an *intent* and
nothing else -- no model, no network -- so the rules that decide whether your
words get sent to a search engine are readable in one screen.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass

# "hey flow", "hi flow", "ok flow", "flow," -- the wake phrase for an edit.
_WAKE = re.compile(
    r"^\s*(?:(?:hey|hi|hello|ok|okay|yo)\s*,?\s*)?flow\s*[,:]?\s+(?P<rest>.+)$",
    re.IGNORECASE | re.DOTALL,
)

# "search google for X", "ask claude about X", "hey chatgpt X".
_SEARCH = re.compile(
    # The opener is required, exactly as documented: "ask", "search" or "hey"
    # followed by the engine. A bare "google the weather" is dictation.
    r"^\s*\b(?:ask|search|hey)\b\s*,?\s*"
    # The lookahead, not \b: a hyphen is a word boundary, so \b alone lets
    # "google-ish thing" through as a Google search.
    r"(?P<engine>google|perplexity|chat\s*gpt|claude)(?=[\s,.:!?]|$)\s*"
    # Only the connectors Wispr documents. Question words belong to the query:
    # swallowing "what" turns "what is the capital of Peru" into "is the
    # capital of Peru".
    r"(?:,\s*)?(?:about|for|on|to)?\s*"
    r"(?P<query>.*)$",
    re.IGNORECASE | re.DOTALL,
)

# Where each engine takes a query. Kept here, visible, rather than built from
# the spoken words: a command must never be able to name its own destination.
SEARCH_URLS = {
    "google": "https://www.google.com/search?q=",
    "perplexity": "https://www.perplexity.ai/search?q=",
    "chatgpt": "https://chatgpt.com/?q=",
    "claude": "https://claude.ai/new?q=",
}

ENGINE_LABELS = {
    "google": "Google", "perplexity": "Perplexity",
    "chatgpt": "ChatGPT", "claude": "Claude",
}

EDIT_PROMPT = (
    "You edit text on command. The user selected some text and spoke an "
    "instruction about it. Apply the instruction and output only the resulting "
    "text: no preamble, no quotes, no explanation, no commentary. Keep the "
    "speaker's meaning and any names or technical terms. If the instruction "
    "asks a question about the text rather than a change to it, answer in "
    "place, as plainly as possible."
)


@dataclass(slots=True, frozen=True)
class Command:
    """What a spoken command asks for.

    ``kind`` is "edit", "search" or "none". An "edit" carries the instruction;
    a "search" carries the engine and the query it would open.
    """

    kind: str
    instruction: str = ""
    engine: str = ""
    query: str = ""
    spoken: str = ""

    @property
    def url(self) -> str:
        if self.kind != "search":
            return ""
        return SEARCH_URLS[self.engine] + urllib.parse.quote_plus(self.query)

    @property
    def summary(self) -> str:
        if self.kind == "search":
            return f"{ENGINE_LABELS[self.engine]}: {self.query}"
        if self.kind == "edit":
            return self.instruction
        return ""


def _tidy(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip(".,;:!? ")


def parse(spoken: str, *, selection: str = "") -> Command:
    """Turn a spoken phrase into a :class:`Command`.

    ``selection`` is appended to a search query, the way Wispr appends
    highlighted text to the words you said.
    """
    text = (spoken or "").strip()
    if not text:
        return Command("none", spoken=spoken)

    # Search is checked first: "hey chatgpt ..." also matches nothing else,
    # and "hey flow, ask claude ..." should search rather than edit.
    wake = _WAKE.match(text)
    inner = wake.group("rest") if wake else text

    search = _SEARCH.match(inner)
    if search and (search.group("query").strip() or selection.strip()):
        engine = re.sub(r"\s+", "", search.group("engine")).lower()
        query = _tidy(search.group("query"))
        extra = _tidy(selection)
        full = " ".join(part for part in (query, extra) if part)
        if full:
            return Command("search", engine=engine, query=full, spoken=spoken)

    if wake:
        instruction = _tidy(wake.group("rest"))
        if instruction:
            return Command("edit", instruction=instruction, spoken=spoken)

    return Command("none", spoken=spoken)
