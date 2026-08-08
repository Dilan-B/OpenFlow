"""Dictionary, snippets, and style presets.

Dictionary is the accuracy lever for "non-words": product names, companies,
jargon. It works on two fronts --

  1. Cloud Whisper accepts a vocabulary hint in its prompt, biasing
     recognition toward your terms before they're ever mis-heard.
  2. A deterministic fuzzy pass repairs what still comes back wrong:
     "open flow" / "openflo" -> "OpenFlow", "grok" -> "Groq".

Snippets expand a spoken trigger into saved text ("my email sig" -> the whole
signature). Styles append a tone instruction to the LLM cleanup prompt.

All of it lives in ~/.openflow/personalization.json.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import CONFIG_DIR

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9']*")

STYLES: dict[str, str] = {
    "default": "",
    "professional": (
        "STYLE: The speaker wants polished professional prose. Prefer complete "
        "sentences and neutral wording, but change no facts and add no content."
    ),
    "casual": (
        "STYLE: Keep the speaker's casual, conversational tone exactly as is. "
        "Do not formalize their wording."
    ),
    "concise": (
        "STYLE: The speaker wants brevity. You may drop redundant qualifiers "
        "and repeated phrases, but never drop information."
    ),
    "email": (
        "STYLE: This dictation is an email body. Break it into short paragraphs "
        "at topic changes. Change no wording."
    ),
}


@dataclass(slots=True)
class Snippet:
    trigger: str
    text: str


@dataclass(slots=True)
class Personalization:
    dictionary: list[str] = field(default_factory=list)
    snippets: list[Snippet] = field(default_factory=list)
    style: str = "default"
    last_fixes: int = 0    # replacements made by the most recent apply()

    _path: Path = None  # type: ignore[assignment]
    _lock: threading.Lock = None  # type: ignore[assignment]

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> "Personalization":
        path = path or (CONFIG_DIR / "personalization.json")
        inst = cls()
        inst._path = path
        inst._lock = threading.Lock()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            inst.dictionary = [str(t) for t in data.get("dictionary", [])]
            inst.snippets = [
                Snippet(str(s["trigger"]), str(s["text"]))
                for s in data.get("snippets", [])
                if s.get("trigger")
            ]
            if data.get("style") in STYLES:
                inst.style = data["style"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
        return inst

    def save(self) -> None:
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(
                    json.dumps(
                        {
                            "dictionary": self.dictionary,
                            "snippets": [
                                {"trigger": s.trigger, "text": s.text}
                                for s in self.snippets
                            ],
                            "style": self.style,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass

    # -- dictionary --------------------------------------------------------
    def add_term(self, term: str) -> bool:
        term = term.strip()
        if not term or term.lower() in (t.lower() for t in self.dictionary):
            return False
        self.dictionary.append(term)
        self.save()
        return True

    def remove_term(self, term: str) -> None:
        self.dictionary = [t for t in self.dictionary if t != term]
        self.save()

    def vocabulary_hint(self, budget: int = 180) -> str:
        """Comma-separated vocabulary for the cloud Whisper prompt."""
        out: list[str] = []
        used = 0
        for term in self.dictionary:
            used += len(term) + 2
            if used > budget:
                break
            out.append(term)
        return ", ".join(out)

    def apply_dictionary(self, text: str) -> str:
        self.last_fixes = 0
        if not self.dictionary or not text:
            return text
        for term in self.dictionary:
            text, fixed = _fix_term(text, term)
            self.last_fixes += fixed
        return text

    # -- snippets ----------------------------------------------------------
    def add_snippet(self, trigger: str, body: str) -> bool:
        trigger, body = trigger.strip(), body.strip()
        if not trigger or not body:
            return False
        self.snippets = [s for s in self.snippets if _norm(s.trigger) != _norm(trigger)]
        self.snippets.append(Snippet(trigger, body))
        self.save()
        return True

    def remove_snippet(self, trigger: str) -> None:
        self.snippets = [s for s in self.snippets if s.trigger != trigger]
        self.save()

    def apply_snippets(self, text: str) -> str:
        """If the whole utterance is a snippet trigger, expand it."""
        spoken = _norm(text)
        if not spoken:
            return text
        for snippet in self.snippets:
            if spoken == _norm(snippet.trigger):
                return snippet.text
        return text

    # -- style -------------------------------------------------------------
    def set_style(self, name: str) -> None:
        if name in STYLES:
            self.style = name
            self.save()

    def style_instruction(self) -> str:
        return STYLES.get(self.style, "")

    # -- the full post-pass ------------------------------------------------
    def apply(self, text: str) -> str:
        return self.apply_dictionary(self.apply_snippets(text))


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _distance(a: str, b: str, cutoff: int) -> int:
    """Levenshtein with early exit once every path exceeds ``cutoff``."""
    if abs(len(a) - len(b)) > cutoff:
        return cutoff + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        best = i
        for j, cb in enumerate(b, 1):
            cost = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            )
            current.append(cost)
            best = min(best, cost)
        if best > cutoff:
            return cutoff + 1
        previous = current
    return previous[-1]


def _tolerance(term: str) -> int:
    """How many edits away a transcription may be and still mean this term.

    Short terms get zero tolerance beyond casing: at four letters, one edit is
    the difference between "Groq" and the common word "grow", and silently
    rewriting real words is worse than missing a correction.
    """
    stripped = term.replace(" ", "")
    if len(stripped) <= 4:
        return 0
    if len(stripped) <= 8:
        return 1
    return 2


def _fix_term(text: str, term: str) -> tuple[str, int]:
    from .phonetics import is_common_word, phonetically_matchable, phrase_key

    term_words = term.split()
    term_key = term.replace(" ", "").lower()
    cutoff = _tolerance(term)
    # Phonetic matching is what catches names the transcriber has never seen
    # -- "cuber netties" for "Kubernetes". Only long-enough keys qualify.
    phonetic = phonetically_matchable(term)
    term_sound = phrase_key(term_words) if phonetic else ""

    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return text, 0

    window = len(term_words)
    replacements: list[tuple[int, int]] = []   # (start, end) spans in text
    taken_until = -1

    for size in (window, window + 1) if window > 1 else (1, 2):
        for i in range(len(matches) - size + 1):
            span_start = matches[i].start()
            span_end = matches[i + size - 1].end()
            if span_start <= taken_until:
                continue
            words_here = [m.group(0) for m in matches[i:i + size]]
            candidate = "".join(words_here).lower()
            if candidate == term_key:
                if text[span_start:span_end] != term:
                    replacements.append((span_start, span_end))
                    taken_until = span_end
            elif cutoff and _distance(candidate, term_key, cutoff) <= cutoff:
                replacements.append((span_start, span_end))
                taken_until = span_end
            elif phonetic and phrase_key(words_here) == term_sound:
                # A single ordinary English word that merely rhymes with the
                # term is far likelier to be what the speaker said.
                if not (size == 1 and is_common_word(words_here[0])):
                    replacements.append((span_start, span_end))
                    taken_until = span_end

    for start, end in sorted(replacements, reverse=True):
        text = text[:start] + term + text[end:]
    return text, len(replacements)


# Module-level instance shared by the STT engines (for the vocabulary hint)
# and the app (for the post-pass). Loaded lazily so tests can build their own.
_shared: Personalization | None = None


def shared() -> Personalization:
    global _shared
    if _shared is None:
        _shared = Personalization.load()
    return _shared
