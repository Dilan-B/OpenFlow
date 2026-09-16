"""Learned corrections: what the user fixed, applied to what they say next.

The gap this closes: OpenFlow mis-hears "Dilan" as "Dylan", the user fixes it
by hand, and the next dictation gets it wrong in exactly the same way. Nothing
in the pipeline ever saw the fix. ``capture.py`` records whether an insertion
was *undone*, which is a one-bit signal -- it says something was wrong, never
what the right answer was. A correction is the complete label: the text we
produced and the text the user wanted.

How a correction is used, in increasing order of leverage:

1. **Replayed deterministically** (:meth:`Corrections.apply`) on every later
   transcript, so the same mistake is repaired before it reaches the screen.
2. **Fed to the cleanup LLM** (:meth:`Corrections.prompt_hint`), so the model
   is told which spellings are already settled and stops "fixing" them back.
3. **Promoted into the dictionary** (:meth:`Corrections.promote`) when the fix
   is a single proper noun, which biases *recognition itself* -- the Whisper
   prompt carries the term, so the error stops happening upstream instead of
   being patched downstream.

Only substitutions are learned. If the user deletes a sentence or adds one,
that is editing their own words, not correcting ours, and replaying it later
would be wrong. Learning only from replacements keeps this to the one thing it
can be confident about: we wrote X, they meant Y.

Everything lives in ``~/.openflow/corrections.json`` and is never transmitted.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .config import CONFIG_DIR

# A correction longer than this on either side is a rewrite, not a fix. Four
# words covers "the gRPC cutover" and every name-plus-surname case; beyond it
# the odds that the span recurs verbatim collapse anyway.
MAX_SPAN_CHUNKS = 4

# Replay a correction automatically once it has been made this many times.
# One sighting is kept but not applied: a single edit is as likely to be the
# user changing their mind as us getting it wrong. A proper noun is exempt --
# see _is_name_fix.
AUTO_APPLY_COUNT = 2

# Keep the store bounded. Corrections are ranked by count, then recency, so
# pruning drops the one-off edits that never recurred.
MAX_ENTRIES = 300

# How many corrections to name in the LLM prompt. These are tokens paid on
# every dictation, so this is deliberately small and ranked by count.
PROMPT_HINT_LIMIT = 12

_CHUNK_RE = re.compile(r"\S+")
_STRIP_RE = re.compile(r"[^\w']+")


@dataclass(slots=True)
class Correction:
    before: str           # what OpenFlow produced
    after: str            # what the user changed it to
    count: int = 1
    at: float = 0.0       # last time it was seen
    applied: int = 0      # times replayed onto a later transcript

    @property
    def key(self) -> str:
        return _norm(self.before)


def _norm(text: str) -> str:
    return " ".join(_STRIP_RE.sub("", c).lower() for c in _CHUNK_RE.findall(text)).strip()


def _is_name_fix(before: str, after: str) -> bool:
    """True when the fix looks like a proper noun or product name.

    These are trusted on a single sighting, because they are the errors the
    user cannot prevent and will otherwise hit forever: the speaker's own
    spelling of a name is not a matter of opinion, and no amount of repeating
    themselves will make the recogniser guess it. An internal capital
    ("OpenFlow", "gRPC") or a leading capital on a word that is not simply the
    sentence case of the original both qualify.
    """
    if not after or len(_CHUNK_RE.findall(after)) > 2:
        return False
    if any(ch.isupper() for ch in after[1:]):
        return True
    return after[:1].isupper() and after.lower() != before.lower()


def diff_corrections(before: str, after: str) -> list[tuple[str, str]]:
    """Extract (was, meant) substitutions between two versions of a text.

    Word-level rather than character-level: a character diff of "Dylan" ->
    "Dilan" yields a one-letter rule that would corrupt every other word
    containing that letter pair. Matching is done on a punctuation-stripped,
    lowercased key so that a comma landing next to a word does not read as a
    change, while the *stored* spans keep their original form -- case is
    exactly what a name fix is usually about.
    """
    before_chunks = _CHUNK_RE.findall(before)
    after_chunks = _CHUNK_RE.findall(after)
    if not before_chunks or not after_chunks:
        return []

    matcher = SequenceMatcher(
        a=[_STRIP_RE.sub("", c).lower() for c in before_chunks],
        b=[_STRIP_RE.sub("", c).lower() for c in after_chunks],
        autojunk=False,
    )

    out: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Matching ignores case, so a run can be "equal" and still hold the
            # single most common correction there is: "grpc" -> "gRPC",
            # "openflow" -> "OpenFlow". Those would be invisible if this branch
            # just skipped ahead.
            for offset in range(i2 - i1):
                was, meant = before_chunks[i1 + offset], after_chunks[j1 + offset]
                was_core = _STRIP_RE.sub("", was)
                meant_core = _STRIP_RE.sub("", meant)
                if was_core == meant_core:
                    continue  # identical, or differing only in punctuation
                # A capital on the first word of the text is sentence case, not
                # a name fix. Learning it would turn every "so" into "So".
                # A real name recurs mid-sentence soon enough to be caught
                # there, so under-learning here costs little.
                if i1 + offset == 0 and was_core[1:] == meant_core[1:]:
                    continue
                out.append((was.strip(), meant.strip()))
            continue
        # Pure insertions and deletions are the user editing their own prose.
        if tag != "replace":
            continue
        if i2 - i1 > MAX_SPAN_CHUNKS or j2 - j1 > MAX_SPAN_CHUNKS:
            continue
        was = " ".join(before_chunks[i1:i2])
        meant = " ".join(after_chunks[j1:j2])
        # Trailing punctuation is not a correction worth replaying.
        was_key, meant_key = _norm(was), _norm(meant)
        if not was_key or not meant_key:
            continue
        if was_key == meant_key and was.lower() == meant.lower():
            continue
        out.append((was.strip(), meant.strip()))
    return out


class Corrections:
    """The learned-correction store. Every mutation persists immediately."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (CONFIG_DIR / "corrections.json")
        self._lock = threading.Lock()
        self.entries: list[Correction] = []
        self.last_applied: list[str] = []
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        entries = []
        for raw in data.get("corrections", []):
            try:
                entries.append(Correction(
                    before=str(raw["before"]), after=str(raw["after"]),
                    count=int(raw.get("count", 1)), at=float(raw.get("at", 0.0)),
                    applied=int(raw.get("applied", 0)),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        self.entries = entries

    def save(self) -> None:
        with self._lock:
            self._prune()
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(
                    json.dumps(
                        {"corrections": [asdict(e) for e in self.entries]}, indent=2
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass

    def _prune(self) -> None:
        if len(self.entries) <= MAX_ENTRIES:
            return
        self.entries.sort(key=lambda e: (e.count, e.at), reverse=True)
        del self.entries[MAX_ENTRIES:]

    # -- learning ----------------------------------------------------------
    def learn(self, before: str, after: str) -> list[Correction]:
        """Record what changed between what we wrote and what the user kept.

        Returns the corrections this edit created or reinforced. A no-op when
        the two texts are the same, which is the common case -- the caller can
        hand every edit to this method without checking first.
        """
        if not before.strip() or not after.strip() or before.strip() == after.strip():
            return []

        touched: list[Correction] = []
        for was, meant in diff_corrections(before, after):
            key = _norm(was)
            for entry in self.entries:
                if entry.key == key and entry.after == meant:
                    entry.count += 1
                    entry.at = time.time()
                    touched.append(entry)
                    break
            else:
                entry = Correction(before=was, after=meant, at=time.time())
                self.entries.append(entry)
                touched.append(entry)
        if touched:
            self.save()
        return touched

    def active(self) -> list[Correction]:
        """Corrections trusted enough to replay, longest span first.

        Longest-first matters: "gRPC cutover" must win over a rule for "gRPC"
        alone, or the shorter replacement fires first and the longer one no
        longer matches what is on the page.
        """
        ready = [
            e for e in self.entries
            if e.count >= AUTO_APPLY_COUNT or _is_name_fix(e.before, e.after)
        ]
        ready.sort(key=lambda e: (len(e.before), e.count), reverse=True)
        return ready

    # -- replay ------------------------------------------------------------
    def apply(self, text: str) -> str:
        """Replay learned corrections onto a fresh transcript."""
        self.last_applied = []
        if not text:
            return text
        for entry in self.active():
            pattern = re.compile(
                r"(?<!\w)" + r"[\s]+".join(
                    re.escape(c) for c in _CHUNK_RE.findall(entry.before)
                ) + r"(?!\w)",
                re.IGNORECASE,
            )
            text, hits = pattern.subn(entry.after.replace("\\", r"\\"), text)
            if hits:
                entry.applied += hits
                self.last_applied.append(f"{entry.before} -> {entry.after}")
        return text

    # -- handing knowledge upstream ---------------------------------------
    def promote(self, personal) -> list[str]:
        """Push single-word name fixes into the dictionary.

        This is the only step that improves *recognition* rather than patching
        its output: dictionary terms ride along in the Whisper prompt, so the
        name is biased for before it is ever mis-heard. Multi-word spans are
        left out -- the vocabulary hint has a tight budget and single terms buy
        more of it.
        """
        added: list[str] = []
        for entry in self.entries:
            if len(_CHUNK_RE.findall(entry.after)) != 1:
                continue
            if not _is_name_fix(entry.before, entry.after):
                continue
            term = entry.after.strip(".,;:!?\"'")
            if term and personal.add_term(term):
                added.append(term)
        return added

    def prompt_hint(self) -> str:
        """An instruction line naming the settled spellings, for the LLM.

        Without this the cleanup model undoes the work: it sees an unfamiliar
        name, decides it is a transcription error, and helpfully restores the
        common spelling we just corrected away from.
        """
        ready = sorted(self.active(), key=lambda e: e.count, reverse=True)
        pairs = [f'"{e.before}" -> "{e.after}"' for e in ready[:PROMPT_HINT_LIMIT]]
        if not pairs:
            return ""
        return (
            "LEARNED CORRECTIONS: this speaker has previously corrected these "
            "exact mistakes. Apply them and never reverse them: "
            + "; ".join(pairs)
            + "."
        )

    # -- housekeeping ------------------------------------------------------
    def forget(self, before: str, after: str) -> None:
        self.entries = [
            e for e in self.entries if not (e.before == before and e.after == after)
        ]
        self.save()

    def purge(self) -> int:
        count = len(self.entries)
        self.entries = []
        self.save()
        return count

    def stats(self) -> dict:
        return {
            "total": len(self.entries),
            "active": len(self.active()),
            "replays": sum(e.applied for e in self.entries),
        }


_shared: Corrections | None = None
_shared_lock = threading.Lock()


def shared() -> Corrections:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = Corrections()
        return _shared
