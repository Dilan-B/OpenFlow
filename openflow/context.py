"""What is on screen when a dictation starts -- Wispr Flow's context awareness.

Wispr "gets names right" because it looks at what you are replying to: the
name in the email thread, the person in the Slack channel, the file open in
your editor. This module gathers the same three things when the hotkey goes
down, in the background, so nothing on the hotkey path waits for it:

* **names** -- proper nouns in the focused window's visible text (UI
  Automation on Windows, the Accessibility API on macOS);
* **files** -- in an IDE (Cursor, VS Code, Windsurf), the files of the project
  that window has open, so "look at auth dot ts" can become ``@auth.ts``;
* **identifiers** -- the multi-word variable, function and class names used in
  that project, so "get user name" can come back as ``getUserName``.

All of it feeds the transcriber's vocabulary hint, the cleanup model's list of
known terms, and the containment guard's allow-list. Nothing read here is
stored or sent anywhere except inside those requests, password fields are never
read, and every failure degrades to an empty context -- which is exactly how
OpenFlow behaved before this existed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

log = logging.getLogger(__name__)

WINDOW_TEXT_CHARS = 20_000
MAX_NAMES = 40
MAX_FILES = 4_000
MAX_IDENTIFIERS = 300
# The UI Automation walk crosses into the target process; bound it.
UIA_TIMEOUT_MS = 400

IDE_APPS = {
    # process / app name -> folder under %APPDATA% (or Application Support)
    "cursor.exe": "Cursor", "cursor": "Cursor",
    "code.exe": "Code", "code": "Code", "visual studio code": "Code",
    "code - insiders.exe": "Code - Insiders",
    "windsurf.exe": "Windsurf", "windsurf": "Windsurf",
}


@dataclass(slots=True, frozen=True)
class DictationContext:
    app: str = ""
    title: str = ""
    names: tuple[str, ...] = ()
    files: tuple[str, ...] = ()           # workspace-relative paths
    identifiers: tuple[str, ...] = ()
    workspace: str = ""

    @property
    def is_ide(self) -> bool:
        return self.app.lower() in IDE_APPS

    def file_names(self) -> list[str]:
        """Bare file names, unique, in index order."""
        seen: dict[str, None] = {}
        for path in self.files:
            seen.setdefault(path.rsplit("/", 1)[-1], None)
        return list(seen)

    def terms(self, budget: int = 400) -> str:
        """Comma-separated hint for a transcriber prompt, most useful first."""
        out, used = [], 0
        pool = list(self.names) + self.file_names()[:60] + list(self.identifiers)
        for term in pool:
            used += len(term) + 2
            if used > budget:
                break
            out.append(term)
        return ", ".join(out)

    def allowed_words(self) -> set[str]:
        """Words cleanup may write although the transcript did not contain
        them -- a name spelled the way the screen spells it, a file tag."""
        words: set[str] = set()
        for term in list(self.names) + self.file_names() + list(self.identifiers):
            words.update(w.lower() for w in re.findall(r"[A-Za-z0-9]+", term))
        return words


EMPTY = DictationContext()


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’\-]*[A-Za-z]|[A-Za-z]")
_GREETING = re.compile(
    r"(?:(?i:\b(?:hi|hey|hello|dear|thanks|thank you|cheers|best|regards|from|to|cc))"
    r"|@)[ \t]*[,:]?\s+([A-Z][A-Za-z'’\-]+(?:[ \t]+[A-Z][A-Za-z'’\-]+)?)")
_MIXED = re.compile(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b")


_CALENDAR = frozenset("""
monday tuesday wednesday thursday friday saturday sunday january february
march april june july august september october november december today
tomorrow yesterday
""".split())


def _is_ordinary(word: str) -> bool:
    from .phonetics import COMMON_WORDS
    from .text.casing import OPENERS, _COMMON_SUFFIXES

    lower = word.lower()
    return (lower in COMMON_WORDS or lower in OPENERS or lower in _CALENDAR
            or (len(lower) > 5 and lower.endswith(_COMMON_SUFFIXES)))


def extract_names(text: str, limit: int = MAX_NAMES) -> list[str]:
    """Proper nouns in ``text``, most frequent first.

    A capitalized word counts only with evidence it is a name, not a label or
    a sentence start: it appears capitalized mid-sentence (after a lowercase
    word or a comma), it follows a greeting or a header ("Hi Sahed", "From:
    Sydney Park"), or it is mixed-case ("OpenFlow", "iPhone"). A word the text
    also writes in lowercase is never a name.
    """
    if not text:
        return []
    # Addresses and links are lowercase by convention, not evidence that the
    # name in them is an ordinary word: "sahed@x.com" must not veto "Sahed".
    text = re.sub(r"\S+@\S+|https?://\S+|www\.\S+", " ", text)
    lowercase = {w.lower() for w in _TOKEN.findall(text) if w.islower()}
    counts: Counter[str] = Counter()

    def credit(name: str, weight: int = 1) -> None:
        parts = name.split()
        if not parts or any(p.lower() in lowercase for p in parts):
            return
        if len(parts) == 1 and (_is_ordinary(name) or len(name) < 3):
            return
        counts[name] += weight

    for match in _GREETING.finditer(text):
        full = match.group(1).strip()
        credit(full, 3)
        if " " in full:
            for part in full.split():
                credit(part, 2)
    for match in _MIXED.finditer(text):
        credit(match.group(0), 1)

    for line in text.splitlines():
        tokens = list(_TOKEN.finditer(line))
        for i, tok in enumerate(tokens):
            word = tok.group(0)
            if not word[:1].isupper() or i == 0:
                continue
            between = line[tokens[i - 1].end():tok.start()]
            if re.search(r"[.!?:;|•\-–—]", between):
                continue         # a sentence or a label starts here
            prev = tokens[i - 1].group(0)
            if prev[:1].isupper() and prev.lower() not in lowercase \
                    and not _is_ordinary(prev) and between.strip() == "":
                continue         # second half of a two-word name, credited below
            if not (prev.islower() or "," in between):
                continue
            credit(word)
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is not None and nxt.group(0)[:1].isupper() \
                    and line[tok.end():nxt.start()].strip() == "" \
                    and not _is_ordinary(nxt.group(0)):
                credit(nxt.group(0))
                credit(f"{word} {nxt.group(0)}")
    return [name for name, _ in counts.most_common(limit)]


# ---------------------------------------------------------------------------
# Window text
# ---------------------------------------------------------------------------
_UIA_NAME = 30005
_UIA_VALUE = 30045
_UIA_IS_PASSWORD = 30019
_TREE_SCOPE_DESCENDANTS = 4


def read_window_text() -> str:
    try:
        if sys.platform == "win32":
            return _window_text_windows()
        if sys.platform == "darwin":
            return _window_text_macos()
    except Exception as exc:     # accessibility fails in app-specific ways
        log.debug("window text unavailable: %s", exc)
    return ""


def _window_text_windows() -> str:
    import ctypes

    import comtypes
    import comtypes.client

    from .input.caret import _uia_module

    comtypes.CoInitialize()
    try:
        uia = _uia_module()
        automation = comtypes.client.CreateObject(
            uia.CUIAutomation, interface=uia.IUIAutomation)
        try:
            bounded = automation.QueryInterface(uia.IUIAutomation2)
            bounded.ConnectionTimeout = UIA_TIMEOUT_MS
            bounded.TransactionTimeout = UIA_TIMEOUT_MS
        except Exception:
            pass
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return ""
        root = automation.ElementFromHandle(hwnd)
        parts: list[str] = []
        used = 0

        # One cross-process call for the whole tree: every element's name,
        # value and password flag come back cached.
        request = automation.CreateCacheRequest()
        for prop in (_UIA_NAME, _UIA_VALUE, _UIA_IS_PASSWORD):
            request.AddProperty(prop)
        found = root.FindAllBuildCache(
            _TREE_SCOPE_DESCENDANTS, automation.CreateTrueCondition(), request)
        for i in range(found.Length if found is not None else 0):
            element = found.GetElement(i)
            try:
                if element.GetCachedPropertyValue(_UIA_IS_PASSWORD):
                    continue
            except Exception:
                pass
            for prop in (_UIA_NAME, _UIA_VALUE):
                try:
                    value = element.GetCachedPropertyValue(prop)
                except Exception:
                    continue
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    used += len(value)
            if used > WINDOW_TEXT_CHARS:
                break
        return "\n".join(parts)[:WINDOW_TEXT_CHARS]
    finally:
        comtypes.CoUninitialize()


def _window_text_macos() -> str:
    from ApplicationServices import (  # type: ignore[import-not-found]
        AXUIElementCopyAttributeValue,
        AXUIElementCreateSystemWide,
        kAXFocusedUIElementAttribute,
        kAXSubroleAttribute,
        kAXValueAttribute,
    )

    system = AXUIElementCreateSystemWide()
    err, element = AXUIElementCopyAttributeValue(system, kAXFocusedUIElementAttribute, None)
    if err or element is None:
        return ""
    err, subrole = AXUIElementCopyAttributeValue(element, kAXSubroleAttribute, None)
    if not err and subrole == "AXSecureTextField":
        return ""
    err, value = AXUIElementCopyAttributeValue(element, kAXValueAttribute, None)
    return value[:WINDOW_TEXT_CHARS] if not err and isinstance(value, str) else ""


# ---------------------------------------------------------------------------
# IDE workspace
# ---------------------------------------------------------------------------
def _storage_path(folder: str) -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path.home() / ".config"
    return base / folder / "User" / "globalStorage" / "storage.json"


def _uri_to_path(uri: str) -> Path | None:
    if not uri or not uri.startswith("file:"):
        return None
    path = unquote(urlparse(uri).path)
    if sys.platform == "win32" and re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    return Path(path)


def ide_workspace(app: str, title: str) -> Path | None:
    """The folder the focused IDE window has open, read from the editor's own
    window-state file and matched against the window title."""
    folder = IDE_APPS.get(app.lower())
    if not folder:
        return None
    try:
        data = json.loads(_storage_path(folder).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    state = data.get("windowsState") or {}
    uris: list[str] = []
    for window in [state.get("lastActiveWindow") or {}] + list(state.get("openedWindows") or []):
        if window.get("folder"):
            uris.append(window["folder"])
    for entry in (data.get("backupWorkspaces") or {}).get("folders") or []:
        if entry.get("folderUri"):
            uris.append(entry["folderUri"])
    candidates = [p for p in (_uri_to_path(u) for u in uris) if p is not None]
    segments = {s.strip().lower() for s in re.split(r"\s+[-—–]\s+", title or "")}
    for path in candidates:
        if path.name.lower() in segments and path.is_dir():
            return path
    for path in candidates:
        if path.is_dir():
            return path
    return None


_SKIP_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__", "dist",
    "build", "out", ".next", "target", ".idea", ".vscode", "coverage",
    ".mypy_cache", ".pytest_cache", "vendor", "Pods", ".gradle", "bin", "obj",
})
_SOURCE_EXT = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs", ".java",
    ".kt", ".swift", ".rb", ".php", ".cs", ".cpp", ".cc", ".c", ".h", ".hpp",
    ".vue", ".svelte", ".scala", ".dart", ".sql", ".m",
})
_IDENT = re.compile(r"\b(?:[a-z]+(?:[A-Z][a-z0-9]+)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+"
                    r"|[a-z]+(?:_[a-z0-9]+)+)\b")


def list_files(root: Path, limit: int = MAX_FILES) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.splitlines()[:limit]
    except (OSError, subprocess.SubprocessError):
        pass
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        rel = Path(dirpath).relative_to(root).as_posix()
        for name in filenames:
            files.append(name if rel == "." else f"{rel}/{name}")
            if len(files) >= limit:
                return files
    return files


def index_identifiers(root: Path, files: list[str], limit: int = MAX_IDENTIFIERS,
                      max_files: int = 400, max_bytes: int = 200_000) -> list[str]:
    """Multi-word identifiers (camelCase, PascalCase, snake_case) by use count.

    Single-word names are left out on purpose: "users" or "config" are also
    English, and rewriting English into code would be the worse mistake.
    """
    counts: Counter[str] = Counter()
    sources = [f for f in files if Path(f).suffix.lower() in _SOURCE_EXT][:max_files]
    for rel in sources:
        try:
            path = root / rel
            if path.stat().st_size > max_bytes:
                continue
            counts.update(_IDENT.findall(path.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return [name for name, _ in counts.most_common(limit)]


_workspace_cache: dict[str, tuple[float, tuple[str, ...], tuple[str, ...]]] = {}
_workspace_lock = threading.Lock()
WORKSPACE_TTL_S = 300.0


def workspace_index(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    key = str(root)
    now = time.monotonic()
    with _workspace_lock:
        cached = _workspace_cache.get(key)
        if cached and now - cached[0] < WORKSPACE_TTL_S:
            return cached[1], cached[2]
    files = list_files(root)
    identifiers = index_identifiers(root, files)
    with _workspace_lock:
        _workspace_cache[key] = (now, tuple(files), tuple(identifiers))
    return tuple(files), tuple(identifiers)


# ---------------------------------------------------------------------------
# Spoken file references
# ---------------------------------------------------------------------------
def tag_files(text: str, files: list[str]) -> str:
    """Turn explicit spoken file names into Cursor/Windsurf ``@`` tags.

    Only unambiguous forms are rewritten here: the name with its extension,
    as the transcriber wrote it ("auth.ts") or as spoken ("auth dot ts").
    Bare mentions ("look at auth") are left to the cleanup model, which sees
    the project's file list and can tell a file from an English word.
    """
    if not text or not files:
        return text
    by_name: dict[str, str] = {}
    for name in files:
        by_name.setdefault(name.lower(), name)
    for name in sorted(by_name.values(), key=len, reverse=True):
        if "." not in name.strip("."):
            continue
        stem, ext = name.rsplit(".", 1)
        spoken_stem = r"[\s_\-]*".join(re.escape(p) for p in _split_words(stem))
        pattern = re.compile(
            rf"(?<![@\w/.])(?:{re.escape(stem)}|{spoken_stem})(?:\s*\.\s*|\s+dot\s+){re.escape(ext)}\b",
            re.IGNORECASE)
        text = pattern.sub("@" + name, text)
    return text


def _split_words(identifier: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", identifier)
    return [w for w in re.split(r"[\s_\-.]+", spaced) if w]


# ---------------------------------------------------------------------------
# Background reader
# ---------------------------------------------------------------------------
def gather(app: str, title: str) -> DictationContext:
    names: list[str] = []
    files: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    workspace = ""
    root = ide_workspace(app, title)
    if root is not None:
        workspace = str(root)
        try:
            files, identifiers = workspace_index(root)
        except Exception as exc:
            log.debug("could not index %s: %s", root, exc)
    else:
        text = read_window_text()
        if len(text) < 200:
            # Chromium and Electron (Slack, Discord, every browser) build
            # their accessibility tree only once a client asks for it, so the
            # first read is nearly empty. The user is still talking; ask again.
            time.sleep(0.7)
            text = read_window_text() or text
        names = extract_names(f"{title}\n{text}")
    return DictationContext(app=app, title=title, names=tuple(names), files=files,
                            identifiers=identifiers, workspace=workspace)


class ContextReader:
    """Gather the context off-thread while the user is still speaking."""

    def __init__(self, app: str, title: str) -> None:
        self.app, self.title = app, title
        self._result: DictationContext | None = None
        self._done = threading.Event()

    def start(self) -> "ContextReader":
        threading.Thread(target=self._run, name="openflow-context", daemon=True).start()
        return self

    def _run(self) -> None:
        try:
            self._result = gather(self.app, self.title)
        except Exception as exc:
            log.debug("context gathering failed: %s", exc)
        finally:
            self._done.set()

    def result(self, wait_s: float = 0.15) -> DictationContext:
        self._done.wait(wait_s)
        if self._done.is_set() and self._result is not None:
            return self._result
        return DictationContext(app=self.app, title=self.title)


def repair_names(text: str, names) -> str:
    """Respell near-misses of on-screen names the way the screen spells them.

    The same fuzzy and phonetic matcher as the personal dictionary, so the
    same guards apply: short names only fix casing, and an ordinary English
    word that merely sounds like a name is left alone.
    """
    if not text or not names:
        return text
    from .personalization import _fix_term

    for name in names:
        if len(name.replace(" ", "")) >= 4:
            text, _count = _fix_term(text, name)
    return text


def with_own_name(context: DictationContext, name: str) -> DictationContext:
    """The speaker's own name is always a known name -- it ends every email
    they sign ("Thanks, Dilan"), and it is the one name a transcriber cannot
    guess the spelling of from sound alone."""
    name = (name or "").strip()
    if not name or name in context.names:
        return context
    from dataclasses import replace

    extra = tuple(dict.fromkeys(n for n in (name, name.split()[0]) if n not in context.names))
    return replace(context, names=context.names + extra)
