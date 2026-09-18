"""Provider API keys saved from the Settings page.

The environment still wins -- GROQ_API_KEY and friends work as before. But a
Mac app started from the Dock never sees what a shell profile exports, so
there is also somewhere the app itself can keep a key: the macOS login
Keychain, through the ``security`` tool that ships with the OS.

Keys never go in config.json. On other platforms saving is unsupported for
now and the environment is the only source.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading

log = logging.getLogger(__name__)

SERVICE = "OpenFlow"

# The keys the Settings page can save, with where to get a free one.
KNOWN = {
    "GROQ_API_KEY": ("Groq", "https://console.groq.com/keys"),
    "GEMINI_API_KEY": ("Gemini", "https://aistudio.google.com/apikey"),
}

# Provider keys are URL-safe tokens. Anything else is a paste accident, and
# refusing it also keeps the value safe to quote for ``security -i`` below.
_VALID = re.compile(r"^[A-Za-z0-9_\-.]{8,256}$")

_lock = threading.Lock()
_cache: dict[str, str | None] = {}


def supported() -> bool:
    return sys.platform == "darwin"


def valid(value: str) -> bool:
    return bool(_VALID.match(value.strip()))


def load(name: str) -> str | None:
    """The saved key for ``name``, or None. Cached: this runs per dictation."""
    if not supported():
        return None
    with _lock:
        if name in _cache:
            return _cache[name]
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", SERVICE, "-a", name, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        value = result.stdout.strip() if result.returncode == 0 else ""
    except Exception as exc:  # pragma: no cover - platform specific
        log.debug("could not read %s from the keychain: %s", name, exc)
        return None     # not cached: a transient failure should retry
    with _lock:
        _cache[name] = value or None
    return value or None


def save(name: str, value: str) -> None:
    """Store ``value`` for ``name`` in the login Keychain, replacing any old one.

    Raises ValueError for a malformed key, RuntimeError if the save fails.
    """
    if not supported():
        raise RuntimeError("saving keys is only supported on macOS")
    value = value.strip()
    if name not in KNOWN or not valid(value):
        raise ValueError("that does not look like an API key")
    # Through ``security -i`` on stdin rather than as an argument, so the key
    # never appears in the process list.
    command = f'add-generic-password -U -s {SERVICE} -a {name} -w "{value}"\n'
    result = subprocess.run(["security", "-i"], input=command,
                            capture_output=True, text=True, timeout=10)
    if result.returncode != 0 or "error" in result.stderr.lower():
        raise RuntimeError(result.stderr.strip() or "the keychain refused the key")
    with _lock:
        _cache[name] = value


def delete(name: str) -> None:
    if not supported():
        return
    subprocess.run(["security", "delete-generic-password", "-s", SERVICE, "-a", name],
                   capture_output=True, timeout=10)
    with _lock:
        _cache[name] = None
