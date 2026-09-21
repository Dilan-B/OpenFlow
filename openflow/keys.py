"""Provider API keys saved from the Settings page.

The environment still wins -- GROQ_API_KEY and friends work as before. But a
Mac app started from the Dock never sees what a shell profile exports, so
there is also somewhere the app itself can keep a key: the macOS login
Keychain, through the ``security`` tool that ships with the OS, or on Windows
the Credential Manager, through advapi32.

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


def _backend() -> str | None:
    if sys.platform == "darwin":
        return "mac"
    if sys.platform == "win32":
        return "win"
    return None


def supported() -> bool:
    return _backend() is not None


def store_name() -> str:
    """What the Settings page calls the place keys are kept."""
    return "Credential Manager" if _backend() == "win" else "Keychain"


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
        if _backend() == "win":
            value = _win_read(name)
        else:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", SERVICE, "-a", name, "-w"],
                capture_output=True, text=True, timeout=5,
            )
            value = result.stdout.strip() if result.returncode == 0 else ""
    except Exception as exc:  # pragma: no cover - platform specific
        log.debug("could not read saved %s: %s", name, exc)
        return None     # not cached: a transient failure should retry
    with _lock:
        _cache[name] = value or None
    return value or None


def save(name: str, value: str) -> None:
    """Store ``value`` for ``name`` in the OS key store, replacing any old one.

    Raises ValueError for a malformed key, RuntimeError if the save fails.
    """
    if not supported():
        raise RuntimeError("saving keys is not supported on this platform")
    value = value.strip()
    if name not in KNOWN or not valid(value):
        raise ValueError("that does not look like an API key")
    if _backend() == "win":
        _win_write(name, value)
        with _lock:
            _cache[name] = value
        return
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
    if _backend() == "win":
        _win_delete(name)
    else:
        subprocess.run(["security", "delete-generic-password", "-s", SERVICE, "-a", name],
                       capture_output=True, timeout=10)
    with _lock:
        _cache[name] = None


# ------------------------------------------------------------------ Windows
# Generic credentials in the user's Credential Manager, named "OpenFlow/<KEY>".
# Protected by DPAPI under the user's login, like the macOS Keychain.

_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2
_ERROR_NOT_FOUND = 1168


def _win_target(name: str) -> str:
    return f"{SERVICE}/{name}"


def _win_api():
    import ctypes
    from ctypes import wintypes

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD),
                    ("dwHighDateTime", wintypes.DWORD)]

    class CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                 ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
    advapi.CredReadW.restype = wintypes.BOOL
    advapi.CredWriteW.argtypes = [ctypes.POINTER(CREDENTIALW), wintypes.DWORD]
    advapi.CredWriteW.restype = wintypes.BOOL
    advapi.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    advapi.CredDeleteW.restype = wintypes.BOOL
    advapi.CredFree.argtypes = [ctypes.c_void_p]
    advapi.CredFree.restype = None
    return ctypes, advapi, CREDENTIALW


def _win_read(name: str) -> str:
    ctypes, advapi, CREDENTIALW = _win_api()
    pcred = ctypes.POINTER(CREDENTIALW)()
    if not advapi.CredReadW(_win_target(name), _CRED_TYPE_GENERIC, 0, ctypes.byref(pcred)):
        err = ctypes.get_last_error()
        if err == _ERROR_NOT_FOUND:
            return ""
        raise OSError(err, "CredReadW failed")
    try:
        cred = pcred.contents
        blob = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
    finally:
        advapi.CredFree(pcred)
    return blob.decode("utf-8", errors="replace").strip()


def _win_write(name: str, value: str) -> None:
    ctypes, advapi, CREDENTIALW = _win_api()
    data = value.encode("utf-8")
    blob = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    cred = CREDENTIALW()
    cred.Type = _CRED_TYPE_GENERIC
    cred.TargetName = _win_target(name)
    cred.CredentialBlobSize = len(data)
    cred.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
    cred.Persist = _CRED_PERSIST_LOCAL_MACHINE
    cred.UserName = name
    if not advapi.CredWriteW(ctypes.byref(cred), 0):
        raise RuntimeError(f"Credential Manager refused the key ({ctypes.get_last_error()})")


def _win_delete(name: str) -> None:
    ctypes, advapi, _ = _win_api()
    if not advapi.CredDeleteW(_win_target(name), _CRED_TYPE_GENERIC, 0):
        err = ctypes.get_last_error()
        if err != _ERROR_NOT_FOUND:
            log.debug("could not delete saved %s: %s", name, err)
