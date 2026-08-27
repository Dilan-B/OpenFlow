"""Per-application dictation profiles.

The right output depends on where it lands. A terminal wants a bare command
with no capital letter and no full stop -- "Ship it." is a syntax error. A code
editor wants no smart quotes, because a curly quote in source is a bug that
takes ten minutes to spot. A chat window wants neither trailing punctuation nor
a capital, because that is how people write in chat.

Profiles are matched against the foreground process name at the moment the
hotkey is released, which is the window the text is about to go into.

Everything here is data. The matching is a plain exact-name lookup: substring
matching would have "code.exe" catch "vscode.exe" and anything else with those
letters in it, and a dictation tool guessing wrong is worse than not guessing.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Profile:
    name: str
    # Add a full stop and capitalize the first word.
    terminal_punctuation: bool = True
    capitalize: bool = True
    # Straight quotes and hyphens only. Curly punctuation in code or a shell
    # is either a syntax error or an invisible bug.
    plain_punctuation: bool = False
    description: str = ""


DEFAULT = Profile(name="default", description="Ordinary prose")

PROFILES: dict[str, Profile] = {
    "prose": DEFAULT,
    "code": Profile(
        name="code", terminal_punctuation=False, capitalize=False,
        plain_punctuation=True,
        description="No trailing full stop, no capital, straight quotes",
    ),
    "shell": Profile(
        name="shell", terminal_punctuation=False, capitalize=False,
        plain_punctuation=True,
        description="Bare commands -- a full stop would be a syntax error",
    ),
    "chat": Profile(
        name="chat", terminal_punctuation=False, capitalize=True,
        description="No trailing full stop, the way people write in chat",
    ),
}

# Foreground executable -> profile. Lowercased, matched exactly.
APP_PROFILES: dict[str, str] = {
    # Terminals
    "windowsterminal.exe": "shell",
    "cmd.exe": "shell",
    "powershell.exe": "shell",
    "pwsh.exe": "shell",
    "conemu64.exe": "shell",
    "alacritty.exe": "shell",
    "wezterm-gui.exe": "shell",
    "terminal": "shell",              # macOS
    "iterm2": "shell",
    "gnome-terminal": "shell",
    # Editors and IDEs
    "code.exe": "code",
    "code - insiders.exe": "code",
    "cursor.exe": "code",
    "devenv.exe": "code",
    "idea64.exe": "code",
    "pycharm64.exe": "code",
    "webstorm64.exe": "code",
    "rider64.exe": "code",
    "clion64.exe": "code",
    "sublime_text.exe": "code",
    "zed.exe": "code",
    "notepad++.exe": "code",
    # Chat
    "slack.exe": "chat",
    "discord.exe": "chat",
    "teams.exe": "chat",
    "ms-teams.exe": "chat",
    "telegram.exe": "chat",
    "whatsapp.exe": "chat",
    "signal.exe": "chat",
}


def foreground_process() -> str:
    """Lowercased executable name of the focused window, or "" if unknown.

    Never raises: this runs on the dictation path, and a failure to identify
    the window is a reason to use the default profile, not to lose the text.
    """
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return ""

        # PROCESS_QUERY_LIMITED_INFORMATION: enough for the image name, and
        # unlike PROCESS_QUERY_INFORMATION it works against elevated processes.
        handle = kernel32.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(260)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(
                    handle, 0, buffer, ctypes.byref(size)):
                return ""
            return buffer.value.rsplit("\\", 1)[-1].lower()
        finally:
            kernel32.CloseHandle(handle)
    except Exception as exc:
        log.debug("could not identify the foreground process: %s", exc)
        return ""


def profile_for(process: str, overrides: dict[str, str] | None = None) -> Profile:
    """Resolve a profile for an executable name. Unknown apps get prose."""
    if not process:
        return DEFAULT
    key = process.lower()
    name = (overrides or {}).get(key) or APP_PROFILES.get(key)
    if name is None:
        return DEFAULT
    profile = PROFILES.get(name)
    if profile is None:
        log.warning("unknown profile %r mapped for %s; using the default", name, key)
        return DEFAULT
    return profile


def apply_profile(text: str, profile: Profile) -> str:
    """Reshape finished text for where it is about to land."""
    if not text:
        return text
    out = text

    if profile.plain_punctuation:
        for curly, straight in (("’", "'"), ("‘", "'"),
                                ("“", '"'), ("”", '"'),
                                ("—", "--"), ("–", "-"),
                                ("…", "...")):
            out = out.replace(curly, straight)

    if not profile.terminal_punctuation:
        # Only a full stop. A question or exclamation carries meaning that the
        # speaker chose, even in a chat window.
        out = out.rstrip()
        if out.endswith(".") and not out.endswith(".."):
            out = out[:-1]

    if not profile.capitalize and out[:1].isupper():
        # Only touch the first character, and only when the rest of the word is
        # lowercase -- "API" and "I" must survive.
        first_word = out.split(" ", 1)[0].strip(".,;:!?")
        if first_word.isalpha() and first_word[1:].islower():
            out = out[0].lower() + out[1:]

    return out
