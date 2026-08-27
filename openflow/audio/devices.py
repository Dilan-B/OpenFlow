"""Input device enumeration.

PortAudio lists every host API's view of the same hardware, so a machine with
three microphones offers eighteen entries -- the same webcam four times over.
That is a fine API and a terrible dropdown, so we collapse them to one entry
per physical device.

Two wrinkles drive the shape of this module. MME truncates device names at 31
characters ('Microphone (2- Logitech Webcam ' with the model lost), so names
are matched on a truncated key and the longest spelling wins for display. And
MME is what ``sounddevice`` picks by default, while WASAPI is the native path
with better latency and sample rates -- so when the same device appears under
several APIs we hand back the best one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace

log = logging.getLogger(__name__)

# Best first. WDM-KS is deliberately absent: it is the kernel-streaming back
# door, and it re-lists every device plus a few Windows itself keeps disabled
# (Stereo Mix, Line In). Anything genuinely available to record from shows up
# under one of these three, so this list is also the filter.
_HOSTAPI_RANK = ("Windows WASAPI", "Windows DirectSound", "MME")

# MME's name field. Anything longer is cut, which is why names are keyed short.
_NAME_KEY_LEN = 30

# Aliases for "whatever Windows is set to", which the System default entry
# already covers. Listing them again invites picking the same device twice.
_PSEUDO = {
    "microsoft sound mapper - input",
    "primary sound capture driver",
}


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    hostapi: str
    channels: int
    samplerate: float


def _pretty(name: str) -> str:
    r"""Device names are driver strings, not labels. WDM-KS in particular hands
    back things like ``Headset (@System32\drivers\bthhfenum.sys,#2;%1
    Hands-Free%0\r\n;(WH-1000XM5))`` -- the useful part is the last
    parenthesised group. Keep the leading noun and that, drop the rest."""
    name = " ".join(name.split())
    if "@" in name or ".sys" in name:
        groups = re.findall(r"\(([^()]+)\)", name)
        friendly = groups[-1].strip() if groups else ""
        head = name.split("(", 1)[0].strip()
        if friendly and head:
            return f"{head} ({friendly})"
        if friendly:
            return friendly
    return re.sub(r"\s*\(\s*\)$", "", name)


def _rank(hostapi: str) -> int:
    try:
        return _HOSTAPI_RANK.index(hostapi)
    except ValueError:
        return len(_HOSTAPI_RANK)


def list_input_devices() -> list[InputDevice]:
    """One entry per physical microphone, best host API each, sorted by name."""
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - depends on install extras
        log.warning("cannot list input devices: %s", exc)
        return []

    chosen: dict[str, InputDevice] = {}
    longest: dict[str, str] = {}
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        name = str(dev["name"]).strip()
        if not name or name.lower() in _PSEUDO:
            continue

        hostapi = str(sd.query_hostapis(dev["hostapi"])["name"])
        if hostapi not in _HOSTAPI_RANK:
            continue

        key = name.lower()[:_NAME_KEY_LEN]
        if len(name) > len(longest.get(key, "")):
            longest[key] = name

        current = chosen.get(key)
        if current is None or _rank(hostapi) < _rank(current.hostapi):
            chosen[key] = InputDevice(
                index=index,
                name=name,
                hostapi=hostapi,
                channels=int(dev["max_input_channels"]),
                samplerate=float(dev["default_samplerate"]),
            )

    devices = [replace(dev, name=_pretty(longest[key])) for key, dev in chosen.items()]
    return sorted(devices, key=lambda d: d.name.lower())


def default_input_index() -> int | None:
    """What PortAudio uses when input_device is None."""
    try:
        import sounddevice as sd

        index = sd.default.device[0]
    except Exception:  # pragma: no cover - depends on host audio state
        return None
    return index if isinstance(index, int) and index >= 0 else None


def describe(index: int | None) -> str:
    """Human-readable name for a configured index, for logs and the UI."""
    if index is None:
        default = default_input_index()
        return f"System default ({describe(default)})" if default is not None             else "System default"
    try:
        import sounddevice as sd

        raw = str(sd.query_devices(index)["name"]).strip()
    except Exception:
        return f"device {index} (unavailable)"

    # Prefer the deduped spelling: the raw one may be an MME truncation.
    key = raw.lower()[:_NAME_KEY_LEN]
    for device in list_input_devices():
        if device.name.lower()[:_NAME_KEY_LEN] == key:
            return device.name
    return _pretty(raw)
