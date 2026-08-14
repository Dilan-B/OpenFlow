"""Check whether a newer release exists.

A fix only reaches people who know there is one. Today an installed copy has
no way to find out -- the "Start with Windows" crash sat in the published
installer with no channel to tell anyone it had been fixed.

Deliberately a *check*, not an auto-installer. Silently replacing a binary that
holds a global keyboard hook and the microphone is not something to do behind
someone's back; this surfaces a version and a link, and the user decides.

One anonymous GET to api.github.com, no query parameters, nothing about the
user in it. Off is one setting away.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import __version__

log = logging.getLogger(__name__)

RELEASES_API = "https://api.github.com/repos/Dilan-B/OpenFlow/releases/latest"
RELEASES_PAGE = "https://github.com/Dilan-B/OpenFlow/releases/latest"
TIMEOUT_S = 6.0
# GitHub rejects urllib's default agent on some paths, the same way Groq does
# (see docs/research-2026-08.md) -- send something real.
USER_AGENT = f"OpenFlow/{__version__} (+{RELEASES_PAGE})"

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


@dataclass(slots=True)
class Release:
    version: str
    url: str
    notes: str = ""


def parse_version(raw: str) -> tuple[int, int, int] | None:
    match = _VERSION_RE.match((raw or "").strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def is_newer(candidate: str, current: str = __version__) -> bool:
    """True when ``candidate`` is a strictly higher x.y.z than ``current``.

    Unparseable input is never newer: a malformed tag upstream should not
    nag every user forever.
    """
    left, right = parse_version(candidate), parse_version(current)
    if left is None or right is None:
        return False
    return left > right


def check(timeout: float = TIMEOUT_S) -> Release | None:
    """Return the latest release when it is newer than this build, else None.

    Never raises. Being offline, rate-limited, or behind a proxy that mangles
    the response are all ordinary conditions, not errors worth surfacing.
    """
    request = urllib.request.Request(
        RELEASES_API,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        log.debug("update check skipped: %s", exc)
        return None

    tag = str(payload.get("tag_name") or "")
    if not is_newer(tag):
        return None
    return Release(
        version=tag.lstrip("v"),
        url=str(payload.get("html_url") or RELEASES_PAGE),
        notes=str(payload.get("body") or "")[:2000],
    )


def check_if_due(config, now: float | None = None) -> Release | None:
    """Rate-limited wrapper for startup. Honours ``config.updates``."""
    settings = getattr(config, "updates", None)
    if settings is None or not settings.check_on_startup:
        return None

    now = now if now is not None else time.time()
    interval = max(1, settings.interval_hours) * 3600
    if now - settings.last_checked_at < interval:
        return None

    settings.last_checked_at = now
    try:
        config.save()
    except Exception as exc:
        log.debug("could not persist update-check timestamp: %s", exc)
    return check()
