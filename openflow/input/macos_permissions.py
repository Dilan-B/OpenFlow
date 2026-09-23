"""Ask macOS for the two permissions a global dictation hotkey needs.

Without Accessibility and Input Monitoring, pynput's event tap only sees keys
pressed while OpenFlow itself is focused -- the app looks fine and simply
ignores the hotkey everywhere else. Checking quietly is not enough either:
recent macOS lists an app under Privacy & Security only once it has *asked*,
so an app that never asks cannot even be found to be switched on. And because
the build is ad-hoc signed, every update is a new app to macOS and loses its
grants.

So: ask with the system prompt when a grant is missing, and let the app notice
when it arrives (see OpenFlowApp._check_permissions) instead of needing a
restart. All no-ops off macOS.

A grant left over from an older build is the nasty case: Privacy & Security
shows OpenFlow switched on, but the switch belongs to the old signature, so
the new build is refused and nothing the user can see explains why. Before
asking, a new build therefore clears its own stale entries (clear_stale), once
per version, so the prompt adds a fresh switch that actually applies.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

IS_MAC = sys.platform == "darwin"
BUNDLE_ID = "io.github.dilan-b.openflow"
# tccutil's names for the two Privacy & Security lists.
ACCESSIBILITY = "Accessibility"
INPUT_MONITORING = "ListenEvent"


def trusted() -> bool:
    """Accessibility granted and keys from other apps visible."""
    return not missing()


def missing() -> list[str]:
    """The tccutil service names OpenFlow is not trusted for."""
    if not IS_MAC:
        return []
    try:
        import HIServices
        import Quartz

        out = []
        if not HIServices.AXIsProcessTrusted():
            out.append(ACCESSIBILITY)
        if not Quartz.CGPreflightListenEventAccess():
            out.append(INPUT_MONITORING)
        return out
    except Exception as exc:  # pragma: no cover - platform specific
        log.debug("could not check permissions: %s", exc)
        return []


def clear_stale(version: str, state_file: Path) -> list[str]:
    """Remove OpenFlow's entries for whatever is missing, once per version.

    Only the frozen app does this: from source, the trusted process is the
    terminal or Python, not the bundle. Resetting an entry the user never
    switched on is harmless; it just gets re-added when request() asks.
    Returns the services reset."""
    if not IS_MAC or not getattr(sys, "frozen", False):
        return []
    try:
        done = json.loads(state_file.read_text(encoding="utf-8")).get("cleared_for")
    except (OSError, ValueError, AttributeError):
        done = None
    if done == version:
        return []
    services = missing()
    for service in services:
        try:
            subprocess.run(["tccutil", "reset", service, BUNDLE_ID],
                           check=True, capture_output=True, timeout=10)
            log.info("cleared stale %s entry", service)
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("could not clear %s entry: %s", service, exc)
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({"cleared_for": version}), encoding="utf-8")
    except OSError as exc:
        log.debug("could not record permission reset: %s", exc)
    return services


def request() -> None:
    """Show macOS's own prompts for whatever is missing. Each prompt adds
    OpenFlow to its list in Privacy & Security and links straight there."""
    if not IS_MAC:
        return
    try:
        import HIServices
        import Quartz

        if not HIServices.AXIsProcessTrusted():
            HIServices.AXIsProcessTrustedWithOptions(
                {HIServices.kAXTrustedCheckOptionPrompt: True})
            log.info("asked for Accessibility permission")
        if not Quartz.CGPreflightListenEventAccess():
            Quartz.CGRequestListenEventAccess()
            log.info("asked for Input Monitoring permission")
    except Exception as exc:  # pragma: no cover - platform specific
        log.warning("could not request permissions: %s", exc)
