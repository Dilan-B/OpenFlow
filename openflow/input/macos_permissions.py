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
"""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)

IS_MAC = sys.platform == "darwin"


def trusted() -> bool:
    """Accessibility granted and keys from other apps visible."""
    if not IS_MAC:
        return True
    try:
        import HIServices
        import Quartz

        return bool(HIServices.AXIsProcessTrusted()) and \
            bool(Quartz.CGPreflightListenEventAccess())
    except Exception as exc:  # pragma: no cover - platform specific
        log.debug("could not check permissions: %s", exc)
        return True     # unknown: do not nag


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
