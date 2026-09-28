"""Address-family ordering for outbound calls.

A network with dead IPv6 egress does not make cloud calls fail, it makes them
pathologically slow. getaddrinfo returns AAAA records first, and
socket.create_connection applies its timeout to each address in turn, so a host
with eight IPv6 addresses burns eight full timeouts before it ever tries IPv4.
Every timeout in the app is then multiplied by the AAAA count: a 6.6 s cleanup
budget became 53 s of wall time, and one dictation took 103 s against a 3 s
norm, with the log reporting times that were exactly address-count x timeout.

The fix is ordering, not disabling. IPv4 goes first and IPv6 stays behind it,
so a genuinely IPv6-only network still connects -- it just stops being the
default on a network where it cannot work.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable

log = logging.getLogger(__name__)

# A literal address, so the probe never depends on DNS -- which resolves fine
# on a broken-IPv6 network and would hide the fault. Google public DNS answers
# on TCP/53.
PROBE_ADDRESS = ("2001:4860:4860::8888", 53)

# Long enough for a real network to answer, short enough that a dead one costs
# nothing noticeable at startup.
PROBE_TIMEOUT_S = 1.5

_original_getaddrinfo: Callable | None = None


def ipv4_first(infos: list) -> list:
    """getaddrinfo results with IPv4 ahead of IPv6, order preserved within each
    family -- the resolver's preference between equals still means something."""
    return sorted(infos, key=lambda info: info[0] != socket.AF_INET)


def ipv6_egress_works(timeout_s: float = PROBE_TIMEOUT_S) -> bool:
    """Whether an IPv6 connection can actually leave this machine. Having an
    IPv6 address is not the question; being able to reach anything with it is."""
    if not socket.has_ipv6:
        return False
    sock = None
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.settimeout(timeout_s)
        sock.connect(PROBE_ADDRESS)
        return True
    except OSError:
        return False
    finally:
        if sock is not None:
            sock.close()


def prefer_ipv4_if_broken(probe: Callable[[], bool] = ipv6_egress_works) -> bool:
    """Probe IPv6 egress once and, if it is unusable, order IPv4 first for the
    rest of the process. Returns whether the reordering was installed."""
    global _original_getaddrinfo

    try:
        usable = probe()
    except Exception as exc:
        # A probe that cannot even run tells us as much as one that fails.
        log.debug("IPv6 probe failed to run (%s); assuming unusable", exc)
        usable = False

    if usable:
        return _original_getaddrinfo is not None

    if _original_getaddrinfo is not None:
        return True

    _original_getaddrinfo = socket.getaddrinfo
    resolve = _original_getaddrinfo

    def getaddrinfo(*args, **kwargs):
        return ipv4_first(resolve(*args, **kwargs))

    socket.getaddrinfo = getaddrinfo
    log.warning(
        "IPv6 cannot reach the network from this machine; preferring IPv4 for "
        "cloud calls. Left alone this makes every request wait out one timeout "
        "per IPv6 address before falling back."
    )
    return True


def uninstall() -> None:
    """Restore the real resolver. Exists for tests and for a network that is
    repaired without restarting the app."""
    global _original_getaddrinfo
    if _original_getaddrinfo is not None:
        socket.getaddrinfo = _original_getaddrinfo
        _original_getaddrinfo = None
