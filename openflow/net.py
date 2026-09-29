"""Address-family ordering for outbound calls.

A network with dead IPv6 egress does not make cloud calls fail, it makes them
pathologically slow. getaddrinfo returns AAAA records first, and
socket.create_connection applies its timeout to each address in turn, so a host
with eight IPv6 addresses burns eight full timeouts before it ever tries IPv4.
Every timeout in the app is then multiplied by the AAAA count: a 6.6 s cleanup
budget became 53 s of wall time, and one dictation took 103 s against a 3 s
norm, with the logged figures landing on exactly address-count x timeout.

Two things shape this module.

The check cannot use DNS. A broken-IPv6 network resolves names perfectly well,
so anything name-based reports health and hides the fault; the probe dials a
literal address instead.

The check cannot run only at startup. IPv6 here comes and goes -- the dictation
that took 103 s was made nineteen minutes into a session that began on a
working network. A once-at-boot answer would have been "fine" and stayed wrong
for the rest of the session, so the answer is re-derived on a TTL: briefly when
IPv6 looks healthy, so breakage surfaces within a minute, and for longer once it
is known bad, because that is the case where probing itself costs a timeout and
where IPv4 is already carrying the traffic safely.

The remedy is ordering, not disabling. IPv4 goes first and IPv6 stays behind it,
so an IPv6-only or NAT64 network still connects -- IPv6 merely stops being the
default where it cannot work.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

# A literal address, so the probe never depends on DNS. Google public DNS
# answers on TCP/53.
PROBE_ADDRESS = ("2001:4860:4860::8888", 53)

# Long enough for a real network to answer, short enough to be unnoticeable.
PROBE_TIMEOUT_S = 1.5

# Healthy: re-check often, since this is the state that can silently rot and a
# probe costs milliseconds. Broken: re-check rarely, since the probe costs a
# full timeout and IPv4 is already serving every request correctly.
HEALTHY_TTL_S = 60.0
BROKEN_TTL_S = 300.0

_lock = threading.Lock()
_original_getaddrinfo: Callable | None = None
_probe: Callable[[], bool] = lambda: True
_usable: bool | None = None
_checked_at = 0.0
_announced = False


def ipv4_first(infos: list) -> list:
    """getaddrinfo results with IPv4 ahead of IPv6, order preserved within each
    family -- the resolver's preference between equals still means something."""
    return sorted(infos, key=lambda info: info[0] != socket.AF_INET)


def ipv6_egress_works(timeout_s: float = PROBE_TIMEOUT_S) -> bool:
    """Whether an IPv6 connection can actually leave this machine. Holding an
    IPv6 address is not the question; reaching anything with it is."""
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


def ipv6_usable(now: float | None = None) -> bool:
    """The cached verdict, re-probed when its TTL has run out."""
    global _usable, _checked_at, _announced

    now = time.monotonic() if now is None else now
    with _lock:
        ttl = HEALTHY_TTL_S if _usable else BROKEN_TTL_S
        if _usable is not None and now - _checked_at < ttl:
            return _usable

        try:
            verdict = _probe()
        except Exception as exc:
            # A probe that cannot run tells us as much as one that fails.
            log.debug("IPv6 probe failed to run (%s); assuming unusable", exc)
            verdict = False

        changed = verdict != _usable
        _usable, _checked_at = verdict, now

        if not verdict and (changed or not _announced):
            _announced = True
            log.warning(
                "IPv6 cannot reach the network from this machine; preferring "
                "IPv4 for cloud calls. Left alone this makes every request wait "
                "out one timeout per IPv6 address before falling back."
            )
        elif verdict and changed:
            log.info("IPv6 egress is working again; leaving resolution order alone")
        return verdict


def install(probe: Callable[[], bool] = ipv6_egress_works) -> None:
    """Route name resolution through the ordering check for this process. Safe
    to call twice; the verdict is decided per lookup, not here."""
    global _original_getaddrinfo, _probe, _usable, _checked_at, _announced

    _probe = probe
    if _original_getaddrinfo is not None:
        _usable, _checked_at, _announced = None, 0.0, False
        return

    _original_getaddrinfo = socket.getaddrinfo
    resolve = _original_getaddrinfo
    _usable, _checked_at, _announced = None, 0.0, False

    def getaddrinfo(*args, **kwargs):
        infos = resolve(*args, **kwargs)
        if len(infos) < 2 or ipv6_usable():
            return infos
        return ipv4_first(infos)

    socket.getaddrinfo = getaddrinfo


def uninstall() -> None:
    """Restore the real resolver. For tests, and for symmetry."""
    global _original_getaddrinfo, _usable, _checked_at, _announced
    if _original_getaddrinfo is not None:
        socket.getaddrinfo = _original_getaddrinfo
        _original_getaddrinfo = None
    _usable, _checked_at, _announced = None, 0.0, False
