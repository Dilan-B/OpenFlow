"""Tests for outbound address-family ordering.

The bug: on a network with dead IPv6 egress, getaddrinfo returns AAAA records
first and socket.create_connection applies its timeout to each address in
turn. A host with eight IPv6 addresses therefore burns eight full timeouts
before it ever reaches IPv4. Measured on the reporting machine: one urlopen
with a 6.6 s timeout took 53.0 s, and a dictation took 103 s against a 3 s
norm -- every slow figure in the log equalled IPv6-address-count x timeout.
"""

from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow import net

V6A = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::1", 443, 0, 0))
V6B = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::2", 443, 0, 0))
V4A = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))
V4B = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.2", 443))


class Ordering(unittest.TestCase):
    def test_ipv4_is_tried_before_ipv6(self):
        ordered = net.ipv4_first([V6A, V6B, V4A, V4B])
        self.assertEqual([info[0] for info in ordered],
                         [socket.AF_INET, socket.AF_INET, socket.AF_INET6, socket.AF_INET6])

    def test_ipv6_is_kept_as_a_fallback_not_dropped(self):
        """Dropping AAAA would strand an IPv6-only network. Order, don't remove."""
        ordered = net.ipv4_first([V6A, V4A])
        self.assertIn(V6A, ordered)
        self.assertEqual(len(ordered), 2)

    def test_order_within_a_family_is_preserved(self):
        """The resolver's own preference between equals still means something."""
        self.assertEqual(net.ipv4_first([V4A, V4B, V6A]), [V4A, V4B, V6A])

    def test_ipv6_only_results_are_left_alone(self):
        self.assertEqual(net.ipv4_first([V6A, V6B]), [V6A, V6B])


class Verdict(unittest.TestCase):
    """The verdict is re-derived on a TTL. A once-at-startup answer would have
    missed the real incident: the 103 s dictation happened nineteen minutes into
    a session that began on a working network."""

    def setUp(self):
        self.addCleanup(net.uninstall)
        self.calls = 0

    def _probe(self, *results):
        answers = list(results)

        def probe():
            self.calls += 1
            return answers[min(self.calls, len(answers)) - 1]

        return probe

    def test_verdict_is_cached_within_the_ttl(self):
        net.install(probe=self._probe(True))
        net.ipv6_usable(now=1000.0)
        net.ipv6_usable(now=1000.0 + net.HEALTHY_TTL_S - 1)
        self.assertEqual(self.calls, 1)

    def test_breakage_is_noticed_once_the_healthy_ttl_expires(self):
        """The whole point: IPv6 working at startup must not be believed forever."""
        net.install(probe=self._probe(True, False))
        self.assertTrue(net.ipv6_usable(now=1000.0))
        self.assertFalse(net.ipv6_usable(now=1000.0 + net.HEALTHY_TTL_S + 1))

    def test_a_broken_verdict_is_held_longer_than_a_healthy_one(self):
        """Probing while broken costs a full timeout, and IPv4 is already safe."""
        self.assertGreater(net.BROKEN_TTL_S, net.HEALTHY_TTL_S)
        net.install(probe=self._probe(False, True))
        self.assertFalse(net.ipv6_usable(now=1000.0))
        self.assertFalse(net.ipv6_usable(now=1000.0 + net.HEALTHY_TTL_S + 1))
        self.assertTrue(net.ipv6_usable(now=1000.0 + net.BROKEN_TTL_S + 1))

    def test_a_probe_that_raises_is_treated_as_broken(self):
        def boom():
            raise OSError("no route to host")

        net.install(probe=boom)
        self.assertFalse(net.ipv6_usable(now=1000.0))


class Resolution(unittest.TestCase):
    def setUp(self):
        self.addCleanup(net.uninstall)

    def test_broken_ipv6_reorders_lookups(self):
        net.install(probe=lambda: False)
        families = [info[0] for info in socket.getaddrinfo("localhost", 80)]
        self.assertEqual(families, sorted(families, key=lambda f: f != socket.AF_INET))

    def test_working_ipv6_leaves_the_order_alone(self):
        net.install(probe=lambda: True)
        expected = [info[0] for info in net._original_getaddrinfo("localhost", 80)]
        self.assertEqual([info[0] for info in socket.getaddrinfo("localhost", 80)], expected)

    def test_installing_twice_does_not_stack_wrappers(self):
        net.install(probe=lambda: False)
        wrapped = socket.getaddrinfo
        net.install(probe=lambda: False)
        self.assertIs(socket.getaddrinfo, wrapped)

    def test_uninstall_restores_the_real_resolver(self):
        original = socket.getaddrinfo
        net.install(probe=lambda: False)
        self.assertIsNot(socket.getaddrinfo, original)
        net.uninstall()
        self.assertIs(socket.getaddrinfo, original)


if __name__ == "__main__":
    unittest.main()
