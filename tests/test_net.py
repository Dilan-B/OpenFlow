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


class Install(unittest.TestCase):
    def setUp(self):
        self.addCleanup(net.uninstall)

    def test_broken_ipv6_installs_the_reordering(self):
        self.assertTrue(net.prefer_ipv4_if_broken(probe=lambda: False))
        infos = socket.getaddrinfo("localhost", 80)
        families = [info[0] for info in infos]
        self.assertEqual(families, sorted(families, key=lambda f: f != socket.AF_INET))

    def test_working_ipv6_is_left_untouched(self):
        original = socket.getaddrinfo
        self.assertFalse(net.prefer_ipv4_if_broken(probe=lambda: True))
        self.assertIs(socket.getaddrinfo, original)

    def test_installing_twice_does_not_stack_wrappers(self):
        net.prefer_ipv4_if_broken(probe=lambda: False)
        wrapped = socket.getaddrinfo
        net.prefer_ipv4_if_broken(probe=lambda: False)
        self.assertIs(socket.getaddrinfo, wrapped)

    def test_uninstall_restores_the_real_resolver(self):
        original = socket.getaddrinfo
        net.prefer_ipv4_if_broken(probe=lambda: False)
        self.assertIsNot(socket.getaddrinfo, original)
        net.uninstall()
        self.assertIs(socket.getaddrinfo, original)

    def test_a_probe_that_raises_is_treated_as_broken(self):
        """A probe blowing up must not take the app down with it."""
        def boom():
            raise OSError("no route to host")
        self.assertTrue(net.prefer_ipv4_if_broken(probe=boom))


if __name__ == "__main__":
    unittest.main()
