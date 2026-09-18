"""The macOS permission helpers must be harmless everywhere else."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.input import macos_permissions  # noqa: E402


class OffMac(unittest.TestCase):
    def test_trusted_and_request_do_nothing(self):
        with mock.patch.object(macos_permissions, "IS_MAC", False), \
                mock.patch.dict("sys.modules", {"HIServices": None, "Quartz": None}):
            self.assertTrue(macos_permissions.trusted())
            macos_permissions.request()     # must not raise or import anything


class OnMac(unittest.TestCase):
    def _fakes(self, ax: bool, listen: bool):
        hi = mock.Mock(AXIsProcessTrusted=mock.Mock(return_value=ax),
                       kAXTrustedCheckOptionPrompt="prompt")
        qz = mock.Mock(CGPreflightListenEventAccess=mock.Mock(return_value=listen))
        return hi, qz

    def test_asks_only_for_what_is_missing(self):
        hi, qz = self._fakes(ax=False, listen=True)
        with mock.patch.object(macos_permissions, "IS_MAC", True), \
                mock.patch.dict("sys.modules", {"HIServices": hi, "Quartz": qz}):
            self.assertFalse(macos_permissions.trusted())
            macos_permissions.request()
        hi.AXIsProcessTrustedWithOptions.assert_called_once_with({"prompt": True})
        qz.CGRequestListenEventAccess.assert_not_called()

    def test_trusted_needs_both(self):
        for ax, listen, expected in ((True, True, True), (True, False, False),
                                     (False, True, False)):
            hi, qz = self._fakes(ax, listen)
            with mock.patch.object(macos_permissions, "IS_MAC", True), \
                    mock.patch.dict("sys.modules", {"HIServices": hi, "Quartz": qz}):
                self.assertEqual(macos_permissions.trusted(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
