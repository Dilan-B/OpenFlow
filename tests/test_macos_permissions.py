"""The macOS permission helpers must be harmless everywhere else."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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


class ClearStale(unittest.TestCase):
    """A grant from an older build shows as on but does not apply; the new
    build clears its own entries once so its prompt adds one that does."""

    def _run(self, ax, listen, state=None, frozen=True):
        hi, qz = OnMac()._fakes(ax, listen)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "macos_permissions.json"
            if state is not None:
                path.write_text(json.dumps(state))
            with mock.patch.object(macos_permissions, "IS_MAC", True), \
                    mock.patch.object(sys, "frozen", frozen, create=True), \
                    mock.patch.dict("sys.modules", {"HIServices": hi, "Quartz": qz}), \
                    mock.patch.object(subprocess, "run") as run:
                cleared = macos_permissions.clear_stale("1.8.0", path)
                saved = json.loads(path.read_text()) if path.exists() else None
        return cleared, run, saved

    def test_resets_only_the_missing_service(self):
        cleared, run, saved = self._run(ax=True, listen=False)
        self.assertEqual(cleared, ["ListenEvent"])
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0],
                         ["tccutil", "reset", "ListenEvent", macos_permissions.BUNDLE_ID])
        self.assertEqual(saved, {"cleared_for": "1.8.0"})

    def test_once_per_version(self):
        cleared, run, _ = self._run(ax=False, listen=False,
                                    state={"cleared_for": "1.8.0"})
        self.assertEqual(cleared, [])
        run.assert_not_called()

    def test_new_version_clears_again(self):
        cleared, _, _ = self._run(ax=False, listen=False,
                                  state={"cleared_for": "1.7.0"})
        self.assertEqual(cleared, ["Accessibility", "ListenEvent"])

    def test_not_from_source(self):
        cleared, run, saved = self._run(ax=False, listen=False, frozen=False)
        self.assertEqual(cleared, [])
        run.assert_not_called()
        self.assertIsNone(saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
