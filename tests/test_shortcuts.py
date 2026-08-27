"""Tests for launcher targets.

The bug these pin: a Startup shortcut written before the app was ever packaged
pointed at ``pythonw.exe -m openflow``, so Windows listed OpenFlow as a nameless
idle Python process forever after -- app_target() never reconsidered once a
build existed, and inside a frozen build it fell through to handing OpenFlow.exe
a ``-m openflow`` it cannot parse.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow import shortcuts

INSTALL_VARS = ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)")


def _make_exe(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    exe = root / "OpenFlow.exe"
    exe.write_bytes(b"")
    return exe


class AppTarget(unittest.TestCase):
    """Every case runs with the real install roots hidden, so a build on the
    developer's own machine cannot decide the outcome."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        self._env = {var: os.environ.pop(var, None) for var in INSTALL_VARS}
        self.addCleanup(self._restore_env)

        self._frozen = getattr(sys, "frozen", None)
        self._executable = sys.executable
        self._project_root = shortcuts.PROJECT_ROOT
        self.addCleanup(self._restore_sys)

        # Default for every test: bare checkout, nothing built anywhere.
        if hasattr(sys, "frozen"):
            del sys.frozen
        shortcuts.PROJECT_ROOT = self.tmp / "checkout"

    def _restore_env(self):
        for var, value in self._env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

    def _restore_sys(self):
        if self._frozen is None:
            if hasattr(sys, "frozen"):
                del sys.frozen
        else:
            sys.frozen = self._frozen
        sys.executable = self._executable
        shortcuts.PROJECT_ROOT = self._project_root

    def test_frozen_app_launches_itself_with_no_arguments(self):
        exe = _make_exe(self.tmp / "installed")
        sys.frozen = True
        sys.executable = str(exe)

        target, arguments = shortcuts.app_target()

        self.assertEqual(target, exe.resolve())
        self.assertEqual(arguments, "")

    def test_installed_build_beats_source_checkout(self):
        exe = _make_exe(self.tmp / "local" / "Programs" / "OpenFlow")
        os.environ["LOCALAPPDATA"] = str(self.tmp / "local")

        target, arguments = shortcuts.app_target()

        self.assertEqual(target, exe)
        self.assertEqual(arguments, "")

    def test_local_pyinstaller_build_beats_source_checkout(self):
        exe = _make_exe(self.tmp / "checkout" / "dist" / "OpenFlow")

        target, arguments = shortcuts.app_target()

        self.assertEqual(target, exe)
        self.assertEqual(arguments, "")

    def test_bare_checkout_falls_back_to_the_module(self):
        target, arguments = shortcuts.app_target()

        self.assertEqual(arguments, "-m openflow")
        self.assertIn(target.name, {"pythonw.exe", Path(sys.executable).name})


class WorkingDirectory(unittest.TestCase):
    def test_exe_runs_from_its_own_directory(self):
        exe = Path("C:/Users/someone/AppData/Local/Programs/OpenFlow/OpenFlow.exe")
        self.assertEqual(shortcuts.app_workdir(exe), exe.parent)

    def test_module_form_runs_from_the_checkout(self):
        interpreter = Path("C:/Python314/pythonw.exe")
        self.assertEqual(shortcuts.app_workdir(interpreter), shortcuts.PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
