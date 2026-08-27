"""Tests for launcher/startup-entry generation.

The bug these pin: in a PyInstaller build ``sys.executable`` is OpenFlow.exe,
not an interpreter, and ``__file__`` points inside _internal/. Resolving the
launch target without accounting for that produced a Startup shortcut reading
``OpenFlow.exe -m openflow --minimized`` -- argparse rejected ``-m``, the app
exited 2 at sign-in, and "start with Windows" looked like a crash.
"""

from __future__ import annotations

import shlex
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow import shortcuts  # noqa: E402
from openflow.__main__ import main  # noqa: E402

FAKE_EXE = r"C:\Users\someone\AppData\Local\Programs\OpenFlow\OpenFlow.exe"


def _frozen():
    """Stand in for a running PyInstaller build."""
    return mock.patch.multiple(shortcuts, FROZEN=True), mock.patch.object(
        sys, "executable", FAKE_EXE
    )


class FrozenLaunchTarget(unittest.TestCase):
    def setUp(self):
        for patcher in _frozen():
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_target_is_the_exe_itself(self):
        target, arguments = shortcuts.app_target()
        self.assertEqual(target, Path(FAKE_EXE))
        self.assertEqual(arguments, "", "a frozen exe takes no interpreter flags")

    def test_no_interpreter_flags_leak_into_the_shortcut(self):
        _, arguments = shortcuts.app_target()
        self.assertNotIn("-m", arguments)
        self.assertNotIn("openflow", arguments)

    def test_startup_arguments_are_accepted_by_the_cli(self):
        # This is the regression proper: whatever set_launch_at_login puts in
        # the .lnk has to survive the app's own argument parser.
        _, arguments = shortcuts.app_target()
        argv = shlex.split((arguments + " --minimized").strip())
        self.assertEqual(argv, ["--minimized"])

        parser_exit = None
        with mock.patch("openflow.app.OpenFlowApp") as app:
            app.return_value.run.return_value = 0
            try:
                main(argv)
            except SystemExit as exc:      # argparse's exit(2) on a bad flag
                parser_exit = exc.code
        self.assertIsNone(parser_exit,
                          f"startup arguments {argv} were rejected by the CLI")

    def test_workdir_is_the_install_directory(self):
        self.assertEqual(shortcuts.app_workdir(), Path(FAKE_EXE).parent)
        self.assertNotIn("_internal", str(shortcuts.app_workdir()))

    def test_icon_comes_from_the_exe_resource(self):
        # Nothing should be generated next to a read-only install.
        self.assertEqual(shortcuts.app_icon(), Path(FAKE_EXE))


class SourceCheckoutLaunchTarget(unittest.TestCase):
    def test_falls_back_to_the_module(self):
        with mock.patch.multiple(shortcuts, FROZEN=False), \
                mock.patch.object(shortcuts, "PROJECT_ROOT", Path(r"C:\nope")):
            target, arguments = shortcuts.app_target()
        self.assertEqual(arguments, "-m openflow")
        self.assertEqual(target, shortcuts.pythonw())


class WindowedStdio(unittest.TestCase):
    """A build with no console must still be able to write."""

    def test_none_handles_are_replaced(self):
        from openflow.__main__ import _ensure_stdio

        with mock.patch.object(sys, "stdout", None), \
                mock.patch.object(sys, "stderr", None):
            windowed = _ensure_stdio()
            self.assertTrue(windowed)
            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)
            sys.stdout.write("must not raise")
            sys.stderr.write("must not raise")

    def test_a_real_console_is_left_alone(self):
        from openflow.__main__ import _ensure_stdio

        self.assertFalse(_ensure_stdio())


if __name__ == "__main__":
    unittest.main(verbosity=2)
