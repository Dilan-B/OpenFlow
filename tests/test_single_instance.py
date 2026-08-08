"""Tests for the single-instance guard.

Two running copies each bind the global hotkey, so one keypress starts two
recordings that race to paste into the same cursor. These tests pin the
handoff: the second launch must decline the lock and ask the first to show.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from PySide6.QtWidgets import QApplication

    from openflow.single_instance import SingleInstance

    QT_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on install extras
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, "PySide6 not installed")
class SingleInstanceLock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        # A name per test, so a leftover socket cannot leak between them.
        self.name = f"OpenFlow.Test.{self.id().rsplit('.', 1)[-1]}"
        self.held: list[SingleInstance] = []

    def tearDown(self):
        for instance in self.held:
            instance.release()

    def _instance(self) -> SingleInstance:
        instance = SingleInstance(self.name)
        self.held.append(instance)
        return instance

    def _pump(self, seconds: float = 0.3) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)

    def test_first_caller_becomes_primary(self):
        self.assertTrue(self._instance().acquire())

    def test_second_caller_is_refused(self):
        self.assertTrue(self._instance().acquire())
        self.assertFalse(self._instance().acquire())

    def test_second_caller_asks_the_first_to_show(self):
        primary = self._instance()
        self.assertTrue(primary.acquire())
        shown: list[int] = []
        primary.activate_requested.connect(lambda: shown.append(1))

        self.assertFalse(self._instance().acquire())
        self._pump()
        self.assertTrue(shown, "the running instance was never asked to surface")

    def test_lock_is_reusable_after_release(self):
        first = self._instance()
        self.assertTrue(first.acquire())
        first.release()
        self.assertTrue(self._instance().acquire(),
                        "a clean exit must free the lock for the next launch")

    def test_stale_name_does_not_block_a_restart(self):
        # Simulates a killed process: the socket name lingers with nothing
        # listening. The next launch must still be able to claim it.
        from PySide6.QtNetwork import QLocalServer

        orphan = QLocalServer()
        orphan.listen(self.name)
        orphan.close()          # name may persist, no listener behind it

        self.assertTrue(self._instance().acquire())


if __name__ == "__main__":
    unittest.main(verbosity=2)
