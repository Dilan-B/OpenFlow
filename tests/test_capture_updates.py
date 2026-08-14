"""Tests for eval capture and the update check.

Capture writes a record of what somebody said, so the tests that matter most
are the ones proving it stays off until asked and goes away when unasked.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openflow.capture as capture_module  # noqa: E402
from openflow.capture import Capture, _split_for  # noqa: E402
from openflow.config import Config  # noqa: E402
from openflow.updates import Release, is_newer, parse_version  # noqa: E402


class CaptureBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        patches = [
            mock.patch.object(capture_module, "CONFIG_DIR", root),
            mock.patch.object(capture_module, "EVENTS_PATH", root / "capture" / "events.jsonl"),
            mock.patch.object(capture_module, "AUDIO_DIR", root / "capture" / "audio"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.config = Config()

    def dictate(self, cap: Capture, raw="um the deploy", cleaned="The deploy", **kw):
        return cap.record(
            raw=raw, cleaned=cleaned, stt_engine="parakeet_onnx", clean_engine="rules",
            duration_s=2.0, rms=0.05, latency_ms=310.0,
            strategies=kw.pop("strategies", []), fillers_removed=["um"],
            repetitions_collapsed=[], uncertain=kw.pop("uncertain", False), **kw)

    def events(self) -> list[dict]:
        path = capture_module.EVENTS_PATH
        if not path.exists():
            return []
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]


class OffByDefault(CaptureBase):
    def test_nothing_is_written_when_disabled(self):
        cap = Capture(self.config)
        self.assertFalse(cap.enabled)
        self.assertIsNone(self.dictate(cap))
        self.assertFalse(capture_module.EVENTS_PATH.exists(),
                         "capture must not create files until it is switched on")

    def test_reject_is_also_a_no_op_when_disabled(self):
        Capture(self.config).reject("whatever")
        self.assertFalse(capture_module.EVENTS_PATH.exists())


class Recording(CaptureBase):
    def setUp(self):
        super().setUp()
        self.config.capture.enabled = True
        self.cap = Capture(self.config)

    def test_a_dictation_is_appended(self):
        record_id = self.dictate(self.cap)
        self.assertIsNotNone(record_id)
        events = self.events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["raw"], "um the deploy")
        self.assertEqual(events[0]["cleaned"], "The deploy")
        self.assertFalse(events[0]["rejected"])

    def test_audio_is_a_separate_opt_in(self):
        self.dictate(self.cap, audio=[0.0, 0.1, -0.1])
        self.assertEqual(self.events()[0]["audio_path"], "",
                         "audio must not be written just because capture is on")

    def test_audio_is_written_once_opted_in(self):
        import numpy as np

        self.config.capture.audio = True
        self.dictate(self.cap, audio=np.zeros(1600, dtype="float32"), sample_rate=16000)
        path = self.events()[0]["audio_path"]
        self.assertTrue(path)
        self.assertTrue((capture_module.CONFIG_DIR / path).exists())

    def test_undo_marks_the_record_rejected(self):
        record_id = self.dictate(self.cap)
        self.cap.reject(record_id)
        self.assertTrue(self.events()[0]["rejected"])

    def test_reject_without_an_id_marks_the_most_recent(self):
        self.dictate(self.cap, raw="first")
        self.dictate(self.cap, raw="second")
        self.cap.reject()
        events = self.events()
        self.assertFalse(events[0]["rejected"])
        self.assertTrue(events[1]["rejected"])

    def test_rejecting_does_not_add_a_second_record(self):
        self.cap.reject(self.dictate(self.cap))
        self.assertEqual(len(self.events()), 1)

    def test_a_write_failure_never_costs_the_dictation(self):
        with mock.patch("pathlib.Path.open", side_effect=OSError("disk full")):
            self.assertIsNone(self.dictate(self.cap))   # returns, does not raise

    def test_purge_removes_everything(self):
        import numpy as np

        self.config.capture.audio = True
        self.dictate(self.cap, audio=np.zeros(160, dtype="float32"))
        self.assertTrue(capture_module.EVENTS_PATH.exists())
        Capture.purge()
        self.assertFalse(capture_module.EVENTS_PATH.exists())
        self.assertEqual(list(capture_module.AUDIO_DIR.glob("*.wav")), [])

    def test_stats_counts_what_is_held(self):
        self.cap.reject(self.dictate(self.cap, raw="a"))
        self.dictate(self.cap, raw="b")
        stats = Capture.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["rejected"], 1)


class Splits(unittest.TestCase):
    def test_a_record_lands_in_the_same_split_forever(self):
        self.assertEqual(_split_for("abc123"), _split_for("abc123"))

    def test_both_splits_are_reachable(self):
        splits = {_split_for(f"id-{i}") for i in range(200)}
        self.assertEqual(splits, {"train", "eval"})

    def test_eval_is_roughly_the_configured_share(self):
        held = sum(_split_for(f"id-{i}") == "eval" for i in range(2000))
        self.assertTrue(0.15 < held / 2000 < 0.25, f"eval share was {held / 2000:.2f}")


class Versions(unittest.TestCase):
    def test_parses_with_and_without_a_v(self):
        self.assertEqual(parse_version("v1.2.3"), (1, 2, 3))
        self.assertEqual(parse_version("1.2.3"), (1, 2, 3))

    def test_newer_and_older(self):
        self.assertTrue(is_newer("v1.2.0", "1.1.9"))
        self.assertTrue(is_newer("v2.0.0", "1.9.9"))
        self.assertFalse(is_newer("v1.0.0", "1.0.0"))
        self.assertFalse(is_newer("v0.9.9", "1.0.0"))

    def test_garbage_is_never_newer(self):
        # A malformed upstream tag must not nag every user forever.
        for junk in ("", "latest", "nightly", "v", None):
            self.assertFalse(is_newer(junk, "1.0.0"), junk)

    def test_release_carries_a_url(self):
        self.assertTrue(Release(version="1.2.3", url="https://x").url)


class UpdateGating(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_disabled_means_no_request(self):
        self.config.updates.check_on_startup = False
        with mock.patch("openflow.updates.check") as fetch:
            from openflow.updates import check_if_due

            self.assertIsNone(check_if_due(self.config))
            fetch.assert_not_called()

    def test_it_does_not_check_twice_inside_the_interval(self):
        from openflow.updates import check_if_due

        self.config.updates.last_checked_at = 1_000_000.0
        with mock.patch("openflow.updates.check") as fetch, \
                mock.patch.object(Config, "save"):
            check_if_due(self.config, now=1_000_000.0 + 60)
            fetch.assert_not_called()

    def test_it_checks_once_the_interval_has_passed(self):
        from openflow.updates import check_if_due

        self.config.updates.last_checked_at = 1_000_000.0
        later = 1_000_000.0 + self.config.updates.interval_hours * 3600 + 1
        with mock.patch("openflow.updates.check", return_value=None) as fetch, \
                mock.patch.object(Config, "save"):
            check_if_due(self.config, now=later)
            fetch.assert_called_once()

    def test_a_network_failure_is_not_an_error(self):
        import urllib.error

        from openflow.updates import check

        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.URLError("offline")):
            self.assertIsNone(check())


if __name__ == "__main__":
    unittest.main(verbosity=2)
