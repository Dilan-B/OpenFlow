"""Tests for the LLM output guards and the free-tier quota ledger.

These cover the parts of the pipeline that protect the user from a model that
misbehaves, which is exactly where a silent failure would be most expensive:
a hallucinated substitution gets typed straight into their document.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.config import Config  # noqa: E402
from openflow.llm.base import ProviderError, check_containment, sanitize  # noqa: E402
from openflow.llm.quota import QuotaLedger  # noqa: E402
from openflow.stt.engines import build_engine  # noqa: E402


class Containment(unittest.TestCase):
    """The documented ASR post-correction failure: a same-length substitution
    of a similar-sounding entity (arXiv 2601.15397)."""

    def test_catches_entity_substitution(self):
        invented = check_containment("I like Al Gore.", "I like algorithms")
        self.assertIn("gore", invented)

    def test_sanitize_rejects_entity_substitution(self):
        with self.assertRaises(ProviderError) as ctx:
            sanitize("I like Al Gore.", original="um, I like algorithms")
        self.assertIn("did not say", str(ctx.exception))

    def test_allows_pure_deletion(self):
        self.assertEqual([], check_containment("I like algorithms.", "um I like algorithms"))

    def test_allows_contraction_changes(self):
        self.assertEqual([], check_containment("I'll ship it.", "i will ship it"))
        self.assertEqual([], check_containment("I will ship it.", "i'll ship it"))

    def test_allows_added_articles(self):
        self.assertEqual([], check_containment("Ship the build.", "ship build"))

    def test_case_and_punctuation_insensitive(self):
        self.assertEqual([], check_containment("Friday at 3.", "friday at 3"))

    def test_rejects_appended_commentary(self):
        with self.assertRaises(ProviderError):
            sanitize("Ship it. Let me know if you need anything else!", original="ship it")

    def test_still_rejects_runaway_length(self):
        with self.assertRaises(ProviderError) as ctx:
            sanitize("ship it " * 40, original="ship it")
        self.assertIn("longer than input", str(ctx.exception))

    def test_strips_preamble_and_quotes(self):
        self.assertEqual(
            "Ship it on Friday.",
            sanitize('Here is your text: "Ship it on Friday."', original="ship it on friday"),
        )

    def test_empty_completion_rejected(self):
        with self.assertRaises(ProviderError):
            sanitize("   ", original="ship it")


class Quota(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "quota.json"
        self.now = 1_000_000.0

    def tearDown(self):
        self.tmp.cleanup()

    def ledger(self) -> QuotaLedger:
        return QuotaLedger(self.path, clock=lambda: self.now)

    def test_daily_request_ceiling(self):
        q = self.ledger()
        self.assertTrue(q.has_headroom("groq", 2))
        q.record("groq")
        q.record("groq")
        self.assertFalse(q.has_headroom("groq", 2))

    def test_no_limit_means_unlimited(self):
        self.assertTrue(self.ledger().has_headroom("groq", None))

    def test_hourly_audio_budget_blocks_before_sending(self):
        q = self.ledger()
        q.record("groq", audio_seconds=7_000)
        # 7000 used + 300 wanted > 7200, so we must not send.
        self.assertFalse(q.has_audio_headroom("groq", 7_200, 300))
        self.assertTrue(q.has_audio_headroom("groq", 7_200, 100))

    def test_audio_budget_is_a_rolling_hour(self):
        q = self.ledger()
        q.record("groq", audio_seconds=7_000)
        self.assertAlmostEqual(7_000, q.audio_seconds_used("groq"))
        self.now += 3_601                      # an hour and a second later
        self.assertEqual(0, q.audio_seconds_used("groq"))
        self.assertTrue(q.has_audio_headroom("groq", 7_200, 300))

    def test_exhaust_marks_provider_spent(self):
        q = self.ledger()
        q.exhaust("gemini", 1_400)
        self.assertFalse(q.has_headroom("gemini", 1_400))

    def test_state_survives_restart(self):
        q = self.ledger()
        q.record("groq", audio_seconds=120)
        reopened = self.ledger()
        self.assertEqual(1, reopened.used("groq"))
        self.assertAlmostEqual(120, reopened.audio_seconds_used("groq"))

    def test_unwritable_path_does_not_raise(self):
        q = QuotaLedger(Path(self.tmp.name) / "nope" / "x" / "quota.json",
                        clock=lambda: self.now)
        q.record("groq", audio_seconds=5)      # must not raise
        self.assertEqual(1, q.used("groq"))

    def test_corrupt_file_is_ignored(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(0, self.ledger().used("groq"))

    def test_snapshot_shape(self):
        q = self.ledger()
        q.record("groq", audio_seconds=30)
        snap = q.snapshot()
        self.assertEqual(1, snap["requests"]["groq"])
        self.assertAlmostEqual(30, snap["audio_seconds"]["groq"])
        json.dumps(snap)                        # must stay serializable


class EngineRegistry(unittest.TestCase):
    def test_all_default_backends_resolve(self):
        config = Config()
        for name in config.stt.backends:
            engine = build_engine(name, config)
            self.assertTrue(hasattr(engine, "transcribe"))
            self.assertTrue(hasattr(engine, "available"))

    def test_parakeet_leads_the_local_chain(self):
        backends = Config().stt.backends
        self.assertIn("parakeet_onnx", backends)
        self.assertLess(backends.index("parakeet_onnx"), backends.index("faster_whisper"))

    def test_moonshine_is_selectable(self):
        engine = build_engine("moonshine", Config())
        self.assertEqual("moonshine", engine.name)
        self.assertTrue(engine.is_local)

    def test_unknown_engine_raises(self):
        with self.assertRaises(ValueError):
            build_engine("nope", Config())


if __name__ == "__main__":
    unittest.main(verbosity=2)
