"""Tests for LLM backend readiness.

The bug behind these: a config saved before a default changed kept naming
gemini-1.5-flash long after Google retired it. available() only checks that an
API key exists, so `--check` reported "ready" while every cleanup 404ed and
fell through to the rules pass -- a backend that looked healthy and did
nothing, for as long as nobody read the log.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.config import Config, LlmConfig
from openflow.llm.base import ProviderError
from openflow.llm.providers import GeminiProvider


class _Stub(GeminiProvider):
    """A Gemini provider whose round-trip is scripted."""

    def __init__(self, outcome, key="test-key", model="gemini-flash-lite-latest"):
        config = Config()
        config.llm.gemini_model = model
        self.cfg = config.llm
        self.key = key
        self.model = model
        self._outcome = outcome

    def complete(self, system, user, *, strict=True):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class Verify(unittest.TestCase):
    def test_working_backend_reports_no_problem(self):
        self.assertIsNone(_Stub("ok").verify())

    def test_retired_model_is_named(self):
        """The whole point: available() says yes, verify() says which model."""
        stub = _Stub(ProviderError("HTTP 404: not found for API version v1beta"),
                     model="gemini-1.5-flash")
        self.assertTrue(stub.available())
        self.assertEqual(stub.verify(),
                         "model 'gemini-1.5-flash' is not available to this key")

    def test_quota_exhaustion_is_distinguished_from_a_bad_model(self):
        stub = _Stub(ProviderError("HTTP 429: You exceeded your current quota"))
        self.assertEqual(stub.verify(), "daily quota exhausted")

    def test_missing_key_short_circuits_before_the_network(self):
        stub = _Stub(AssertionError("must not be called"), key=None)
        self.assertEqual(stub.verify(), "GEMINI_API_KEY is not set")

    def test_other_failures_are_reported_on_one_line(self):
        stub = _Stub(ProviderError("HTTP 503: high demand\n  spikes\n  later"))
        problem = stub.verify()
        self.assertNotIn("\n", problem)
        self.assertIn("503", problem)
        self.assertLessEqual(len(problem), 110)


class Timeouts(unittest.TestCase):
    def test_warmup_gets_a_bigger_budget_than_a_request(self):
        """Sharing one timeout is what broke warm-up: a disk-cold llama3.1:8b
        takes ~18 s to load, so the 6 s request budget guaranteed the cold load
        landed on the first dictation instead."""
        llm = LlmConfig()
        self.assertGreater(llm.warmup_timeout_s, llm.timeout_s)
        self.assertGreaterEqual(llm.warmup_timeout_s, 20.0)

    def test_default_model_is_an_alias_not_a_pinned_version(self):
        """Pinned names retire. An alias cannot rot the same way."""
        self.assertTrue(LlmConfig().gemini_model.endswith("-latest"))


if __name__ == "__main__":
    unittest.main()
