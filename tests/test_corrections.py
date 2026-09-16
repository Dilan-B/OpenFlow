"""Tests for learned corrections, the config migration, and 429 handling."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.config import Config  # noqa: E402
from openflow.corrections import (  # noqa: E402
    AUTO_APPLY_COUNT, Corrections, diff_corrections,
)
from openflow.llm.quota import is_daily_exhaustion  # noqa: E402
from openflow.personalization import Personalization  # noqa: E402


def _store() -> Corrections:
    return Corrections(Path(tempfile.mkdtemp()) / "corrections.json")


class Diffing(unittest.TestCase):
    def test_extracts_a_single_word_substitution(self):
        self.assertEqual(
            diff_corrections("I met Dylan on Friday.", "I met Dilan on Friday."),
            [("Dylan", "Dilan")],
        )

    def test_ignores_pure_punctuation_changes(self):
        self.assertEqual(
            diff_corrections("rebuild the index", "rebuild the index."), []
        )

    def test_ignores_insertions_and_deletions(self):
        """Adding or cutting a clause is the user editing their own prose.

        Replaying that later would delete words they actually said.
        """
        self.assertEqual(
            diff_corrections("Ship it on Friday.", "Ship it on Friday. Thanks."), []
        )
        self.assertEqual(
            diff_corrections("Ship it on Friday if you can.", "Ship it on Friday."), []
        )

    def test_learns_a_multi_word_span(self):
        self.assertEqual(
            diff_corrections("run the G R P C cutover", "run the gRPC cutover"),
            [("G R P C", "gRPC")],
        )

    def test_rejects_a_span_longer_than_the_cap(self):
        """A whole-sentence rewrite is not a correction worth replaying."""
        self.assertEqual(
            diff_corrections(
                "one two three four five six",
                "alpha bravo charlie delta echo foxtrot",
            ),
            [],
        )


class Learning(unittest.TestCase):
    def test_learns_and_replays_a_name(self):
        store = _store()
        store.learn("I met Dylan on Friday.", "I met Dilan on Friday.")
        # A capitalized proper-noun fix is trusted immediately: repeating
        # yourself will never make the recogniser guess your spelling.
        self.assertEqual(store.apply("Dylan is joining."), "Dilan is joining.")

    def test_ordinary_word_needs_a_second_sighting(self):
        store = _store()
        store.learn("the flaky test", "the flakey test")
        self.assertEqual(store.apply("the flaky test"), "the flaky test")
        store.learn("the flaky test", "the flakey test")
        self.assertEqual(store.apply("the flaky test"), "the flakey test")

    def test_repeating_a_correction_increments_rather_than_duplicates(self):
        store = _store()
        store.learn("Dylan", "Dilan")
        store.learn("Dylan", "Dilan")
        self.assertEqual(len(store.entries), 1)
        self.assertEqual(store.entries[0].count, 2)

    def test_identical_texts_teach_nothing(self):
        store = _store()
        self.assertEqual(store.learn("same text", "same text"), [])
        self.assertEqual(store.entries, [])

    def test_replacement_respects_word_boundaries(self):
        """A rule for "Dylan" must not fire inside "Dylans" or "Bobdylan"."""
        store = _store()
        store.learn("Dylan", "Dilan")
        self.assertEqual(store.apply("Bobdylan stayed."), "Bobdylan stayed.")

    def test_longer_spans_replace_before_shorter_ones(self):
        store = _store()
        store.learn("the grpc cutover", "the gRPC cutover")
        store.learn("grpc", "gRPC")
        self.assertEqual(
            store.apply("finish the grpc cutover"), "finish the gRPC cutover"
        )

    def test_survives_a_reload(self):
        path = Path(tempfile.mkdtemp()) / "corrections.json"
        Corrections(path).learn("Dylan", "Dilan")
        self.assertEqual(Corrections(path).apply("Dylan"), "Dilan")

    def test_forget_removes_a_rule(self):
        store = _store()
        store.learn("Dylan", "Dilan")
        store.forget("Dylan", "Dilan")
        self.assertEqual(store.apply("Dylan"), "Dylan")


class PushingKnowledgeUpstream(unittest.TestCase):
    def test_promotes_a_name_into_the_dictionary(self):
        """The point of promotion: bias recognition, not just patch output."""
        store = _store()
        personal = Personalization.load(Path(tempfile.mkdtemp()) / "p.json")
        store.learn("I met Dylan today.", "I met Dilan today.")
        self.assertEqual(store.promote(personal), ["Dilan"])
        self.assertIn("Dilan", personal.dictionary)

    def test_does_not_promote_a_multi_word_span(self):
        store = _store()
        personal = Personalization.load(Path(tempfile.mkdtemp()) / "p.json")
        store.learn("run the G R P C job", "run the gRPC Cutover job")
        self.assertEqual(store.promote(personal), [])

    def test_prompt_hint_names_active_corrections(self):
        store = _store()
        store.learn("Dylan", "Dilan")
        hint = store.prompt_hint()
        self.assertIn("Dilan", hint)
        self.assertIn("never reverse", hint.lower())

    def test_prompt_hint_is_empty_with_nothing_learned(self):
        self.assertEqual(_store().prompt_hint(), "")


class TransientRateLimits(unittest.TestCase):
    """A per-minute 429 must not disable a backend for the rest of the day."""

    def test_per_minute_limit_is_not_daily(self):
        self.assertFalse(is_daily_exhaustion(
            "HTTP 429: Rate limit reached for model X on requests per minute (RPM)"
        ))

    def test_per_day_limit_is_daily(self):
        self.assertTrue(is_daily_exhaustion(
            "HTTP 429: Rate limit reached for model X on requests per day (RPD)"
        ))

    def test_unrecognised_wording_is_treated_as_transient(self):
        self.assertFalse(is_daily_exhaustion("HTTP 429: too many requests"))


class ConfigMigration(unittest.TestCase):
    def _write(self, data: dict) -> Path:
        path = Path(tempfile.mkdtemp()) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_turns_on_a_cleanup_pass_that_was_gated_into_a_no_op(self):
        cfg = Config.load(self._write({
            "llm": {"enabled": False, "only_when_uncertain": True,
                    "backends": ["gemini", "ollama", "rules"]},
        }))
        self.assertTrue(cfg.llm.enabled)
        self.assertFalse(cfg.llm.only_when_uncertain)
        self.assertEqual(cfg.llm.backends[0], "groq")
        self.assertEqual(cfg.schema, 1)

    def test_moves_off_the_turbo_transcription_model(self):
        cfg = Config.load(self._write({"stt": {"groq_model": "whisper-large-v3-turbo"}}))
        self.assertEqual(cfg.stt.groq_model, "whisper-large-v3")

    def test_leaves_a_deliberately_chosen_model_alone(self):
        cfg = Config.load(self._write({"stt": {"groq_model": "whisper-large-v3-turbo"},
                                       "schema": 1}))
        self.assertEqual(cfg.stt.groq_model, "whisper-large-v3-turbo")

    def test_does_not_re_enable_cleanup_after_the_user_turns_it_off(self):
        """The migration runs once. Turning it back off has to stick."""
        cfg = Config.load(self._write({"schema": 1, "llm": {"enabled": False}}))
        self.assertFalse(cfg.llm.enabled)

    def test_adds_the_separate_cleanup_quota_key(self):
        cfg = Config.load(self._write({"llm": {"daily_limits": {"groq": 2000}}}))
        self.assertIn("groq_chat", cfg.llm.daily_limits)
        self.assertEqual(cfg.llm.daily_limits["groq"], 2000)


if __name__ == "__main__":
    unittest.main()
