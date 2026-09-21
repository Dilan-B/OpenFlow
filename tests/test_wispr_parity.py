"""Wispr Flow parity: literal uses survive, the model sees untrimmed text.

The model-quality half of parity is scored against tests/corpus/wispr_cases.json
by the harness (it needs a network and an API key). These tests pin the
deterministic half, which runs offline and on every commit.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.config import SCHEMA, Config  # noqa: E402
from openflow.llm.cleaner import LLMCleaner  # noqa: E402
from openflow.llm.prompts import FEW_SHOT  # noqa: E402
from openflow.llm.quota import QuotaLedger  # noqa: E402
from openflow.text.cleaner import RuleBasedCleaner, prepare_for_model  # noqa: E402

CORPUS = Path(__file__).parent / "corpus"


def _clean(text: str) -> str:
    return RuleBasedCleaner().clean(text).text


class LiteralUsesSurviveTheRulesPass(unittest.TestCase):
    """Deleting a verb breaks the sentence; deleting a filler only tidies it.

    These are the regressions that shipped: the rules pass read every "you
    know" and "I mean" as a disfluency.
    """

    def test_you_know_as_a_verb_is_kept(self):
        self.assertEqual(_clean("you know the answer already"),
                         "You know the answer already.")

    def test_do_you_know_is_kept(self):
        self.assertEqual(_clean("do you know the way"), "Do you know the way?")

    def test_i_mean_it_is_kept(self):
        self.assertEqual(_clean("I mean it this time"), "I mean it this time.")

    def test_you_know_as_filler_is_still_removed(self):
        self.assertEqual(_clean("the build is you know flaky"), "The build is flaky.")

    def test_comma_bounded_you_know_is_filler_even_before_an_object(self):
        """A comma makes it an aside, whatever follows."""
        self.assertEqual(_clean("it's, you know, the usual"), "It's the usual.")

    def test_i_mean_before_a_replacement_is_still_a_correction(self):
        """"the" is deliberately not a verb-object for "I mean", so the phrase
        is still consumed as a pivot. (Swapping "fifth" for "sixth" is beyond
        the rules pass's slot typing -- ordinals are the model's job.)"""
        self.assertNotIn("mean", _clean("the meeting is on the fifth, I mean the sixth"))


class ModelSeesWordsTheRulesWouldDelete(unittest.TestCase):
    """The containment guard forbids words absent from the model's input, so a
    model handed the rules' *output* can never undo a rules deletion."""

    def test_prepared_text_keeps_every_word(self):
        raw = "um so you know the answer I mean it"
        self.assertEqual(prepare_for_model(raw).lower().split(), raw.lower().split())

    def test_spoken_commands_are_still_resolved_for_the_model(self):
        self.assertIn("\n", prepare_for_model("first point new paragraph second point"))

    def test_cleaner_hands_the_provider_untrimmed_text(self):
        seen: list[str] = []

        class Recorder:
            name, is_local, small = "groq", False, True

            def available(self):
                return True

            def complete(self, system, user, *, strict=True, allowed=frozenset()):
                seen.append(user)
                return user

        cfg = Config()
        cfg.llm.backends = ["groq"]
        ledger = QuotaLedger(Path(tempfile.mkdtemp()) / "q.json")
        cleaner = LLMCleaner(cfg, quota=ledger)
        with mock.patch("openflow.llm.cleaner.build_provider", return_value=Recorder()), \
                mock.patch("openflow.llm.cleaner.system_prompt_for", return_value=""):
            cleaner.clean("can we meet tuesday at five or actually make it friday at three")
        self.assertEqual(len(seen), 1)
        # The rules pass alone reduces this to "Make it Friday at three." --
        # the frame the model needs is only there if it saw the raw words.
        self.assertIn("can we meet", seen[0].lower())


class PromptHygiene(unittest.TestCase):
    def test_few_shot_pairs_never_reuse_a_scored_case(self):
        """Examples copied from the corpus would score recall, not behaviour."""
        scored = set()
        for path in CORPUS.glob("*.json"):
            for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
                scored.add(case["input"].lower())
        for example_in, _out in FEW_SHOT:
            self.assertNotIn(example_in.lower(), scored)

    def test_few_shot_outputs_only_delete(self):
        """Every pair must obey the same containment rule the guard enforces."""
        from openflow.llm.base import check_containment

        for example_in, example_out in FEW_SHOT:
            self.assertEqual(check_containment(example_out, example_in), [],
                             example_in)


class CleanupModelMigration(unittest.TestCase):
    def _load(self, data: dict) -> Config:
        path = Path(tempfile.mkdtemp()) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return Config.load(path)

    def test_moves_off_the_retired_default(self):
        cfg = self._load({"schema": 1, "llm": {"groq_model": "openai/gpt-oss-20b"}})
        self.assertEqual(cfg.llm.groq_model, "qwen/qwen3.8-27b")
        self.assertEqual(cfg.schema, SCHEMA)

    def test_leaves_any_other_model_alone(self):
        cfg = self._load({"schema": 1, "llm": {"groq_model": "openai/gpt-oss-120b"}})
        self.assertEqual(cfg.llm.groq_model, "openai/gpt-oss-120b")


if __name__ == "__main__":
    unittest.main()
