"""Wispr parity: paid models, screen context, long and multilingual dictation,
whisper-quiet audio, email layout, file tagging, and the start/stop sounds."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from openflow import keys  # noqa: E402
from openflow.config import (  # noqa: E402
    Config, languages, llm_chain, stt_chain, unlimited, _migrate,
)
from openflow.context import (  # noqa: E402
    DictationContext, extract_names, ide_workspace, repair_names, tag_files,
    with_own_name,
)
from openflow.formatting import AppKind, CaretContext, layout_email, smart_format  # noqa: E402
from openflow.llm.base import ProviderError, check_containment, sanitize  # noqa: E402

RATE = 16_000
OUTLOOK = AppKind("email", False)
SLACK = AppKind("work", True)


def tone(seconds: float, amp: float = 0.2, freq: float = 220.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class Tiers(unittest.TestCase):
    def test_free_tier_keeps_the_configured_chains(self):
        cfg = Config()
        self.assertEqual(stt_chain(cfg), cfg.stt.backends)
        self.assertEqual(llm_chain(cfg), cfg.llm.backends)
        self.assertFalse(unlimited(cfg, "groq"))

    def test_pro_puts_paid_engines_first_and_keeps_fallbacks(self):
        cfg = Config()
        cfg.models.tier = "pro"
        self.assertEqual(stt_chain(cfg)[:2], ["openai", "deepgram"])
        self.assertIn("parakeet_onnx", stt_chain(cfg))
        self.assertEqual(llm_chain(cfg)[:2], ["anthropic", "openai"])
        self.assertEqual(llm_chain(cfg)[-1], "rules")

    def test_pro_with_a_chosen_engine_leads_with_it_once(self):
        cfg = Config()
        cfg.models.tier = "pro"
        cfg.models.transcription = "groq"
        cfg.models.cleanup = "gemini"
        self.assertEqual(stt_chain(cfg).count("groq"), 1)
        self.assertEqual(stt_chain(cfg)[0], "groq")
        self.assertEqual(llm_chain(cfg)[0], "gemini")

    def test_pro_lifts_limits_except_groq_unless_it_is_paid(self):
        cfg = Config()
        cfg.models.tier = "pro"
        self.assertTrue(unlimited(cfg, "gemini"))
        self.assertFalse(unlimited(cfg, "groq_chat"))
        cfg.models.groq_paid = True
        self.assertTrue(unlimited(cfg, "groq_chat"))

    def test_paid_keys_are_saveable(self):
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY"):
            self.assertIn(name, keys.KNOWN)
            self.assertNotIn(name, keys.FREE)


class Migration(unittest.TestCase):
    def test_the_old_two_minute_cap_becomes_twenty(self):
        cfg = Config()
        cfg.schema, cfg.audio.max_seconds = 3, 120
        _migrate(cfg)
        self.assertEqual(cfg.audio.max_seconds, 1200)

    def test_a_chosen_cap_is_kept(self):
        cfg = Config()
        cfg.schema, cfg.audio.max_seconds = 3, 300
        _migrate(cfg)
        self.assertEqual(cfg.audio.max_seconds, 300)


class Languages(unittest.TestCase):
    def test_primary_first_without_duplicates(self):
        cfg = Config()
        cfg.stt.language, cfg.stt.languages = "en", ["en", "gu", "hi"]
        self.assertEqual(languages(cfg), ["en", "gu", "hi"])

    def test_empty_means_detect(self):
        cfg = Config()
        cfg.stt.language, cfg.stt.languages = "", []
        self.assertEqual(languages(cfg), [])

    def test_groq_retries_a_language_the_speaker_does_not_use(self):
        from openflow.stt.engines import GroqWhisper

        cfg = Config()
        cfg.stt.language, cfg.stt.languages = "en", ["en", "es"]
        engine = GroqWhisper(cfg)
        calls = []

        def post(fields, wav, timeout):
            calls.append(dict(fields))
            if "language" not in fields:
                return {"text": "Hallo", "language": "german"}
            return {"text": "Hello"}

        with mock.patch.object(GroqWhisper, "key", "gsk_test"), \
                mock.patch.object(engine, "_post", side_effect=post), \
                mock.patch("openflow.stt.engines.vocabulary_prompt", return_value=""):
            self.assertEqual(engine.transcribe(tone(1.0), RATE), "Hello")
        self.assertEqual(calls[0]["response_format"], "verbose_json")
        self.assertEqual(calls[1]["language"], "en")
        self.assertEqual(engine.last_language, "en")

    def test_non_english_skips_the_english_rules(self):
        from openflow.llm.cleaner import LLMCleaner

        cfg = Config()
        cfg.llm.enabled = False
        cleaner = LLMCleaner(cfg, quota=mock.Mock())
        spanish = "eh bueno podemos vernos el martes"
        # The English filler/pivot lexicon would otherwise treat "no", "like"
        # and friends as English words.
        self.assertEqual(cleaner.clean(spanish, language="es").text, spanish)


class LongDictation(unittest.TestCase):
    def test_split_points_land_in_pauses(self):
        from openflow.stt.router import split_points

        speech, pause = tone(9.0), np.zeros(int(0.6 * RATE), np.float32)
        audio = np.concatenate([speech, pause] * 8)            # ~77 s
        cuts = split_points(audio, RATE, 28)
        self.assertTrue(cuts)
        for cut in cuts:
            self.assertLess(abs(float(audio[cut])), 1e-6, cut / RATE)
        pieces = np.diff([0] + cuts + [len(audio)]) / RATE
        self.assertLessEqual(max(pieces), 28.0)

    def test_short_audio_is_not_split(self):
        from openflow.stt.router import split_points

        self.assertEqual(split_points(tone(10.0), RATE, 28), [])

    def test_router_chunks_to_the_engines_window_and_joins(self):
        from openflow.stt.router import SttRouter

        cfg = Config()
        cfg.stt.backends = ["fake"]
        router = SttRouter(cfg, quota=mock.Mock(**{"has_headroom.return_value": True,
                                                   "has_audio_headroom.return_value": True}))
        seen = []
        engine = SimpleNamespace(
            name="fake", is_local=False, max_chunk_s=28, available=lambda: True,
            transcribe=lambda a, r, context=None: seen.append(len(a) / r) or "piece,",
        )
        router._engines["fake"] = engine
        audio = np.concatenate([tone(9.0), np.zeros(int(0.6 * RATE), np.float32)] * 8)
        out = router.transcribe(audio, RATE)
        self.assertGreater(len(seen), 2)
        self.assertLessEqual(max(seen), 28.0)
        self.assertEqual(out.text, " ".join(["piece,"] * len(seen)))

    def test_join_lowercases_a_mid_sentence_seam_but_not_names(self):
        from openflow.stt.router import _join

        self.assertEqual(_join(["Seventh,", "Check the logs."]), "Seventh, check the logs.")
        self.assertEqual(_join(["Ask", "Sydney first."]), "Ask Sydney first.")

    def test_output_budget_grows_with_the_dictation(self):
        from openflow.llm.providers import output_budget

        self.assertEqual(output_budget("short one"), 512)
        self.assertGreater(output_budget("word " * 3000), 6000)


class WhisperQuiet(unittest.TestCase):
    def test_a_whisper_is_not_discarded_as_silence(self):
        from openflow.audio.conditioning import condition, is_silent

        rng = np.random.default_rng(1)
        room = rng.normal(0, 0.00008, RATE * 3).astype(np.float32)
        envelope = np.zeros(RATE * 3, np.float32)
        envelope[RATE // 2:int(RATE * 2.5)] = 1
        whisper = room + (rng.normal(0, 0.0009, RATE * 3) * envelope).astype(np.float32)
        self.assertFalse(is_silent(whisper, RATE))
        self.assertGreater(float(np.abs(condition(whisper, RATE)).max()), 0.5)

    def test_room_noise_alone_is_still_silence(self):
        from openflow.audio.conditioning import is_silent

        room = np.random.default_rng(2).normal(0, 0.00008, RATE * 3).astype(np.float32)
        self.assertTrue(is_silent(room, RATE))


class ScreenNames(unittest.TestCase):
    EMAIL = (
        "Inbox - Gmail\nFrom: Sahed Rahman <sahed@example.com>\nHi Dilan,\n"
        "I talked to Sydney and Priya about the OpenFlow launch, and Thursday works.\n"
        "Reply Forward Archive Settings\nBest,\nSahed"
    )

    def test_names_come_from_greetings_headers_and_mid_sentence(self):
        names = extract_names(self.EMAIL)
        for name in ("Sahed", "Dilan", "Sydney", "Priya", "OpenFlow"):
            self.assertIn(name, names)

    def test_labels_weekdays_and_sentence_starts_are_not_names(self):
        names = extract_names(self.EMAIL)
        for word in ("Reply", "Settings", "Thursday", "Inbox", "Best"):
            self.assertNotIn(word, names)

    def test_near_misses_take_the_screen_spelling(self):
        self.assertEqual(repair_names("hey sidney, ping sahid", ["Sydney", "Sahed"]),
                         "hey Sydney, ping Sahed")

    def test_ordinary_words_are_not_turned_into_names(self):
        self.assertEqual(repair_names("the market is up", ["Marek"]), "the market is up")

    def test_own_name_is_always_known(self):
        ctx = with_own_name(DictationContext(), "Dilan")
        self.assertEqual(ctx.names, ("Dilan",))
        self.assertEqual(repair_names("Thanks, Dylan", ctx.names), "Thanks, Dilan")

    def test_containment_allows_screen_spellings_only(self):
        allowed = DictationContext(names=("Sahed",)).allowed_words()
        self.assertEqual(check_containment("Hey Sahed", "hey sahid", allowed), [])
        self.assertEqual(check_containment("Hey Marcus", "hey sahid", allowed), ["marcus"])
        with self.assertRaises(ProviderError):
            sanitize("Hey Marcus", original="hey sahid", allowed=allowed)


class CodeContext(unittest.TestCase):
    FILES = ["auth.ts", "enums.ts", "user_service.py", "index.ts"]

    def test_spoken_file_names_become_tags(self):
        self.assertEqual(
            tag_files("look at auth dot ts and enums.ts, then user service dot py",
                      self.FILES),
            "look at @auth.ts and @enums.ts, then @user_service.py")

    def test_bare_words_and_existing_tags_are_left_alone(self):
        self.assertEqual(tag_files("fix the auth flow in @auth.ts", self.FILES),
                         "fix the auth flow in @auth.ts")

    def test_workspace_is_read_from_the_editors_window_state(self):
        import tempfile

        with tempfile.TemporaryDirectory() as home:
            project = Path(home) / "SlopeSide"
            project.mkdir()
            storage = Path(home) / "Cursor" / "User" / "globalStorage" / "storage.json"
            storage.parent.mkdir(parents=True)
            storage.write_text(json.dumps({"windowsState": {"lastActiveWindow": {
                "folder": project.as_uri()}}}), encoding="utf-8")
            with mock.patch("openflow.context._storage_path",
                            lambda folder: Path(home) / folder / "User" / "globalStorage"
                            / "storage.json"):
                found = ide_workspace("cursor.exe", "App.tsx - SlopeSide - Cursor")
            self.assertEqual(found, project)
            self.assertIsNone(ide_workspace("notepad.exe", "x"))

    def test_identifiers_are_multi_word_only(self):
        import tempfile

        from openflow.context import index_identifiers

        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "a.ts").write_text(
                "const getUserName = () => users; let referral_count = 0; class UserService {}",
                encoding="utf-8")
            found = index_identifiers(Path(root), ["a.ts"])
        self.assertEqual(set(found), {"getUserName", "referral_count", "UserService"})

    def test_ai_editors_write_prose(self):
        from openflow.profiles import profile_for

        # The old "code" profile lowercased and dropped the full stop -- right
        # for a code line, wrong for the chat prompts most IDE dictation is.
        out = smart_format("Hey, can you look at @auth.ts? It’s broken.",
                           profile=profile_for("cursor.exe"))
        self.assertEqual(out, "Hey, can you look at @auth.ts? It's broken.")

    def test_prompt_names_the_files_and_the_tag_rule(self):
        from openflow.llm.prompts import context_section

        ctx = DictationContext(app="cursor.exe", files=("src/auth.ts",),
                               identifiers=("getUserName",))
        section = context_section(ctx)
        self.assertIn("auth.ts", section)
        self.assertIn("@", section)
        self.assertIn("getUserName", section)


class EmailAndTimes(unittest.TestCase):
    def test_greeting_and_sign_off_get_their_own_lines(self):
        out = smart_format(
            "Hey Sahed, there's three things Wispr does really well. Thanks, Dilan.",
            kind=OUTLOOK)
        self.assertEqual(out, "Hey Sahed,\n\nThere's three things Wispr does really well."
                              "\n\nThanks,\nDilan")

    def test_no_layout_mid_email_or_outside_email(self):
        text = "Hi Sam, see you at 5."
        self.assertEqual(smart_format(text, kind=SLACK), "Hi Sam, see you at 5")
        mid = CaretContext(before="Following up on this. ")
        self.assertNotIn("\n", smart_format(text, kind=OUTLOOK, context=mid))

    def test_layout_leaves_plain_messages_alone(self):
        self.assertEqual(layout_email("Can we move it to Friday?"), "Can we move it to Friday?")

    def test_the_slack_example_from_the_video(self):
        for text in ("Hey Sydney, want to meet at 5:30 PM.?",
                     "Hey Sydney, want to meet at 5.30 PM?"):
            self.assertEqual(smart_format(text, kind=SLACK), "Hey Sydney, want to meet at 5:30 PM?")

    def test_dotted_decimals_and_real_abbreviations_survive(self):
        self.assertEqual(smart_format("It costs 5.30 dollars."), "It costs 5.30 dollars.")
        self.assertEqual(smart_format("See you at 10.45 a.m.?"), "See you at 10:45 a.m.?")


class RulesFixes(unittest.TestCase):
    def test_sentence_final_right_is_a_word(self):
        from openflow.text.cleaner import RuleBasedCleaner

        rules = RuleBasedCleaner()
        self.assertEqual(rules.clean("it gets names right").text, "It gets names right.")
        self.assertEqual(rules.clean("that is the plan, right").text, "That is the plan.")

    def test_clock_times_keep_their_colon(self):
        from openflow.text.cleaner import RuleBasedCleaner

        self.assertEqual(RuleBasedCleaner().clean("meet at 5:30 p.m.").text, "Meet at 5:30 PM.")


class PaidProviders(unittest.TestCase):
    def test_anthropic_request_shape(self):
        from openflow.llm.providers import AnthropicProvider

        response = SimpleNamespace(stop_reason="end_turn",
                                   content=[SimpleNamespace(type="text", text="Hey Sahed.")])
        client = mock.Mock()
        client.beta.messages.create.return_value = response
        fake = mock.Mock()
        fake.Anthropic.return_value = client
        with mock.patch.dict("sys.modules", {"anthropic": fake}), \
                mock.patch("openflow.llm.providers.api_key", return_value="sk-ant-x"):
            out = AnthropicProvider(Config()).complete(
                "SYSTEM", "hey sahid", allowed=frozenset({"sahed"}))
        self.assertEqual(out, "Hey Sahed.")
        kwargs = client.beta.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-opus-5")
        self.assertEqual(kwargs["fallbacks"], "default")
        self.assertEqual(kwargs["output_config"], {"effort": "low"})
        self.assertEqual(kwargs["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(kwargs["messages"][-1], {"role": "user", "content": "hey sahid"})
        self.assertNotIn("temperature", kwargs)

    def test_anthropic_refusal_falls_through(self):
        from openflow.llm.providers import AnthropicProvider

        client = mock.Mock()
        client.beta.messages.create.return_value = SimpleNamespace(stop_reason="refusal", content=[])
        fake = mock.Mock()
        fake.Anthropic.return_value = client
        with mock.patch.dict("sys.modules", {"anthropic": fake}), \
                mock.patch("openflow.llm.providers.api_key", return_value="sk-ant-x"), \
                self.assertRaises(ProviderError):
            AnthropicProvider(Config()).complete("S", "hello")

    def test_openai_chat_turns_reasoning_off(self):
        from openflow.llm import providers

        with mock.patch.object(providers, "api_key", return_value="sk-x"), \
                mock.patch.object(providers, "_post", return_value={
                    "choices": [{"message": {"content": "Hello."}}]}) as post:
            self.assertEqual(providers.OpenAIProvider(Config()).complete("S", "hello"), "Hello.")
        payload = post.call_args.args[1]
        self.assertEqual(payload["model"], "gpt-5.6-luna")
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("temperature", payload)

    def test_openai_transcribe_sends_languages_and_keywords(self):
        from openflow.stt import engines

        cfg = Config()
        cfg.stt.language, cfg.stt.languages = "en", ["en", "hi"]
        captured = {}

        def send(request, timeout):
            captured["body"] = request.data
            return {"text": "hello"}

        ctx = DictationContext(names=("Sahed",))
        with mock.patch.object(engines, "api_key", return_value="sk-x"), \
                mock.patch.object(engines, "_send", side_effect=send):
            self.assertEqual(engines.OpenAITranscribe(cfg).transcribe(tone(1.0), RATE,
                                                                      context=ctx), "hello")
        body = captured["body"]
        self.assertEqual(body.count(b'name="languages[]"'), 2)
        self.assertIn(b"gpt-transcribe", body)
        self.assertIn(b"Sahed", body)

    def test_deepgram_multilingual_and_keyterms(self):
        from urllib.parse import parse_qs, urlparse

        from openflow.stt import engines

        cfg = Config()
        cfg.stt.language, cfg.stt.languages = "en", ["en", "es"]
        captured = {}

        def send(request, timeout):
            captured["url"] = request.full_url
            return {"results": {"channels": [{"alternatives": [{"transcript": " hola "}]}]}}

        with mock.patch.object(engines, "api_key", return_value="dg"), \
                mock.patch.object(engines, "_send", side_effect=send):
            out = engines.Deepgram(cfg).transcribe(
                tone(1.0), RATE, context=DictationContext(names=("Marta",)))
        self.assertEqual(out, "hola")
        query = parse_qs(urlparse(captured["url"]).query)
        self.assertEqual(query["model"], ["nova-3"])
        self.assertEqual(query["language"], ["multi"])
        self.assertIn("Marta", query["keyterm"])


class Sounds(unittest.TestCase):
    def test_start_rises_and_stop_falls(self):
        from openflow.audio import chime

        def first_pitches(pops):
            return [pop[1] for pop in pops[:2]]

        start, stop = first_pitches(chime._START), first_pitches(chime._STOP)
        self.assertLess(start[0], start[1])
        self.assertGreater(stop[0], stop[1])

    def test_both_are_short_clean_and_quiet(self):
        from openflow.audio import chime

        for kind in ("start", "stop"):
            samples = chime.samples(kind)
            self.assertLess(len(samples) / chime.SAMPLE_RATE, 0.2)
            self.assertLessEqual(float(np.abs(samples).max()), chime.VOLUME + 1e-6)
            self.assertAlmostEqual(float(samples[0]), 0.0, places=4)
            self.assertAlmostEqual(float(samples[-1]), 0.0, places=4)

    def test_stop_never_raises(self):
        from openflow.audio import chime

        with mock.patch.dict("sys.modules", {"sounddevice": None}):
            chime.play_stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
