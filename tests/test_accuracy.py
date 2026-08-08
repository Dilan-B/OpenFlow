"""Tests for the accuracy layer: phonetics, audio conditioning, question marks."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openflow.audio.conditioning import condition, is_silent  # noqa: E402
from openflow.personalization import Personalization  # noqa: E402
from openflow.phonetics import metaphone, phrase_key  # noqa: E402
from openflow.text.cleaner import RuleBasedCleaner  # noqa: E402
from openflow.text.punctuation import fix_punctuation, looks_like_question  # noqa: E402

RATE = 16_000


def _personal(terms: list[str]) -> Personalization:
    tmp = tempfile.mkdtemp()
    p = Personalization.load(Path(tmp) / "p.json")
    for term in terms:
        p.add_term(term)
    return p


def _tone(seconds: float, amplitude: float = 0.3, rate: int = RATE):
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype("float32")


class Phonetics(unittest.TestCase):
    def test_homophones_share_a_key(self):
        self.assertEqual(metaphone("Groq"), metaphone("grock"))
        self.assertEqual(metaphone("Sara"), metaphone("Sarah"))
        self.assertEqual(metaphone("Kate"), metaphone("Cate"))

    def test_distinct_words_differ(self):
        self.assertNotEqual(metaphone("Kubernetes"), metaphone("elephant"))
        self.assertNotEqual(metaphone("Postgres"), metaphone("progress"))

    def test_phrase_key_ignores_word_splits(self):
        self.assertEqual(phrase_key(["OpenFlow"]), phrase_key(["open", "flow"]))
        self.assertEqual(phrase_key(["Kubernetes"]), phrase_key(["kuber", "netes"]))

    def test_silent_clusters(self):
        self.assertEqual(metaphone("knight"), metaphone("night"))
        self.assertEqual(metaphone("wright"), metaphone("right"))

    def test_empty_input(self):
        self.assertEqual("", metaphone(""))
        self.assertEqual("", metaphone("123"))


class PhoneticDictionaryRepair(unittest.TestCase):
    def test_repairs_a_misheard_company_name(self):
        p = _personal(["Kubernetes"])
        self.assertEqual("deploy it on Kubernetes today",
                         p.apply_dictionary("deploy it on cuber netties today"))

    def test_repairs_split_product_name(self):
        p = _personal(["OpenFlow"])
        self.assertEqual("I use OpenFlow daily",
                         p.apply_dictionary("I use open flo daily"))

    def test_common_word_is_never_swallowed(self):
        # "Peter" and "people" are not homophones, but a short term whose key
        # collides with an everyday word must never win.
        p = _personal(["Kate"])
        self.assertEqual("the cat sat down", p.apply_dictionary("the cat sat down"))

    def test_counts_the_fixes_it_made(self):
        p = _personal(["Kubernetes"])
        p.apply_dictionary("cuber netties and cuber netties")
        self.assertEqual(2, p.last_fixes)

    def test_no_fixes_reported_when_clean(self):
        p = _personal(["Kubernetes"])
        p.apply_dictionary("nothing to see here")
        self.assertEqual(0, p.last_fixes)


class QuestionInference(unittest.TestCase):
    def test_aux_plus_subject(self):
        self.assertTrue(looks_like_question("did you send the report"))
        self.assertEqual("Did you send the report?",
                         fix_punctuation("did you send the report"))

    def test_wh_plus_aux(self):
        self.assertEqual("What is the status?", fix_punctuation("what is the status"))
        self.assertEqual("How many people are coming?",
                         fix_punctuation("how many people are coming"))

    def test_statements_stay_statements(self):
        self.assertEqual("We should ship it.", fix_punctuation("we should ship it"))
        self.assertEqual("How to reset the token.",
                         fix_punctuation("how to reset the token"))

    def test_existing_punctuation_is_respected(self):
        # The PRD's own example ends a "Can we ..." sentence with a period;
        # when ASR supplies punctuation we must not override it.
        self.assertEqual("Can we meet up on Friday at 3.",
                         fix_punctuation("Can we meet up on Friday at 3."))

    def test_only_the_final_sentence_is_examined(self):
        self.assertEqual("Ship it. Can you review it?",
                         fix_punctuation("Ship it. can you review it"))

    def test_full_pipeline(self):
        out = RuleBasedCleaner().clean("um did you, you know, send the invoice").text
        self.assertEqual("Did you send the invoice?", out)


class AudioConditioning(unittest.TestCase):
    def test_removes_dc_offset(self):
        biased = _tone(1.0) + 0.4
        out = condition(biased, RATE)
        self.assertAlmostEqual(0.0, float(np.mean(out)), places=4)

    def test_normalizes_quiet_audio(self):
        quiet = _tone(1.0, amplitude=0.02)
        out = condition(quiet, RATE)
        self.assertGreater(float(np.max(np.abs(out))), 0.8)

    def test_trims_leading_and_trailing_silence(self):
        silence = np.zeros(int(RATE * 0.8), dtype="float32")
        clip = np.concatenate([silence, _tone(1.0), silence])
        out = condition(clip, RATE)
        self.assertLess(len(out), len(clip))
        self.assertGreater(len(out), RATE * 0.9)   # the speech itself survives

    def test_does_not_amplify_pure_silence(self):
        out = condition(np.zeros(RATE, dtype="float32"), RATE)
        self.assertLess(float(np.max(np.abs(out))), 0.01)

    def test_empty_input_is_safe(self):
        self.assertEqual(0, len(condition(np.zeros(0, dtype="float32"), RATE)))

    def test_silence_detection(self):
        self.assertTrue(is_silent(np.zeros(RATE, dtype="float32"), RATE))
        self.assertTrue(is_silent(np.zeros(100, dtype="float32"), RATE))
        self.assertFalse(is_silent(_tone(1.0), RATE))



class QuietMicrophones(unittest.TestCase):
    """Regression: an absolute silence threshold discarded real speech.

    Measured on a real quiet mic: ambient RMS 0.000488 / peak 0.00827, while
    actual speech still came in under 0.012 RMS -- the gate's value. Mic gain
    varies by more than an order of magnitude between devices, so the gate
    must only catch genuinely dead audio.
    """

    def _speech_like(self, rms: float, seconds: float = 1.5):
        # Voiced speech is not a pure tone: a fundamental plus harmonics, with
        # an amplitude envelope that opens and closes like syllables.
        t = np.arange(int(RATE * seconds)) / RATE
        wave = (np.sin(2 * np.pi * 130 * t)
                + 0.5 * np.sin(2 * np.pi * 260 * t)
                + 0.25 * np.sin(2 * np.pi * 520 * t))
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
        signal = (wave * envelope).astype("float32")
        return (signal * (rms / float(np.sqrt(np.mean(signal**2))))).astype("float32")

    def test_quiet_speech_is_not_discarded(self):
        for rms in (0.004, 0.006, 0.010):
            with self.subTest(rms=rms):
                self.assertFalse(
                    is_silent(self._speech_like(rms), RATE),
                    f"speech at {rms} rms must reach the transcriber")

    def test_measured_ambient_is_still_discarded(self):
        rng = np.random.default_rng(7)
        ambient = (rng.normal(0, 0.00049, RATE) ).astype("float32")
        ambient = np.clip(ambient, -0.00827, 0.00827)
        self.assertTrue(is_silent(ambient, RATE))

    def test_quiet_speech_survives_conditioning(self):
        quiet = self._speech_like(0.005)
        out = condition(quiet, RATE)
        self.assertGreater(len(out), RATE * 0.5, "conditioning trimmed away the speech")
        self.assertGreater(float(np.max(np.abs(out))), 0.5, "quiet speech was not lifted")


class TransientRobustNormalization(unittest.TestCase):
    """A single click must not decide the gain for the whole clip.

    Peak normalization keys off the loudest sample, so one keyboard tap or mic
    pop pins the gain and leaves the actual speech as quiet as it started --
    which is the level ASR handles worst.
    """

    def _quiet_speech(self, rms: float = 0.005, seconds: float = 2.0):
        t = np.arange(int(RATE * seconds)) / RATE
        wave = np.sin(2 * np.pi * 130 * t) + 0.5 * np.sin(2 * np.pi * 260 * t)
        signal = wave.astype("float32")
        return (signal * (rms / float(np.sqrt(np.mean(signal**2))))).astype("float32")

    def test_lone_transient_does_not_suppress_speech(self):
        clip = self._quiet_speech()
        clip[RATE // 2] = 0.95          # one-sample pop
        out = condition(clip, RATE)
        speech_rms = float(np.sqrt(np.mean(out**2)))
        self.assertGreater(speech_rms, 0.05,
                           "speech should be lifted despite the transient")

    def test_never_clips(self):
        clip = self._quiet_speech()
        clip[RATE // 2] = 0.95
        out = condition(clip, RATE)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0,
                             "normalization must not push samples past full scale")

    def test_clean_audio_still_normalized_to_target(self):
        out = condition(self._quiet_speech(0.05), RATE)
        self.assertGreater(float(np.max(np.abs(out))), 0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
