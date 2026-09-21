"""Speech-to-text backends.

Free: Groq Whisper (cloud). Paid: OpenAI gpt-transcribe and Deepgram nova-3.
Local: Parakeet, Moonshine and faster-whisper.

Both take a float32 mono numpy array at the configured sample rate and return
plain text. The router in ``openflow/stt/router.py`` picks between them and
falls back on failure or quota exhaustion.
"""

from __future__ import annotations

import io
import json
import logging
import mimetypes
import urllib.error
import urllib.request
import uuid
import wave

from ..config import Config, api_key, languages

# Whisper names the language it detected; the settings speak ISO codes.
LANGUAGE_NAMES = {
    "en": "english", "bg": "bulgarian", "hr": "croatian", "cs": "czech",
    "da": "danish", "nl": "dutch", "et": "estonian", "fi": "finnish",
    "fr": "french", "de": "german", "el": "greek", "hu": "hungarian",
    "it": "italian", "lv": "latvian", "lt": "lithuanian", "mt": "maltese",
    "pl": "polish", "pt": "portuguese", "ro": "romanian", "ru": "russian",
    "sk": "slovak", "sl": "slovenian", "es": "spanish", "sv": "swedish",
    "uk": "ukrainian", "hi": "hindi", "gu": "gujarati", "mr": "marathi",
    "bn": "bengali", "pa": "punjabi", "ta": "tamil", "te": "telugu",
    "ur": "urdu", "zh": "chinese", "ja": "japanese", "ko": "korean",
    "ar": "arabic", "he": "hebrew", "tr": "turkish", "vi": "vietnamese",
    "id": "indonesian", "ms": "malay", "th": "thai", "tl": "tagalog",
    "fa": "persian", "sw": "swahili", "ca": "catalan", "no": "norwegian",
}
_CODES = {name: code for code, name in LANGUAGE_NAMES.items()}


def language_code(detected: str | None) -> str | None:
    """"english" or "en" -> "en"; None when unknown."""
    if not detected:
        return None
    value = detected.strip().lower()
    return value if value in LANGUAGE_NAMES else _CODES.get(value)

log = logging.getLogger(__name__)

USER_AGENT = "OpenFlow/0.1 (+https://github.com/openflow)"


class SttError(RuntimeError):
    pass


def vocabulary_terms(context=None, budget: int = 400) -> str:
    """Known terms, most specific first: the user's dictionary, then what is
    on screen (names in the email, files and identifiers in the project),
    then names the user has corrected by hand."""
    from ..corrections import shared as corrections
    from ..personalization import shared

    parts = [shared().vocabulary_hint(budget=budget)]
    if context is not None:
        parts.append(context.terms(budget=budget))
    parts.append(", ".join(
        entry.after for entry in corrections().active()
        if len(entry.after.split()) <= 2
    )[:200])
    joined = ", ".join(part for part in parts if part)
    if len(joined) > budget:
        joined = joined[:budget].rsplit(",", 1)[0]
    return joined


def vocabulary_prompt(context=None) -> str:
    """The recognition hint: keep it verbatim, and here are the hard words.

    Whisper reads its prompt as preceding context, which biases the decoder
    toward the spellings it contains. This is the only lever that fixes a name
    *before* it is mis-heard -- everything downstream is repair work on a word
    the model already got wrong, and repair cannot recover a name it has never
    seen. So both sources of known terms go in: the user's dictionary, and the
    names they have corrected by hand.

    The budget is chosen against Whisper's 224-token prompt window. ~400
    characters of comma-separated terms is roughly 100 tokens, which leaves
    the window comfortably clear while carrying several times the terms the
    previous 180-character budget allowed.
    """
    prompt = "Transcribe verbatim, including false starts and filler words."
    vocab = vocabulary_terms(context)
    if vocab:
        prompt += f" Vocabulary: {vocab}."
    return prompt


def to_wav_bytes(audio, sample_rate: int) -> bytes:
    """Encode float32 mono samples as 16-bit PCM WAV in memory."""
    import numpy as np

    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class GroqWhisper:
    """Groq's hosted Whisper -- the fast path in PRD section 4."""

    name = "groq"
    is_local = False
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    # Whisper decodes in 30-second windows and drops words at the seams of
    # its own long-form stitching. Pieces the router cuts at pauses, each
    # inside one window, lose nothing. See router.split_points.
    max_chunk_s = 28

    def __init__(self, config: Config) -> None:
        self.cfg = config

    @property
    def key(self) -> str | None:
        # Read per use, not once: the router is built at startup, and a key
        # saved in Settings afterwards should take effect straight away.
        return api_key("GROQ_API_KEY")

    def available(self) -> bool:
        return bool(self.key)

    last_language: str | None = None

    def transcribe(self, audio, sample_rate: int, context=None) -> str:
        if not self.key:
            raise SttError("GROQ_API_KEY is not set")
        wav = to_wav_bytes(audio, sample_rate)
        spoken = languages(self.cfg)
        fields = {
            "model": self.cfg.stt.groq_model,
            # verbose_json names the detected language, which is what lets a
            # multilingual speaker be held to the languages they chose.
            "response_format": "verbose_json" if len(spoken) != 1 else "json",
            "prompt": vocabulary_prompt(context),
        }
        if len(spoken) == 1:
            fields["language"] = spoken[0]
        data = self._post(fields, wav, _timeout(audio, sample_rate))
        detected = language_code(data.get("language")) or (spoken[0] if len(spoken) == 1 else None)
        if len(spoken) > 1 and detected not in spoken:
            # Whisper guessed a language the speaker never uses (short clips
            # and accents do this). Decode again, pinned to their primary one.
            log.info("groq detected %r, not one of %s; retrying as %s",
                     data.get("language"), spoken, spoken[0])
            fields["language"] = spoken[0]
            fields["response_format"] = "json"
            data = self._post(fields, wav, _timeout(audio, sample_rate))
            detected = spoken[0]
        self.last_language = detected
        return (data.get("text") or "").strip()

    def _post(self, fields: dict, wav: bytes, timeout: float) -> dict:
        body, content_type = _multipart(
            fields=fields, file_field="file", filename="audio.wav", file_bytes=wav)
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": content_type,
                # Cloudflare fronts this API and blocks urllib's default
                # user-agent outright (error 1010). Identify ourselves.
                "User-Agent": USER_AGENT,
            },
        )
        return _send(request, timeout)


def _timeout(audio, sample_rate: int) -> float:
    """20 s for a dictation, more for a long one: upload and decode scale
    with length, and a 4-minute chunk must not time out halfway."""
    seconds = len(audio) / sample_rate if sample_rate else 0.0
    return 20.0 + seconds / 4.0


def _send(request, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SttError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SttError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise SttError(f"malformed response: {exc}") from exc


class OpenAITranscribe:
    """OpenAI's hosted transcription (paid). gpt-transcribe takes a prompt,
    a keyword list and a set of languages -- all three of which are exactly
    the levers dictation needs: context, names, and multilingual speakers."""

    name = "openai"
    is_local = False
    paid = True
    url = "https://api.openai.com/v1/audio/transcriptions"

    def __init__(self, config: Config) -> None:
        self.cfg = config

    @property
    def key(self) -> str | None:
        return api_key("OPENAI_API_KEY")

    def available(self) -> bool:
        return bool(self.key)

    def transcribe(self, audio, sample_rate: int, context=None) -> str:
        if not self.key:
            raise SttError("OPENAI_API_KEY is not set")
        model = self.cfg.stt.openai_model
        spoken = languages(self.cfg)
        parts: list[tuple[str, str]] = [
            ("model", model),
            ("response_format", "json"),
            ("prompt", vocabulary_prompt(context)),
        ]
        if model == "gpt-transcribe":
            parts += [("languages[]", code) for code in spoken]
            terms = [t.strip() for t in vocabulary_terms(context, budget=800).split(",")]
            parts += [("keywords[]", t) for t in terms if t][:100]
        elif len(spoken) == 1:
            parts.append(("language", spoken[0]))
        body, content_type = _multipart_pairs(
            parts, file_field="file", filename="audio.wav",
            file_bytes=to_wav_bytes(audio, sample_rate))
        request = urllib.request.Request(
            self.url, data=body,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": content_type, "User-Agent": USER_AGENT})
        data = _send(request, _timeout(audio, sample_rate))
        return (data.get("text") or "").strip()


class Deepgram:
    """Deepgram nova-3 (paid): the fastest of the hosted recognizers, with
    key-term prompting for names and code-switching ("multi") for speakers of
    more than one language."""

    name = "deepgram"
    is_local = False
    paid = True
    url = "https://api.deepgram.com/v1/listen"

    def __init__(self, config: Config) -> None:
        self.cfg = config

    @property
    def key(self) -> str | None:
        return api_key("DEEPGRAM_API_KEY")

    def available(self) -> bool:
        return bool(self.key)

    def transcribe(self, audio, sample_rate: int, context=None) -> str:
        from urllib.parse import urlencode

        if not self.key:
            raise SttError("DEEPGRAM_API_KEY is not set")
        spoken = languages(self.cfg)
        query: list[tuple[str, str]] = [
            ("model", self.cfg.stt.deepgram_model),
            ("smart_format", "true"),
            ("punctuate", "true"),
            # Fillers stay in: cleanup decides what was a false start, and it
            # cannot see an "um" the recognizer already threw away.
            ("filler_words", "true"),
            ("language", spoken[0] if len(spoken) == 1 else "multi"),
        ]
        # Key terms share a 500-token budget; ~60 short terms fit well inside.
        terms = [t.strip() for t in vocabulary_terms(context, budget=600).split(",")]
        query += [("keyterm", t) for t in terms if t][:60]
        request = urllib.request.Request(
            f"{self.url}?{urlencode(query)}",
            data=to_wav_bytes(audio, sample_rate),
            headers={"Authorization": f"Token {self.key}",
                     "Content-Type": "audio/wav", "User-Agent": USER_AGENT})
        data = _send(request, _timeout(audio, sample_rate))
        try:
            return data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise SttError(f"unexpected response shape: {exc}") from exc


class FasterWhisper:
    """Local faster-whisper. The model is loaded once and kept warm."""

    name = "faster_whisper"
    is_local = True
    max_chunk_s = 28

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self._model = None

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def warm(self) -> None:
        """Load weights ahead of the first dictation so the fallback is not
        a 10-second surprise."""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        log.info("loading local whisper model %s", self.cfg.stt.local_model)
        self._model = WhisperModel(
            self.cfg.stt.local_model,
            device="auto",
            compute_type=self.cfg.stt.local_compute_type,
        )

    def transcribe(self, audio, sample_rate: int, context=None) -> str:
        try:
            self.warm()
        except Exception as exc:  # pragma: no cover - model download/runtime issues
            raise SttError(f"could not load local model: {exc}") from exc

        spoken = languages(self.cfg)
        segments, _info = self._model.transcribe(
            audio,
            language=spoken[0] if len(spoken) == 1 else None,
            beam_size=1,            # greedy: latency beats marginal accuracy here
            vad_filter=True,
            condition_on_previous_text=False,
            # Same vocabulary bias the cloud path gets. Names are exactly where
            # a small local model is weakest, so the hint matters most here.
            initial_prompt=vocabulary_prompt(context),
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class ParakeetOnnx:
    """NVIDIA Parakeet TDT via onnx-asr -- the default local backend.

    Chosen over faster-whisper because it needs only numpy + onnxruntime (no
    PyTorch, no ctranslate2, no ffmpeg), runs well on CPU, and is what the
    wider open-source dictation ecosystem has standardized on for local
    English. See docs/research-2026-08.md.
    """

    name = "parakeet_onnx"
    is_local = True
    # Memory grows with input length; a minute is comfortable on any laptop.
    max_chunk_s = 60

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self._model = None

    def available(self) -> bool:
        try:
            import onnx_asr  # noqa: F401
        except ImportError:
            return False
        return True

    def warm(self) -> None:
        if self._model is not None:
            return
        import onnx_asr

        from .providers import describe, select_providers

        providers = select_providers(self.cfg.stt.device)
        log.info("loading %s on %s (first run downloads weights)",
                 self.cfg.stt.parakeet_model, describe(providers))
        try:
            self._model = onnx_asr.load_model(
                self.cfg.stt.parakeet_model,
                quantization=self.cfg.stt.parakeet_quantization or None,
                providers=providers,
            )
        except Exception as exc:
            # A GPU provider can be installed and still fail to initialize --
            # missing cuDNN, a driver too old, a card already full. Falling
            # back beats leaving the user with no local transcription.
            if providers[:1] == ["CPUExecutionProvider"]:
                raise
            log.warning("%s failed (%s); retrying on CPU", providers[0], exc)
            self._model = onnx_asr.load_model(
                self.cfg.stt.parakeet_model,
                quantization=self.cfg.stt.parakeet_quantization or None,
                providers=["CPUExecutionProvider"],
            )

    def transcribe(self, audio, sample_rate: int, context=None) -> str:
        try:
            self.warm()
        except Exception as exc:
            raise SttError(f"could not load Parakeet: {exc}") from exc

        import numpy as np

        waveform = np.asarray(audio, dtype=np.float32)
        try:
            text = self._model.recognize(waveform, sample_rate=sample_rate)
        except Exception as exc:
            raise SttError(f"parakeet recognition failed: {exc}") from exc
        # Some adapters return a result object rather than a bare string.
        return (text if isinstance(text, str) else getattr(text, "text", str(text))).strip()


class Moonshine:
    """Moonshine -- tuned for short utterances, which is what dictation is.

    Whisper pads every clip to a fixed 30-second window, so a three-second
    phrase costs the same as a twenty-five-second one. Moonshine's compute
    tracks actual audio length, which suits push-to-talk.
    """

    name = "moonshine"
    is_local = True
    max_chunk_s = 30

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self._transcriber = None

    def available(self) -> bool:
        try:
            import moonshine_voice  # noqa: F401
        except ImportError:
            return False
        return True

    def warm(self) -> None:
        if self._transcriber is not None:
            return
        import moonshine_voice as mv
        from moonshine_voice.transcriber import Transcriber

        language = self.cfg.stt.language or "en"
        arch = getattr(mv.ModelArch, self.cfg.stt.moonshine_arch, None)
        log.info("loading moonshine model (%s, %s)", language, self.cfg.stt.moonshine_arch)
        model_name, resolved_arch = mv.get_model_for_language(language, arch)
        self._transcriber = Transcriber(
            mv.get_model_path(model_name), model_arch=resolved_arch
        )

    def transcribe(self, audio, sample_rate: int, context=None) -> str:
        try:
            self.warm()
        except Exception as exc:
            raise SttError(f"could not load Moonshine: {exc}") from exc

        import numpy as np

        samples = np.asarray(audio, dtype=np.float32).tolist()
        stream = None
        try:
            stream = self._transcriber.create_stream()
            # The stream must be started before audio is accepted -- otherwise
            # its VAD is inactive and add_audio fails.
            stream.start()
            stream.add_audio(samples, sample_rate)
            transcript = stream.update_transcription(
                self._transcriber.MOONSHINE_FLAG_FORCE_UPDATE
            )
        except Exception as exc:
            raise SttError(f"moonshine recognition failed: {exc}") from exc
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    log.debug("moonshine stream cleanup failed", exc_info=True)

        return " ".join(line.text.strip() for line in transcript.lines).strip()


def _multipart(*, fields: dict[str, str], file_field: str, filename: str,
               file_bytes: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
        )
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n".encode()
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _multipart_pairs(pairs: list[tuple[str, str]], *, file_field: str, filename: str,
                     file_bytes: bytes) -> tuple[bytes, str]:
    """Like _multipart, but a field may repeat -- ``languages[]=en``,
    ``languages[]=fr`` -- which a dict cannot express."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for key, value in pairs:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n".encode()
            + str(value).encode("utf-8") + b"\r\n"
        )
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n".encode()
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def build_engine(name: str, config: Config):
    if name == "groq":
        return GroqWhisper(config)
    if name == "openai":
        return OpenAITranscribe(config)
    if name == "deepgram":
        return Deepgram(config)
    if name in ("parakeet_onnx", "parakeet"):
        return ParakeetOnnx(config)
    if name == "moonshine":
        return Moonshine(config)
    if name in ("faster_whisper", "local"):
        return FasterWhisper(config)
    raise ValueError(f"unknown STT engine: {name}")
