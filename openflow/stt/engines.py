"""Speech-to-text backends: Groq Whisper (cloud) and faster-whisper (local).

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

from ..config import Config, api_key

log = logging.getLogger(__name__)

USER_AGENT = "OpenFlow/0.1 (+https://github.com/openflow)"


class SttError(RuntimeError):
    pass


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

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.key = api_key("GROQ_API_KEY")

    def available(self) -> bool:
        return bool(self.key)

    def transcribe(self, audio, sample_rate: int) -> str:
        if not self.key:
            raise SttError("GROQ_API_KEY is not set")
        wav = to_wav_bytes(audio, sample_rate)
        # Whisper treats the prompt as a style/vocabulary hint. Two jobs here:
        # keep it verbatim (cleanup is the LLM's job), and bias recognition
        # toward the user's dictionary terms so "Groq" doesn't become "grok".
        prompt = "Transcribe verbatim, including false starts and filler words."
        from ..personalization import shared

        vocab = shared().vocabulary_hint()
        if vocab:
            prompt += f" Vocabulary: {vocab}."
        body, content_type = _multipart(
            fields={
                "model": self.cfg.stt.groq_model,
                "response_format": "json",
                "language": self.cfg.stt.language,
                "prompt": prompt,
            },
            file_field="file",
            filename="audio.wav",
            file_bytes=wav,
        )
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
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise SttError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SttError(str(exc)) from exc
        return (data.get("text") or "").strip()


class FasterWhisper:
    """Local faster-whisper. The model is loaded once and kept warm."""

    name = "faster_whisper"
    is_local = True

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

    def transcribe(self, audio, sample_rate: int) -> str:
        try:
            self.warm()
        except Exception as exc:  # pragma: no cover - model download/runtime issues
            raise SttError(f"could not load local model: {exc}") from exc

        segments, _info = self._model.transcribe(
            audio,
            language=self.cfg.stt.language,
            beam_size=1,            # greedy: latency beats marginal accuracy here
            vad_filter=True,
            condition_on_previous_text=False,
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

        log.info("loading %s (first run downloads weights)", self.cfg.stt.parakeet_model)
        self._model = onnx_asr.load_model(
            self.cfg.stt.parakeet_model,
            quantization=self.cfg.stt.parakeet_quantization or None,
        )

    def transcribe(self, audio, sample_rate: int) -> str:
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

    def transcribe(self, audio, sample_rate: int) -> str:
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


def build_engine(name: str, config: Config):
    if name == "groq":
        return GroqWhisper(config)
    if name in ("parakeet_onnx", "parakeet"):
        return ParakeetOnnx(config)
    if name == "moonshine":
        return Moonshine(config)
    if name in ("faster_whisper", "local"):
        return FasterWhisper(config)
    raise ValueError(f"unknown STT engine: {name}")
