"""LLM backends.

Free: Groq and the Google AI Studio tier, plus local Ollama -- plain HTTP
through ``urllib``, no SDK and no import cost on startup, which matters
because the hotkey listener has to be responsive the moment the app launches.

Paid (models.tier == "pro"): Claude through the official ``anthropic`` SDK,
imported only when a Claude cleanup actually runs, and OpenAI over HTTP.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from ..config import Config, api_key
from .base import Provider, ProviderError, sanitize
from .prompts import FEW_SHOT, build_system_prompt, context_section

# Probing a dead Ollama costs a full connect timeout, and it sits directly on
# the dictation path. Cache the answer: a miss is re-checked occasionally so a
# newly started Ollama gets picked up without a restart, a hit is trusted for
# longer since a running server rarely vanishes mid-session.
_MISS_TTL_S = 30.0
_HIT_TTL_S = 300.0

log = logging.getLogger(__name__)


def _post(url: str, payload: dict, timeout: float, headers: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "OpenFlow/0.1 (+https://github.com/openflow)",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise ProviderError(f"malformed response: {exc}") from exc


def output_budget(text: str) -> int:
    """Tokens to allow for the edited text: a little over the input's own
    length (~0.75 words per token), never less than the old fixed 512."""
    return max(512, int(len(text.split()) * 2.2) + 64)


def scaled_timeout(base: float, text: str) -> float:
    """The configured timeout suits a sentence; a long dictation gets one
    extra second per ~60 words of output to write."""
    return base + len(text.split()) / 60.0


class OllamaProvider:
    """Local Llama 3.1 8B / Gemma 2 9B via the Ollama chat endpoint."""

    name = "ollama"
    is_local = True

    # Class-level so the cache survives the per-dictation provider rebuild.
    # Stores (checked_at, resolved_model_or_None).
    _probe: tuple[float, str | None] | None = None

    def __init__(self, config: Config) -> None:
        self.cfg = config.llm
        self.host = self.cfg.ollama_host.rstrip("/")
        self.model = self.cfg.ollama_model

    def _resolve(self, models: list[dict]) -> str | None:
        """Pick the best installed model for transcript editing.

        Exact configured model wins. Otherwise any text model will do a better
        job than reporting "unavailable" at a user who has Ollama running --
        but vision-tuned models go last, since they are trained for captioning
        and tend to describe rather than edit.
        """
        names = [m.get("name", "") for m in models if m.get("name")]
        if not names:
            return None

        wanted = self.model
        family = wanted.split(":")[0]
        for name in names:
            if name == wanted:
                return name
        for name in names:
            if name.split(":")[0] == family:
                return name

        def is_vision(entry: dict) -> bool:
            capabilities = entry.get("capabilities") or []
            if "vision" in capabilities:
                return True
            return "vision" in entry.get("name", "").lower()

        text_models = [m.get("name") for m in models if not is_vision(m) and m.get("name")]
        return text_models[0] if text_models else names[0]

    def available(self) -> bool:
        now = time.monotonic()
        cached = OllamaProvider._probe
        if cached is not None:
            checked_at, resolved = cached
            if now - checked_at < (_HIT_TTL_S if resolved else _MISS_TTL_S):
                if resolved:
                    self.model = resolved
                return bool(resolved)

        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=0.6) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
            resolved = self._resolve(tags.get("models", []))
        except Exception:
            resolved = None

        if resolved and resolved != self.cfg.ollama_model:
            log.info("ollama: %s not installed; using %s",
                     self.cfg.ollama_model, resolved)
        if resolved:
            self.model = resolved
        OllamaProvider._probe = (now, resolved)
        return bool(resolved)

    def warm(self, timeout_s: float) -> None:
        """Pull the weights into memory. Takes its own timeout because a cold
        load is an order of magnitude slower than a warm request, and this runs
        where nobody is waiting on it."""
        _post(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 1, "temperature": 0.0},
            },
            timeout=timeout_s,
        )

    def complete(self, system: str, user: str, *, strict: bool = True,
                 allowed: frozenset[str] = frozenset()) -> str:
        messages: list[dict] = [{"role": "system", "content": system}]
        # Few-shot pairs keep small models in editing mode (PRD section 5).
        # Two is the useful minimum: one correction, one leave-it-alone. Every
        # extra pair is prompt tokens paid on the dictation path.
        for example_in, example_out in FEW_SHOT[:2]:
            messages.append({"role": "user", "content": example_in})
            messages.append({"role": "assistant", "content": example_out})
        messages.append({"role": "user", "content": user})

        data = _post(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                # Keep the weights resident: a cold load costs ~6 s, which
                # would land on whichever dictation follows an idle spell.
                "keep_alive": "30m",
                "options": {
                    "temperature": self.cfg.temperature,
                    "top_p": 0.9,
                    "num_predict": output_budget(user),
                    # Deterministic edits: no creative sampling.
                    "repeat_penalty": 1.0,
                },
            },
            timeout=scaled_timeout(self.cfg.timeout_s, user),
        )
        return sanitize(data.get("message", {}).get("content", ""), original=user,
                        strict=strict, allowed=allowed)


class GroqProvider:
    """Groq's OpenAI-compatible chat endpoint, on the key already used for STT.

    This is the fast path for cleanup. Measured against the same one-line
    retraction case: gpt-oss-20b 237 ms, gpt-oss-120b 477 ms, Gemini
    flash-lite 585 ms, local llama3.1:8b ~2,700 ms. Speed is the whole
    argument -- cleanup sits between the user releasing the hotkey and the
    text appearing, so every millisecond here is one they wait through.
    """

    name = "groq"
    is_local = False
    # 20B is small enough to wander into rewriting if left unconstrained, so it
    # gets the editing guardrails and few-shot pairs that the local models get,
    # despite being a cloud backend.
    small = True
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, config: Config) -> None:
        self.cfg = config.llm
        self.model = self.cfg.groq_model
        self.key = api_key("GROQ_API_KEY")

    def available(self) -> bool:
        return bool(self.key)

    def verify(self) -> str | None:
        """One real round-trip, for the Settings page. ``available`` only says
        the key exists; this says the model answers to it."""
        if not self.key:
            return "GROQ_API_KEY is not set"
        try:
            self.complete("Reply with the single word: ok", "ok", strict=False)
        except ProviderError as exc:
            reason = " ".join(str(exc).split())
            if "404" in reason or "model_not_found" in reason:
                return f"model {self.model!r} is not available to this key"
            if "429" in reason:
                return "daily quota exhausted"
            return reason[:110]
        return None

    def complete(self, system: str, user: str, *, strict: bool = True,
                 allowed: frozenset[str] = frozenset()) -> str:
        if not self.key:
            raise ProviderError("GROQ_API_KEY is not set")
        messages: list[dict] = [{"role": "system", "content": system}]
        for example_in, example_out in FEW_SHOT:
            messages.append({"role": "user", "content": example_in})
            messages.append({"role": "assistant", "content": example_out})
        messages.append({"role": "user", "content": user})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": output_budget(user),
            "stream": False,
        }
        # gpt-oss models reason before answering. Editing a sentence needs none
        # of it, and the tokens are pure latency on the dictation path.
        if "gpt-oss" in self.model:
            payload["reasoning_effort"] = self.cfg.groq_reasoning_effort

        data = _post(
            self.url,
            payload,
            timeout=scaled_timeout(self.cfg.timeout_s, user),
            headers={"Authorization": f"Bearer {self.key}"},
        )
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected response shape: {exc}") from exc
        return sanitize(text or "", original=user, strict=strict, allowed=allowed)


class GeminiProvider:
    """Google AI Studio free tier (Gemini 1.5 Flash), per PRD section 4."""

    name = "gemini"
    is_local = False
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, config: Config) -> None:
        self.cfg = config.llm
        self.model = self.cfg.gemini_model
        self.key = api_key("GEMINI_API_KEY") or api_key("GOOGLE_AI_STUDIO_KEY")

    def available(self) -> bool:
        # Deliberately offline: this runs on every UI refresh. It answers "is
        # this configured", not "does it work" -- see verify().
        return bool(self.key)

    def verify(self) -> str | None:
        """One real round-trip. None when the backend genuinely works, else a
        short reason. available() cannot see a retired model, which is how a
        config pinned to gemini-1.5-flash kept reporting ready while every
        cleanup silently fell through to the rules pass."""
        if not self.key:
            return "GEMINI_API_KEY is not set"
        try:
            self.complete("Reply with the single word: ok", "ok", strict=False)
        except ProviderError as exc:
            reason = " ".join(str(exc).split())
            if "404" in reason:
                return f"model {self.model!r} is not available to this key"
            if "429" in reason:
                return "daily quota exhausted"
            return reason[:110]
        return None

    def complete(self, system: str, user: str, *, strict: bool = True,
                 allowed: frozenset[str] = frozenset()) -> str:
        if not self.key:
            raise ProviderError("GEMINI_API_KEY is not set")
        data = _post(
            f"{self.endpoint}/{self.model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": self.cfg.temperature,
                    "maxOutputTokens": output_budget(user),
                    "candidateCount": 1,
                },
            },
            timeout=scaled_timeout(self.cfg.timeout_s, user),
            headers={"x-goog-api-key": self.key},
        )
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected response shape: {exc}") from exc
        return sanitize(text, original=user, strict=strict, allowed=allowed)


class OpenAIProvider:
    """OpenAI chat completions (paid). The GPT-5.6 family reasons by default;
    editing a sentence needs none of it, so reasoning is turned off -- every
    reasoning token is latency the speaker waits through."""

    name = "openai"
    is_local = False
    small = False
    paid = True
    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, config: Config) -> None:
        self.cfg = config.llm
        self.model = self.cfg.openai_model
        self.key = api_key("OPENAI_API_KEY")

    def available(self) -> bool:
        return bool(self.key)

    def verify(self) -> str | None:
        if not self.key:
            return "OPENAI_API_KEY is not set"
        try:
            self.complete("Reply with the single word: ok", "ok", strict=False)
        except ProviderError as exc:
            return " ".join(str(exc).split())[:110]
        return None

    def complete(self, system: str, user: str, *, strict: bool = True,
                 allowed: frozenset[str] = frozenset()) -> str:
        if not self.key:
            raise ProviderError("OPENAI_API_KEY is not set")
        messages: list[dict] = [{"role": "system", "content": system}]
        for example_in, example_out in FEW_SHOT:
            messages.append({"role": "user", "content": example_in})
            messages.append({"role": "assistant", "content": example_out})
        messages.append({"role": "user", "content": user})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": output_budget(user),
        }
        if self.model.startswith(("gpt-5", "gpt-6", "o")):
            payload["reasoning_effort"] = "none" if self.model.startswith("gpt-5.6") else "low"
        data = _post(self.url, payload, timeout=scaled_timeout(self.cfg.timeout_s, user),
                     headers={"Authorization": f"Bearer {self.key}"})
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected response shape: {exc}") from exc
        return sanitize(text or "", original=user, strict=strict, allowed=allowed)


class AnthropicProvider:
    """Claude through the official SDK (paid).

    Defaults to claude-opus-5 at effort "low": cleanup is a light edit, and
    thinking is latency. The server-side refusal fallback is on, so a request
    one model declines is re-run on another inside the same call instead of
    costing the speaker their dictation.
    """

    name = "anthropic"
    is_local = False
    small = False
    paid = True

    def __init__(self, config: Config) -> None:
        self.cfg = config.llm
        self.model = self.cfg.anthropic_model
        self.key = api_key("ANTHROPIC_API_KEY")

    def available(self) -> bool:
        return bool(self.key)

    def verify(self) -> str | None:
        if not self.key:
            return "ANTHROPIC_API_KEY is not set"
        try:
            self.complete("Reply with the single word: ok", "ok", strict=False)
        except ProviderError as exc:
            return " ".join(str(exc).split())[:110]
        return None

    def complete(self, system: str, user: str, *, strict: bool = True,
                 allowed: frozenset[str] = frozenset()) -> str:
        if not self.key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("the anthropic package is not installed") from exc

        messages: list[dict] = []
        for example_in, example_out in FEW_SHOT:
            messages.append({"role": "user", "content": example_in})
            messages.append({"role": "assistant", "content": example_out})
        messages.append({"role": "user", "content": user})

        client = anthropic.Anthropic(
            api_key=self.key, max_retries=1,
            timeout=scaled_timeout(self.cfg.timeout_s, user) + 4.0)
        request = dict(
            model=self.model,
            max_tokens=max(1024, output_budget(user)),
            # The system prompt and few-shot pairs are identical on every
            # dictation; cache them.
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config={"effort": self.cfg.anthropic_effort},
        )
        try:
            if self.model in ("claude-opus-5", "claude-fable-5-1"):
                response = client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default", **request)
            else:
                response = client.messages.create(**request)
        except anthropic.RateLimitError as exc:
            raise ProviderError(f"HTTP 429: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"HTTP {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"connection failed: {exc}") from exc

        if response.stop_reason == "refusal":
            raise ProviderError("the model declined this request")
        if response.stop_reason == "max_tokens":
            raise ProviderError("completion cut off at max_tokens")
        text = "".join(block.text for block in response.content if block.type == "text")
        return sanitize(text, original=user, strict=strict, allowed=allowed)


def build_provider(name: str, config: Config) -> Provider:
    if name == "anthropic":
        return AnthropicProvider(config)
    if name == "openai":
        return OpenAIProvider(config)
    if name == "ollama":
        return OllamaProvider(config)
    if name == "gemini":
        return GeminiProvider(config)
    if name == "groq":
        return GroqProvider(config)
    raise ValueError(f"unknown LLM provider: {name}")


def system_prompt_for(provider: Provider, context=None, *, category: str = "",
                      language: str | None = None) -> str:
    # "small" is the real question the supplement answers -- will this model
    # rewrite when asked to edit? Local models all do; so does a 20B cloud one.
    # Default to is_local so providers that never set it keep their behaviour.
    small = getattr(provider, "small", provider.is_local)
    prompt = build_system_prompt(local=small)
    from ..personalization import shared

    personal = shared()
    parts = [prompt]
    # Names the speaker uses and corrections they have taught us. Both are
    # about the same failure: the model "fixing" a term it does not recognise.
    names = personal.vocabulary_hint(budget=400)
    if names:
        parts.append(
            "KNOWN TERMS: the following are spelled correctly and must be kept "
            f"exactly as written: {names}."
        )
    from ..corrections import shared as corrections

    taught = corrections().prompt_hint()
    if taught:
        parts.append(taught)
    section = context_section(context, category=category, language=language, large=not small)
    if section:
        parts.append(section)
    return "\n\n".join(parts)
