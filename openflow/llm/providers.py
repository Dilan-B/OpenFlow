"""LLM backends: local Ollama and the Google AI Studio free tier.

Both speak plain HTTP through ``urllib`` -- no SDK, no extra dependency, and no
import cost on startup, which matters because the hotkey listener has to be
responsive the moment the app launches.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from ..config import Config, api_key
from .base import Provider, ProviderError, sanitize
from .prompts import FEW_SHOT, build_system_prompt

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

    def complete(self, system: str, user: str, *, strict: bool = True) -> str:
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
                    "num_predict": 256,
                    # Deterministic edits: no creative sampling.
                    "repeat_penalty": 1.0,
                },
            },
            timeout=self.cfg.timeout_s,
        )
        return sanitize(data.get("message", {}).get("content", ""), original=user, strict=strict)


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
        return bool(self.key)

    def complete(self, system: str, user: str, *, strict: bool = True) -> str:
        if not self.key:
            raise ProviderError("GEMINI_API_KEY is not set")
        data = _post(
            f"{self.endpoint}/{self.model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": self.cfg.temperature,
                    "maxOutputTokens": 1024,
                    "candidateCount": 1,
                },
            },
            timeout=self.cfg.timeout_s,
            headers={"x-goog-api-key": self.key},
        )
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected response shape: {exc}") from exc
        return sanitize(text, original=user, strict=strict)


def build_provider(name: str, config: Config) -> Provider:
    if name == "ollama":
        return OllamaProvider(config)
    if name == "gemini":
        return GeminiProvider(config)
    raise ValueError(f"unknown LLM provider: {name}")


def system_prompt_for(provider: Provider) -> str:
    prompt = build_system_prompt(local=provider.is_local)
    from ..personalization import shared

    style = shared().style_instruction()
    return f"{prompt}\n\n{style}" if style else prompt
