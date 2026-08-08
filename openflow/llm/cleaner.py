"""LLM-backed cleaner with quota-aware fallback (PRD section 4).

Chain: cloud -> local -> deterministic rules. The rules pass never fails, so
dictation always produces text even with no network and no Ollama running.
"""

from __future__ import annotations

import logging
import time

from ..config import Config
from ..text.cleaner import CleanResult, RuleBasedCleaner
from .base import ProviderError
from .providers import build_provider, system_prompt_for
from .quota import QuotaLedger

log = logging.getLogger(__name__)


class LLMCleaner:
    """Implements the same ``Cleaner`` protocol as ``RuleBasedCleaner``, so the
    test harness can score it against the identical golden corpus."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        provider: str | None = None,
        quota: QuotaLedger | None = None,
    ) -> None:
        self.config = config or Config.load()
        self.quota = quota or QuotaLedger()
        self.rules = RuleBasedCleaner()
        # An explicit provider (used by the harness) pins the chain to one backend.
        self.chain = [provider] if provider else list(self.config.llm.backends)
        self.name = provider or "llm"

    def clean(self, raw: str) -> CleanResult:
        started = time.perf_counter()
        text = raw.strip()
        if not text:
            return CleanResult(text="", raw=raw, engine="noop")

        prepass = self.rules.clean(text) if self.config.llm.rules_prepass else None
        candidate = prepass.text if prepass else text

        if not self.config.llm.enabled:
            result = prepass or self.rules.clean(text)
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        for name in self.chain:
            if name == "rules":
                result = prepass or self.rules.clean(text)
                result.latency_ms = (time.perf_counter() - started) * 1000
                return result

            limit = self.config.llm.daily_limits.get(name)
            if not self.quota.has_headroom(name, limit):
                log.info("%s over daily free-tier limit; falling through", name)
                continue

            try:
                provider = build_provider(name, self.config)
            except ValueError:
                log.warning("unknown LLM backend %r in config; skipping", name)
                continue

            if not provider.available():
                log.info("%s unavailable; falling through", name)
                continue

            try:
                # The LLM sees the rules-cleaned text: fewer tokens, and a
                # small local model has less room to wander.
                out = provider.complete(system_prompt_for(provider), candidate)
            except ProviderError as exc:
                log.warning("%s failed (%s); falling through", name, exc)
                if limit and "429" in str(exc):
                    self.quota.exhaust(name, limit)
                continue

            self.quota.record(name)
            return CleanResult(
                text=out,
                raw=raw,
                retractions=prepass.retractions if prepass else [],
                fillers_removed=prepass.fillers_removed if prepass else [],
                engine=name,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # Every backend declined -- deterministic output is the floor.
        result = prepass or self.rules.clean(text)
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result
