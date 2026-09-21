"""LLM-backed cleaner with quota-aware fallback (PRD section 4).

Chain: cloud -> local -> deterministic rules. The rules pass never fails, so
dictation always produces text even with no network and no Ollama running.
"""

from __future__ import annotations

import logging
import time

from ..config import Config, llm_chain, unlimited
from ..text.cleaner import (
    CleanResult, RuleBasedCleaner, finish_model_output, prepare_for_model,
)
from .base import ProviderError
from .providers import build_provider, system_prompt_for
from .quota import QuotaLedger, is_daily_exhaustion

log = logging.getLogger(__name__)

# Groq serves both transcription and cleanup, on separate free-tier
# allowances. The ledger is keyed by name, so the cleanup backend needs its own
# key -- otherwise a day of cleanup requests would exhaust the counter that
# gates cloud *transcription*, which is the more valuable of the two.
QUOTA_KEY = {"groq": "groq_chat"}


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
        self._pinned = [provider] if provider else None
        self.name = provider or "llm"

    @property
    def chain(self) -> list[str]:
        # Read per dictation: switching to the paid tier in Settings takes
        # effect on the next one.
        return self._pinned or llm_chain(self.config)

    def clean(self, raw: str, *, context=None, category: str = "",
              language: str | None = None) -> CleanResult:
        """Clean ``raw``. ``context`` is what was on screen (names, the open
        project), ``category`` the Wispr app category the text is headed for,
        ``language`` the ISO code the transcriber detected."""
        started = time.perf_counter()
        text = raw.strip()
        if not text:
            return CleanResult(text="", raw=raw, engine="noop")

        # The rules pass is English: its filler, pivot and number lexicons
        # would mangle anything else. Other languages go to the model as
        # spoken, and come back as spoken if every model declines.
        if language and language != "en":
            return self._clean_foreign(text, raw, context, category, language, started)

        # The rules pass still runs first: it is the answer when every model
        # declines, and its retraction analysis feeds the uncertainty gate.
        # But its *output* is not what the model edits -- see
        # prepare_for_model for why a model must see the words the rules
        # would have deleted.
        prepass = self.rules.clean(text) if self.config.llm.rules_prepass else None
        candidate = prepare_for_model(text)

        if not self.config.llm.enabled:
            result = prepass or self.rules.clean(text)
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        # Selective invocation: on ordinary dictation the rules pass already
        # produces what every model produces, so calling one is pure latency.
        # Spend it only where the deterministic pass admitted it was guessing.
        if self.config.llm.only_when_uncertain and prepass is not None \
                and not prepass.uncertain:
            log.debug("rules pass was confident; skipping the LLM")
            prepass.latency_ms = (time.perf_counter() - started) * 1000
            return prepass

        for name in self.chain:
            if name == "rules":
                result = prepass or self.rules.clean(text)
                result.latency_ms = (time.perf_counter() - started) * 1000
                return result

            quota_key = QUOTA_KEY.get(name, name)
            limit = (None if unlimited(self.config, quota_key)
                     else self.config.llm.daily_limits.get(quota_key))
            if not self.quota.has_headroom(quota_key, limit):
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

            allowed = frozenset(context.allowed_words()) if context is not None else frozenset()
            try:
                out = provider.complete(
                    system_prompt_for(provider, context, category=category),
                    candidate, allowed=allowed)
            except ProviderError as exc:
                log.warning("%s failed (%s); falling through", name, exc)
                if limit and "429" in str(exc) and is_daily_exhaustion(str(exc)):
                    self.quota.exhaust(quota_key, limit)
                continue

            self.quota.record(quota_key)
            out, symbol_fixes = finish_model_output(out)
            return CleanResult(
                text=out,
                raw=raw,
                retractions=prepass.retractions if prepass else [],
                fillers_removed=prepass.fillers_removed if prepass else [],
                repetitions_collapsed=prepass.repetitions_collapsed if prepass else [],
                autofixes=symbol_fixes,
                engine=name,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # Every backend declined -- deterministic output is the floor.
        result = prepass or self.rules.clean(text)
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    def _clean_foreign(self, text: str, raw: str, context, category: str,
                       language: str, started: float) -> CleanResult:
        if self.config.llm.enabled:
            allowed = frozenset(context.allowed_words()) if context is not None else frozenset()
            for name in self.chain:
                if name == "rules":
                    break
                quota_key = QUOTA_KEY.get(name, name)
                limit = (None if unlimited(self.config, quota_key)
                         else self.config.llm.daily_limits.get(quota_key))
                if not self.quota.has_headroom(quota_key, limit):
                    continue
                try:
                    provider = build_provider(name, self.config)
                except ValueError:
                    continue
                if not provider.available():
                    continue
                try:
                    out = provider.complete(
                        system_prompt_for(provider, context, category=category,
                                          language=language),
                        text, allowed=allowed)
                except ProviderError as exc:
                    log.warning("%s failed on %s (%s); falling through", name, language, exc)
                    continue
                self.quota.record(quota_key)
                return CleanResult(text=out, raw=raw, engine=name,
                                   latency_ms=(time.perf_counter() - started) * 1000)
        return CleanResult(text=text, raw=raw, engine="none",
                           latency_ms=(time.perf_counter() - started) * 1000)
