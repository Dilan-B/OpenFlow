# Transcription research — August 2026

PRD §5 asks for an active scan of open-source indexes for newer transcription
optimizations and zero-cost APIs. This is that scan, with the caveat that most
"best STT of 2026" pages are vendor SEO content — latency and WER numbers below
are **vendor-reported unless they come from the HuggingFace Open ASR
Leaderboard**, and none were reproduced locally.

## TL;DR — all four recommendations are now implemented

| # | Change | Why | Where |
|---|---|---|---|
| 1 | `onnx-asr` + `parakeet-tdt-0.6b-v3` is now the default local STT backend | Lighter install, faster on CPU; faster-whisper demoted to third in the chain | `stt/engines.py::ParakeetOnnx` |
| 2 | Moonshine added as a selectable backend | Built for 2–10 s utterances, which is exactly our workload; Whisper pays a fixed 30 s-window cost regardless | `stt/engines.py::Moonshine` |
| 3 | `daily_limits.groq` → 2000, plus a rolling hourly audio-seconds budget | Groq's real free tier is 2,000 req/day **and** 7,200 audio-seconds/hour | `llm/quota.py` |
| 4 | Transcript-containment validator on every LLM completion | Full-rewrite models invent entities; length checks cannot catch a same-length substitution | `llm/base.py::check_containment` |

## 1. Local STT — Parakeet via ONNX beats faster-whisper for this workload

[`onnx-asr`](https://github.com/istupakov/onnx-asr) (0.12.0, July 2026) is a
pure-Python ASR runtime: NumPy + onnxruntime, **no PyTorch, no ctranslate2, no
ffmpeg**. Supports Python 3.10–3.14, Windows/Linux/macOS, and loads NVIDIA
NeMo Parakeet/Canary, Whisper, and Zipformer weights.

```python
import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
model.recognize("test.wav")
```

[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
is 600M params, CC-BY-4.0, 25 European languages with automatic language
detection, 6.34% average WER on the Open ASR Leaderboard, and among the highest
throughput of any multilingual open-weight model.

**Verified in this repo's venv** (`pip install --dry-run`, Python 3.14):

- `onnx-asr[cpu,hub]` → onnxruntime 1.28.0 + huggingface-hub. No heavy deps.
- `faster-whisper` → also resolves (1.2.1 + ctranslate2 4.8.1 + av 18.0.0), so
  the current backend is not *broken* — just heavier.

Note `onnx-asr` pins away from onnxruntime 1.24.1 and 1.25.x/1.26.0 (known
incompatibilities); the resolver picked 1.28.0.

**Implemented** as `ParakeetOnnx`; the default chain is now
`groq → parakeet_onnx → faster_whisper`.

### Measured on this machine (CPU, no GPU)

| | fp32 | int8 |
|---|---|---|
| encoder download | 2,322 MB | 622 MB |
| transcribe 3.2 s of speech | — | **148 ms** (~21× realtime) |
| first call (incl. model load) | — | ~47 s, once, on a background thread |

The fp32 encoder being a 2.3 GB download is the wrong trade for a hotkey tool,
so `stt.parakeet_quantization` defaults to `int8`. Set it to `""` for fp32.

Accuracy on the PRD's canonical sentence, transcribed from synthesized speech:

```
heard   : 'Can we meet up on Tuesday at 5 or actually, can we meet up on Friday at 3?'
cleaned : 'Can we meet up on Friday at 3?'
```

## 2. Moonshine — the right shape for push-to-talk

[Moonshine](https://github.com/moonshine-ai/moonshine) (MIT, `pip install
moonshine-voice`, ONNX runtime) is 26M–245M params and is built specifically
for live speech with **flexible input windows**. That last property is the
interesting one for us: Whisper processes a padded 30-second window no matter
how long you actually spoke, so a 3-second dictation costs the same as a
25-second one. Moonshine's compute scales with actual audio length.

Vendor-reported: 107 ms (Moonshine Medium) vs 11,286 ms (Whisper Large v3) on a
MacBook Pro, with Medium claiming lower WER than Whisper Large v3 at 245M
params. Treat the ratio as directional, not a spec.

Languages are narrower than Parakeet's — English, Spanish, Mandarin, Japanese,
Korean, Vietnamese, Ukrainian, Arabic.

**Implemented** as a selectable backend (`"moonshine"` in `stt.backends`).
Measured here on the same synthesized clips: **181 ms** for 3.2 s of audio,
1.25 s including model load, from a `base-en` model that downloads in seconds
rather than minutes. Transcript quality on these two clips was indistinguishable
from Parakeet's.

So the honest summary: on short dictation-shaped audio the two are close on
latency, and Moonshine wins decisively on install weight (tens of MB vs
622 MB). Parakeet keeps the default because it covers 25 languages and has the
stronger leaderboard position; Moonshine is one config line away if you want a
lighter install.

Implementation note: Moonshine's stream API requires `stream.start()` before
`add_audio()`, otherwise its VAD is inactive and the call fails with an opaque
"Unknown error". Cost me one debugging round.

## 3. Free-tier APIs

**Groq remains the best zero-cost cloud path** and the PRD's choice holds up:
2,000 speech-to-text requests/day, plus a 7,200 audio-seconds/hour ceiling
(~2 hours of audio per clock hour). Reported as the most generous perpetual
free tier of any hosted transcription API. Adding a card to the Developer tier
raises limits ~10x at no charge.

`QuotaLedger` now tracks both: daily request count and a **rolling** hourly
audio-seconds total (events carry timestamps and are pruned past the hour, so
the budget recovers gradually rather than resetting on the clock hour). The
router checks the audio budget *before* sending, so we fall back to local
instead of spending a request on a guaranteed 429.

Google AI Studio (Gemini 1.5 Flash) for the text pass is unchanged; the PRD's
architecture here is still current. I did not find a free-tier STT option that
beats Groq's terms.

Verify the exact Groq model IDs against their docs before relying on
`whisper-large-v3-turbo` — model naming there has churned.

## 4. Keeping small models from rewriting (PRD §5, the important one)

The research literature is blunt about the failure mode we're guarding against.
Generative error correction that **rewrites the whole transcript** will invent
entities: the documented example is a user saying *"I like algorithms"* and the
model emitting *"I like Al Gore"* because a similar-sounding entity was in
context ([LOGIC, arXiv 2601.15397](https://arxiv.org/abs/2601.15397v1)). The
mitigations that show up repeatedly:

- **Transcript-preserving constraints** rather than free generation — the model
  may delete and repunctuate, but not introduce.
- **Conservative filtering with confidence thresholds**, accepting a correction
  only when it clears a bar, instead of taking every rewrite.
- **Logit-space anchoring / constrained decoding** — enforced at the decoder,
  not via prompt text, so it cannot be talked out of.
- **Low temperature** (we already run 0.0) and **few-shot examples** (already in
  `prompts.py`).

What this repo does that lines up: the rules pre-pass runs first and the LLM
only ever sees already-cleaned text; `sanitize()` rejects any completion longer
than 1.3x the input, which catches the "started conversing" failure.

**The gap, now closed:** we checked *length*, not *containment*, so a
same-length hallucination (`algorithms` → `Al Gore`) passed. `check_containment`
in `llm/base.py` now requires every alphabetic token in the output to appear in
the input, modulo case, punctuation, and contraction splits, with a small
closed class of function words exempt (articles, auxiliaries) because
instruction 3 legitimately inserts them. A violation raises `ProviderError`,
which the router already treats as "try the next backend" — so the fallback is
the rules output rather than a hallucination. Tested in
`tests/test_guards.py::Containment`.

Ollama's structured-output support constrains *shape* (JSON schema), not
*content*, so it does not solve this; a real grammar-constrained decode over the
input vocabulary would, but needs llama.cpp-level access rather than Ollama's
HTTP API.

## 5. Prior art worth reading

- **[Handy](https://github.com/cjpais/Handy)** — the closest analogue: free,
  offline, cross-platform (Mac/Windows/Linux) push-to-talk dictation, already
  running **both Whisper and Parakeet** locally. Reported ~22k stars and weekly
  commits as of mid-2026. Best source for how the Parakeet-for-dictation path
  behaves in practice.
- **[VoiceInk](https://tryvoiceink.com)** — macOS-only, GPL v3, local Whisper,
  system-wide injection. ~4.3k stars.
- **OpenWhispr** — Whisper + Parakeet, macOS/Windows/Linux, zero retention.
- Also in the space: FluidVoice, Amical, VoiceTypr, nerd-dictation, Talon.

The pattern across all of them: **Parakeet has displaced Whisper as the default
local model for English dictation**, with Whisper retained for multilingual
coverage. That is the single clearest signal from this scan.

## What I did not find

- No zero-cost proxy API with better terms than Groq's free tier.
- No published benchmark of *false-start removal* quality across models — which
  is why `tests/harness.py` exists here. The corpus in this repo appears to be
  the only executable spec for the behavior the PRD describes.

---

## Sources

- [onnx-asr — GitHub](https://github.com/istupakov/onnx-asr) · [PyPI](https://pypi.org/project/onnx-asr/)
- [nvidia/parakeet-tdt-0.6b-v3 — HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Moonshine — GitHub](https://github.com/moonshine-ai/moonshine) · [Moonshine paper (arXiv 2410.15608)](https://arxiv.org/pdf/2410.15608)
- [Best open source STT model in 2026 (benchmarks) — Northflank](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- [Best local STT models 2026: Moonshine vs Parakeet vs Whisper — onresonant](https://www.onresonant.com/resources/local-stt-models-2026)
- [Groq API free tier limits in 2026 — Grizzly Peak Software](https://www.grizzlypeaksoftware.com/articles/p/groq-api-free-tier-limits-in-2026-what-you-actually-get-uwysd6mb)
- [Free speech-to-text APIs (2026): real limits compared — Spokenly](https://spokenly.app/blog/free-speech-to-text-apis)
- [LOGIC: contextual biasing via logit-space integration (arXiv 2601.15397)](https://arxiv.org/abs/2601.15397v1)
- [DiarizationLM: LLM post-processing with transcript preservation (arXiv 2401.03506)](https://arxiv.org/pdf/2401.03506)
- [Best open source Wispr Flow alternatives — OpenAlternative](https://openalternative.co/alternatives/wisprflow)
- [Handy vs Wispr Flow — Voibe](https://www.getvoibe.com/resources/handy-vs-wispr-flow/)
- [13 best Wispr Flow alternatives — VoiceInk](https://tryvoiceink.com/best-wispr-flow-alternatives)

---

## Addendum — measured on this machine, August 2026

Numbers from the real hardware (RTX 4070 Ti, quiet built-in mic), on the same
3.9 s utterance. These supersede the vendor figures above where they conflict.

| Stage | Backend | Latency | Output |
|---|---|---|---|
| STT | Groq Whisper (cloud) | **341 ms** | identical |
| STT | Parakeet int8 (local) | 532 ms | identical |
| Cleanup | rules (deterministic) | **0.1 ms** | identical |
| Cleanup | Gemini flash-lite | 585 ms | identical |
| Cleanup | Ollama llama3.1:8b (GPU) | 2,700 ms | identical |

Every path produced the same sentence. The rules pass already implements the
PRD's editing spec, so the AI pass is paying seconds for an occasional
apostrophe -- it is off by default and lives behind a Settings toggle.

Two API facts that cost real debugging time:

* **`gemini-1.5-flash` is retired.** A key issued in 2026 does not have it, and
  the request 404s. `gemini-flash-lite-latest` is the fast survivor and is an
  alias, so it should not need renaming again. `gemini-2.0-flash-lite` returns
  429 on the free tier; `gemini-2.5-flash` 404s without billing.
* **Groq sits behind Cloudflare, which blocks urllib's default user-agent**
  with HTTP 403 error 1010. Any client must send a real `User-Agent`. Nothing
  in Groq's docs mentions this; the failure looks like an auth problem.
