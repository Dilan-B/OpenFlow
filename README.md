# OpenFlow

**[⬇ Download OpenFlowSetup.exe](https://github.com/Dilan-B/OpenFlow/releases/latest/download/OpenFlowSetup.exe)**
— Windows installer, no Python required. Run it, then launch OpenFlow from the
Start Menu or Desktop.

System-wide voice-to-text for Windows/macOS/Linux. Hold a hotkey, talk, release
— cleaned-up text lands in whatever app had your cursor. Implements
`wispr_flow_clone_spec-v2.pdf` (PRD v2.0).

**Launch:** double-click `OpenFlow` on the Desktop or Start menu — it runs the
packaged `dist/OpenFlow/OpenFlow.exe`, a real windowed app, no Python or
terminal involved. Hold **Ctrl + Win** to dictate anywhere.

The UI is Qt (PySide6), styled after Wispr Flow: warm cream canvas, white rounded content panel, icon sidebar, serif editorial headlines on dark banners, teal data-viz, and a streak heatmap. The window is a Wispr-style workspace: **Dictation** (hotkey, history, stats),
**Insights** (words/day chart, streak, latency), **Dictionary** (names the
transcriber should know — biases cloud recognition and fuzz-repairs local
output: `open flo` → `OpenFlow`, but never touches real words like "grow"),
**Snippets** (say a trigger, get the full text), **Style** (tone presets for
the AI cleanup pass), **Transforms** (rewrite the Scratchpad: formal, shorter,
bullets), **Scratchpad** (dictate long-form into the app itself), and
**Settings**. Spoken punctuation works everywhere: "new paragraph",
"question mark", and a trailing "period" or "comma" do what they say.

While you hold the shortcut, OpenFlow mutes every other app playing audio
(Spotify, videos, calls) through the Windows volume-mixer sessions and
restores them the instant you release — apps you muted yourself stay muted.
The recording pill (96×26, antialiased, fast-attack/slow-decay bar motion)
appears on whichever monitor your cursor is on. Both are toggleable in Settings.

## Status

| Layer | State |
|---|---|
| Sentence-stem removal + test harness | 22/22 golden cases, full unit suite green |
| LLM output guards (containment, length, preamble stripping) | Implemented + tested |
| Free-tier quota ledger (daily requests + rolling hourly audio) | Implemented + tested |
| STT routing (Groq → Parakeet ONNX → faster-whisper) | Implemented; local path verified end to end on real ASR output |
| Post-processing routing (Gemini → Ollama → rules) | Implemented; not exercised against live APIs (no keys set) |
| Orchestrator (overlay + worker + injection) | Boots and carries synthesized speech through to injected text |
| Global hotkeys, mic capture, keystroke injection | Hotkey parsing, mic capture, and overlay verified; real keystroke injection not yet fired into another app |

`python -m openflow --check` reports what's ready on your machine.

## Quick start

A `.venv` already exists with everything you need: the core runtime
(sounddevice, numpy, pynput, PySide6, pyperclip, pycaw) plus both local STT
backends (`onnx-asr` for Parakeet, `moonshine-voice`). Activate it:

```bash
.venv/Scripts/activate
```

`faster-whisper` is the only optional piece left out — it's third in the
fallback chain and pulls in ctranslate2:

```bash
pip install faster-whisper
```

Then just **double-click OpenFlow on your Desktop** (or find it in the Start
menu). No terminal. It opens the app window and starts listening for the
shortcut immediately.

Hold **Ctrl + Win**, speak, release — the text appears at your cursor in
whatever app you were in. Esc cancels mid-recording.

Closing the window doesn't quit: OpenFlow keeps running in the system tray and
the shortcut keeps working. Quit from the tray menu or the button in Settings.

Only one copy ever runs. Launching it again — from the Desktop, the Start menu,
or the Startup entry — surfaces the window you already have instead of starting
a second process. Two instances would each bind the global hotkey, so a single
keypress would start two recordings racing to paste into the same cursor.

To recreate the launchers (already done once):

```bash
python -m openflow --install-shortcuts
```

### The window

- **Hero card** — your current shortcut, big. *Change shortcut* captures the
  next chord you press; the segmented control switches hold-to-talk vs toggle.
- **Stats** — words dictated, minutes saved (vs 40 wpm typing), your speaking
  rate, average end-to-end latency.
- **Recent** — your last dictations. Text is only stored if you turn on *Save
  dictation text*; otherwise it shows word counts and timings.
- **Engines** — a green dot per backend, so "why is it not using the cloud" is
  answerable at a glance.
- **Settings** — start with Windows, close-to-tray, audio ducking, insert method.

### Speed

Measured on this machine, same 3.9 s utterance, all producing identical text:

| Stage | Backend | Latency |
|---|---|---|
| speech-to-text | Groq Whisper (cloud) | **341 ms** |
| speech-to-text | Parakeet int8 (local) | 532 ms |
| cleanup | rules (deterministic) | **0.1 ms** |
| cleanup | Gemini flash-lite | 585 ms |
| cleanup | Ollama llama3.1:8b | 2,700 ms |

End to end that is **~350 ms** from key release to text on screen. The AI
cleanup pass is **off by default**: the deterministic pass already implements
the PRD's editing spec, so on ordinary dictation the AI produces the same
sentence for seconds more. Turn it on in Settings if you want it.

### API keys (optional — everything falls back to local/deterministic)

```bash
setx GROQ_API_KEY "..."
setx GEMINI_API_KEY "..."
```

Keys are read from the environment only; they are never written to the config
file.

### Local-only operation

With no API keys set, the chain is **Parakeet → Ollama → rules** and nothing
leaves the machine. Parakeet int8 weights (622 MB) download on first use into
`~/.cache/huggingface` — already done in this repo. For the LLM half, install
[Ollama](https://ollama.com) and `ollama pull llama3.1:8b`; without it, the
deterministic rules pass handles cleanup on its own.

To try Moonshine instead — smaller and tuned for short utterances — put
`"moonshine"` ahead of `"parakeet_onnx"` in `stt.backends`.

## The stem-removal harness

The PRD hands false-start removal to the LLM. This repo also implements it
deterministically, because that is what makes the behavior *testable* — and the
golden corpus then doubles as the acceptance test for the LLM prompt:

```bash
python -m tests.harness
```

```bash
python -m tests.harness --cleaner ollama -v
```

Same 22 cases, any backend. When you tune `openflow/llm/prompts.py` for a small
local model, this is how you find out whether the change actually helped.
Exact-match and normalized-match are reported separately, so an LLM that made
the right edit with different punctuation shows as `NEAR`, not `FAIL`.

Unit tests (one generated test per corpus case, plus internals, guards, and
the quota ledger):

```bash
python -m unittest discover -s tests -t .
```

### End-to-end check

`tests/verify_pipeline.py` synthesizes speech with Moonshine's TTS, transcribes
it, and runs the cleanup — so the whole audio→text→correction path is exercised
without a microphone:

```bash
python -m tests.verify_pipeline --engine parakeet_onnx
```

Current output on the PRD's canonical sentence, from real ASR:

```
heard    : 'Can we meet up on Tuesday at 5 or actually, can we meet up on Friday at 3?'
cleaned  : 'Can we meet up on Friday at 3?'   [re-anchor] dropped 'Can we meet up on Tuesday at 5'
```

Measured on this machine, CPU only: Parakeet int8 **148 ms** for 3.2 s of audio
(~21× realtime), Moonshine **181 ms**. Model load happens once on a background
thread at startup, so it never lands on the hotkey path.

### What the rules pass does

Four strategies, tried in order, with a deliberate bail-out:

| Strategy | Trigger | Example |
|---|---|---|
| re-anchor | replacement repeats the opening of the retracted clause | `meet tuesday at 5, or actually, meet friday at 3` → `meet friday at 3` |
| restart | replacement opens a fresh independent clause | `ship the beta, actually, let's hold it` → `let's hold it` |
| slot-patch | replacement is a fragment; only matching slots are overwritten | `Order 12 units, or actually 20` → `Order 20 units` |
| pivot-only | ambiguous — delete the pivot phrase, keep both sides | `passes locally, or actually in staging too` |

The last row is the point: when the intent is unclear the tool leaves your words
alone. A dictation tool that occasionally deletes a clause you meant to keep is
worse than one that occasionally leaves an extra one.

### Built-in autofixes (no model, ~0.2 ms)

Repairs a person would make without thinking, so no round trip is spent on
them. Everything here is unambiguous by construction — anything needing
context to be right is left to the AI pass.

| Category | Example |
|---|---|
| contractions | `i dont think we cant` → `I don't think we can't.` |
| acronyms | `the api returns json over https` → `The API returns JSON over HTTPS` |
| brands | `github`, `javascript`, `iphone` → `GitHub`, `JavaScript`, `iPhone` |
| stutters | `the the report` → `the report` |
| emails | `dilan dot bhimani at gmail dot com` → `dilan.bhimani@gmail.com` |
| domains | `github dot com` → `github.com` |
| times | `3 30 p m` → `3:30 PM` |
| spoken symbols | `hashtag`, `ampersand`, `dollar sign` → `#`, `&`, `$` |

The refusals are the actual specification: `its`, `were`, and `hell` are never
touched, because picking the contraction requires knowing what the sentence
means. `lets` is only corrected at a sentence start, where `let's` is the only
sensible reading. Casing is applied only to tokens the transcriber left
entirely lowercase, so `gRPC`, `iOS` and `PostgreSQL` survive untouched.

The pass runs in two halves for a structural reason: word repairs go *before*
sentence splitting (they add no punctuation), while symbol and address repairs
go *after* it — otherwise the splitter reads `gmail.com` as two sentences and
re-spaces it into `Gmail. Com`.

Filler stripping distinguishes hard fillers (`um`, `you know` — always removed)
from soft ones (`like`, `right` — removed only in discourse position), so
"I **like** the new flow" and "take the second **right**" survive intact.

## Architecture

```
openflow/
  __main__.py        CLI: run | --check | --clean | --write-config
  app.py             orchestrator; Qt thread + hotkey thread + worker thread
  config.py          ~/.openflow/config.json, dataclass-backed
  text/              deterministic cleanup (pure stdlib, no imports to pay for)
    tokens.py          tokenizer
    pivots.py          lexicons — tune these, not the algorithm
    stem_removal.py    the four strategies above
    fillers.py         hard/soft filler stripping
    punctuation.py     capitalization + terminal punctuation
    cleaner.py         Cleaner protocol + RuleBasedCleaner
  llm/
    prompts.py         PRD §3 prompt verbatim + local-model guardrails
    base.py            output guards: containment, length, preamble stripping
    providers.py       Ollama, Gemini (urllib, no SDKs)
    quota.py           daily requests + rolling hourly audio budget
    cleaner.py         cloud → local → rules fallback chain
  stt/
    engines.py         Groq Whisper, Parakeet (ONNX), Moonshine, faster-whisper
    router.py          same fallback discipline, quota-aware before sending
  audio/recorder.py  always-open input stream, gated capture
  history.py         recent dictations + lifetime stats
  shortcuts.py       Desktop/Start Menu/Startup .lnk creation (no pywin32)
  input/
    hotkeys.py         push-to-talk and toggle, debounced, runtime rebinding
    injector.py        paste (fast) or type; restores foreground window first
  ui/
    theme.py           one palette for both windows
    main_window.py     the app window
    overlay.py         borderless topmost overlay + waveform
    tray.py            tray icon; window can close without quitting
    icon.py            the mic mark, drawn in code (tray + .ico)
tests/
  corpus/stem_cases.json
  harness.py         scores any cleaner against the corpus
  test_stem_removal.py
  test_guards.py     LLM output guards + quota ledger
docs/
  research-2026-08.md  PRD §5 scan, and why the backends are ordered as they are
```

## Guardrails on the LLM pass

Prompt instructions are advisory; these are enforced in code, after every
completion, on every backend:

- **Containment** — every word in the output must appear in the input (modulo
  case, punctuation, and contraction splits, with articles and auxiliaries
  exempt). This catches the documented ASR-correction failure where a model
  substitutes a similar-sounding entity: *"I like algorithms"* → *"I like Al
  Gore"*. A length check cannot see that; the strings are the same size.
- **No growth** — completions over 1.3× the input are rejected as the model
  having started a conversation instead of editing.
- **Preamble and quote stripping** — "Here is your text:" and wrapping quotes
  are removed before the containment check runs.

A violation raises `ProviderError`, which the router already treats as "try the
next backend" — so a hallucinating model degrades to the deterministic rules
output rather than typing fiction into your document.

## Speed decisions

- The audio stream is opened at launch and stays open; the hotkey only flips a
  capture flag. Opening PortAudio per press costs 100–300 ms on Windows.
- The rules pass runs first (~0.05 ms) and its output is what gets sent to the
  LLM: fewer tokens, and less room for a small model to wander.
- Injection defaults to clipboard paste — constant time regardless of length —
  with per-character typing as the fallback for apps that block paste.
- `faster-whisper` uses greedy decoding (`beam_size=1`); the accuracy delta is
  not worth the latency for dictation.
- The local model is warmed on a background thread at startup so the first
  dictation is never a cold start.
- Backend availability is cached (30 s on a miss, 5 min on a hit). Probing a
  stopped Ollama costs a full connect timeout, and that was landing on every
  single dictation — 720 ms of pure waiting before the rules pass even ran.
- The LLM layer uses `urllib`, so no SDK import cost at launch.

## Configuration

```bash
python -m openflow --write-config
```

Writes `~/.openflow/config.json`. Notable keys:

| Key | Meaning |
|---|---|
| `hotkey.trigger` | `<ctrl>+<cmd>` (default), `<caps_lock>`, `<ctrl>+<shift>+d`, … |
| `hotkey.mode` | `push_to_talk` or `toggle` |
| `stt.backends` | fallback order: `groq`, `parakeet_onnx`, `moonshine`, `faster_whisper` |
| `stt.parakeet_model` | `nemo-parakeet-tdt-0.6b-v3` (25 languages) or `-v2` (English, slightly faster) |
| `stt.parakeet_quantization` | `""` or `int8` to halve load time |
| `stt.moonshine_arch` | `TINY`, `BASE`, `SMALL_STREAMING`, `MEDIUM_STREAMING` |
| `llm.enabled` | master switch for the AI cleanup pass — off by default (see below) |
| `llm.backends` | fallback order: `gemini`, `ollama`, `rules` |
| `llm.daily_limits` | free-tier request ceilings per provider per day |
| `llm.hourly_audio_seconds` | rolling audio-duration ceiling (Groq: 7200/hour) |
| `injection.method` | `paste` (fast) or `type` |

## Research

[docs/research-2026-08.md](docs/research-2026-08.md) — the PRD §5 scan of
current transcription optimizations and free-tier APIs, with four concrete
recommended changes (Parakeet-via-ONNX as the local backend, Moonshine for
short utterances, corrected Groq quota numbers, and a transcript-containment
guard against LLM hallucination).

## Known gaps

- CapsLock as a trigger will still toggle caps state; suppressing that needs a
  low-level hook (`suppress=True` plus a re-emit path) that isn't wired up.
  The same applies to any trigger — keystrokes are not swallowed, so pick a
  chord your apps don't already use. Ctrl+Win is chosen because it's inert on
  Windows: holding Ctrl suppresses the Start menu a bare Win keyup would open.
- macOS requires granting Accessibility and Microphone permissions to the host
  terminal before hotkeys or injection work at all.
- Gemini's `daily_limits` value (1400) is still a placeholder — Groq's 2000/day
  and 7200 audio-s/hour are the published free-tier figures. Verify the Groq
  model IDs against their docs; naming there has churned.
- Containment is word-level, so a model that *reorders* words the speaker did
  say passes the check. Reordering is a far less damaging failure than
  substitution, and the length guard bounds it.
- The overlay shows state, not live text — no partial-transcript preview.
