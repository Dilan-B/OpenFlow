"""User configuration, loaded from ``~/.openflow/config.json``.

Written as plain dataclasses with sane defaults so a first run works with no
config file at all. Anything absent from the file keeps its default.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("OPENFLOW_HOME", Path.home() / ".openflow"))
CONFIG_PATH = CONFIG_DIR / "config.json"

# Bumped whenever a *default* changes in a way an existing config file would
# otherwise mask. ``_migrate`` below carries saved files forward one step at a
# time. Without this, a config written on day one pins every default it ever
# saw -- which is how an install kept transcribing with whisper-large-v3-turbo
# and the cleanup pass switched off, months after both defaults had changed.
SCHEMA = 3


@dataclass(slots=True)
class HotkeyConfig:
    # pynput canonical form. Ctrl+Win is free on Windows: holding Ctrl
    # suppresses the Start menu that a bare Win keyup would open, and unlike
    # Alt+Space it does not collide with the window system menu.
    trigger: str = "<ctrl>+<cmd>"
    # push_to_talk: record while held. toggle: press once to start, again to stop.
    mode: str = "push_to_talk"
    cancel: str = "<esc>"
    # Replace the last insertion with the raw transcript -- "that is not what I
    # said". Ctrl+Shift+Z sits next to the universal undo without colliding
    # with it, so the app's own undo and the host app's stay separate.
    undo: str = "<ctrl>+<shift>+z"
    # Command Mode: hold, speak an instruction, release. Wispr's Windows
    # default, and deliberately a superset of the dictation combo -- the
    # listener hands a just-started dictation over when the third key lands.
    # Empty disables it.
    command: str = "<ctrl>+<cmd>+<alt>"
    # Ignore key-repeat chatter from a held key (milliseconds).
    debounce_ms: int = 120


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16_000      # what Whisper wants; no resampling needed
    channels: int = 1
    block_ms: int = 30
    max_seconds: int = 120         # hard stop so a stuck key cannot eat the disk
    input_device: int | None = None
    # Mute other apps (Spotify, videos, calls) while recording and restore
    # them the moment you let go, via the Windows volume mixer sessions.
    duck_others: bool = True


@dataclass(slots=True)
class SttConfig:
    # Order is the fallback chain. First backend that reports ready wins.
    # parakeet_onnx leads the local options: lighter install and faster on CPU
    # than faster-whisper (see docs/research-2026-08.md). faster_whisper stays
    # last as a already-paid-for safety net.
    backends: list[str] = field(
        default_factory=lambda: ["groq", "parakeet_onnx", "faster_whisper"]
    )
    # onnx-asr model id. nemo-parakeet-tdt-0.6b-v3 is multilingual (25 langs,
    # auto-detect); -v2 is English-only and slightly faster.
    parakeet_model: str = "nemo-parakeet-tdt-0.6b-v3"
    # int8 by default: the fp32 encoder is a 2.3 GB download and loads slowly,
    # which is the wrong trade for a hotkey-driven dictation tool.
    parakeet_quantization: str = "int8"  # "" for fp32
    # ONNX Runtime execution provider: auto | cpu | cuda | directml | coreml.
    # "auto" means CPU, which measured fastest for int8 Parakeet on the
    # hardware in docs/research-2026-08.md -- a transducer's decode loop is
    # dispatch-latency bound, which is not what a GPU helps with. Ask for a GPU
    # by name only, and benchmark it first with scripts/bench_providers.py.
    device: str = "auto"
    # Moonshine: opt in by adding "moonshine" to backends. TINY | BASE |
    # SMALL_STREAMING | MEDIUM_STREAMING.
    moonshine_arch: str = "BASE"
    local_model: str = "base.en"   # faster-whisper: base | small | medium (+ .en)
    local_compute_type: str = "int8"
    # whisper-large-v3, not -turbo. Turbo is a 4-layer distilled decoder: much
    # faster, and measurably worse on exactly the tokens dictation cares about
    # -- proper nouns, product names, rare words. At dictation lengths (a few
    # seconds of audio) the wall-clock difference is a fraction of the network
    # round-trip we are already paying, so the speed buys nothing a user feels
    # while the errors are ones they read.
    groq_model: str = "whisper-large-v3"
    language: str = "en"


@dataclass(slots=True)
class LlmConfig:
    # Master switch for the AI cleanup pass. On by default since the Groq
    # backend landed: measured here, rules 0.1 ms, Groq gpt-oss-20b 237 ms,
    # Gemini flash-lite 585 ms, local Ollama 8B 2,700 ms. The earlier default
    # was off because the only cloud option cost half a second to reproduce
    # what the rules pass already did. At ~240 ms the trade flips -- that is
    # below the threshold where a dictation feels delayed, and it buys the
    # sentence-level repairs the deterministic pass cannot do.
    enabled: bool = True
    # "rules" is always the last resort and never fails.
    backends: list[str] = field(
        default_factory=lambda: ["groq", "gemini", "ollama", "rules"]
    )
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    # Groq's OpenAI-compatible chat endpoint, reusing GROQ_API_KEY.
    #
    # qwen3.8-27b, chosen on the Wispr Flow parity corpus
    # (tests/corpus/wispr_cases.json), two full runs per model, paced so no
    # case fell back to another backend:
    #
    #   qwen3.8-27b          wispr 23/23, 23/23   stem 26, 27/29   ~215 ms
    #   gpt-oss-120b (low)   wispr 21/23, 21/23   stem 22, 23/29   ~365 ms
    #   gpt-oss-20b  (low)   wispr 19/23, 21/23   stem 24, 25/29   ~275 ms
    #   gpt-oss-20b  (med)   wispr 21/23          -                ~430 ms
    #
    # Best and fastest, and it answers without hidden reasoning tokens -- 7
    # completion tokens for a short dictation. gpt-oss's losses were the
    # Wispr-defining cases: keeping the *later* phrasing of a restatement, and
    # keeping "no" when it is an answer rather than a correction.
    #
    # Free tier: 1,000 requests/day and 8,000 tokens/minute. A request is
    # ~1,100 tokens with the prompt and few-shot pairs, so sustained rapid
    # dictation can hit the per-minute cap; the request then falls through to
    # the next backend rather than failing. gpt-oss-20b also caps tokens per
    # *day* (200,000), which this model does not.
    groq_model: str = "qwen/qwen3.8-27b"
    # gpt-oss only: how much the model reasons before answering. Reasoning
    # tokens are latency on the dictation path, but backtracking ("which
    # phrasing did the speaker keep?") is exactly the kind of edit that goes
    # wrong without any. See scripts/bench_cleanup.py.
    groq_reasoning_effort: str = "low"
    # An alias, not a pinned version: pinned names retire. A config saved
    # before this default changed keeps the old name forever, which is how
    # a dead gemini-1.5-flash can outlive the code that stopped naming it.
    gemini_model: str = "gemini-flash-lite-latest"
    timeout_s: float = 6.0
    # Warm-up runs off the dictation path, so it can wait out a cold load;
    # measured ~18 s for llama3.1:8b, against which timeout_s never stood a
    # chance. Sharing one timeout meant warm-up failed and the cold load
    # then landed on the first real dictation instead.
    warmup_timeout_s: float = 90.0
    # Daily free-tier ceilings; crossing one flips the router to the next
    # backend for the rest of the day (PRD section 4).
    # Groq's published free tier is 2,000 speech-to-text requests/day.
    # "groq" is speech-to-text; "groq_chat" is the cleanup LLM. They are
    # separate free-tier allowances on Groq's side, so they get separate
    # counters here -- sharing one would let a day of dictation cleanup
    # switch off cloud transcription, which is the more valuable of the two.
    daily_limits: dict[str, int] = field(
        default_factory=lambda: {"gemini": 1_400, "groq": 2_000, "groq_chat": 1_000}
    )
    # Groq also caps audio *duration*: 7,200 seconds per rolling hour. Requests
    # alone will never hit this at dictation lengths, but a long session can.
    hourly_audio_seconds: dict[str, int] = field(
        default_factory=lambda: {"groq": 7_200}
    )
    temperature: float = 0.0
    # Run the deterministic pass before the LLM so a slow/broken model still
    # leaves you with cleaned-up text.
    rules_prepass: bool = True
    # Call the LLM on every dictation, not only when the rules pass admitted
    # it was guessing. The old True default made llm.enabled almost a no-op:
    # "uncertain" means stem_removal hit its keep-both branch, which is rare,
    # so turning the cleanup on changed nothing the user could see. The rules
    # pass cannot know it mis-handled a sentence -- that is precisely the class
    # of error it has no rule for. Set True to trade quality back for latency.
    only_when_uncertain: bool = False


@dataclass(slots=True)
class UiConfig:
    # The recording pill: small capsule just above the taskbar.
    overlay_width: int = 74
    overlay_height: int = 26
    opacity: float = 0.97
    bottom_margin: int = 46
    accent: str = "#5B8DEF"
    appearance: str = "dark"
    # Main window
    window_width: int = 880
    window_height: int = 660
    start_minimized: bool = False    # launch straight to the tray
    close_to_tray: bool = True       # X hides the window, app keeps listening
    history_limit: int = 50
    # What to call you on the dictation page. Asked once on first run; empty
    # falls back to the Windows account name, which is often not a real name.
    display_name: str = ""


@dataclass(slots=True)
class InjectionConfig:
    # "paste" is far faster for long text; "type" survives apps that block
    # clipboard reads.
    method: str = "paste"
    type_interval_s: float = 0.004
    restore_clipboard: bool = True
    # Give the previously focused window time to come back before we send keys.
    focus_restore_delay_s: float = 0.06


@dataclass(slots=True)
class UpdateConfig:
    """Check GitHub for a newer release. One anonymous request, no auto-install
    -- see openflow/updates.py."""

    check_on_startup: bool = True
    interval_hours: int = 24
    last_checked_at: float = 0.0
    # Tag the user chose to stop being told about.
    skipped_version: str = ""


@dataclass(slots=True)
class ProfileConfig:
    """Reshape output for the app it is about to land in -- no trailing full
    stop in a terminal, straight quotes in a code editor. See profiles.py."""

    enabled: bool = True
    # Extra "executable.exe": "profile" mappings, merged over the built-ins.
    # Profiles are prose | code | shell | chat.
    apps: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class FormattingConfig:
    """Wispr Flow-style Smart Formatting and Flow Styles. See formatting.py."""

    # Spoken lists, numbers as digits, dropping the trailing period in
    # messaging apps, and fitting the text to what surrounds the caret.
    smart: bool = True
    # Read the few characters around the caret (accessibility API) so a
    # mid-sentence dictation is lowercased and spaced to fit. Nothing read is
    # kept; password fields are never read.
    context_aware: bool = True
    # App category -> style. formal | casual | very_casual | excited. Formal
    # everywhere matches what OpenFlow did before styles existed, and is what
    # Wispr does when no style is chosen.
    styles: dict[str, str] = field(default_factory=lambda: {
        "personal": "formal", "work": "formal", "email": "formal", "other": "formal",
    })
    # Extra "executable or app name": "category" mappings, over the built-ins.
    apps: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CommandConfig:
    """Command Mode -- speak an instruction instead of text. See commands.py."""

    enabled: bool = True
    # Reading the selection by copying it (Ctrl+C, then the clipboard is put
    # back) is the only way to see text in apps that expose none to
    # accessibility -- most Electron ones. Only ever used for Command Mode.
    clipboard_fallback: bool = True


@dataclass(slots=True)
class CaptureConfig:
    """Opt-in local dataset capture, for building a real evaluation set.

    Off by default and never transmitted anywhere -- see openflow/capture.py.
    """

    enabled: bool = False
    # Recording your voice is a bigger ask than keeping a sentence, so it is a
    # separate decision even once capture is on.
    audio: bool = False


@dataclass(slots=True)
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    profiles: ProfileConfig = field(default_factory=ProfileConfig)
    formatting: FormattingConfig = field(default_factory=FormattingConfig)
    commands: CommandConfig = field(default_factory=CommandConfig)
    updates: UpdateConfig = field(default_factory=UpdateConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    # Which set of defaults this file was written against. 0 means "predates
    # migrations"; see SCHEMA and _migrate.
    schema: int = 0
    log_transcripts: bool = False   # off by default: dictation is sensitive
    # Set once the first-run welcome has been answered or dismissed. Separate
    # from ui.display_name so that skipping the prompt, or clearing the name
    # later, does not make the app ask again on every launch.
    onboarded: bool = False

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or CONFIG_PATH
        cfg = cls()
        if not path.exists():
            return cfg
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cfg
        _apply(cfg, data)
        _migrate(cfg)
        # Migrate overlay geometry saved by older builds.
        if (cfg.ui.overlay_width, cfg.ui.overlay_height) in (
                (260, 72), (170, 46), (124, 32), (96, 26)):
            cfg.ui.overlay_width, cfg.ui.overlay_height = 74, 26
            cfg.ui.opacity, cfg.ui.bottom_margin = 0.97, 46
        return cfg

    def save(self, path: Path | None = None) -> Path:
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def _migrate(cfg: "Config") -> None:
    """Carry a saved config forward to the current defaults.

    Only touches settings the user is very unlikely to have chosen
    deliberately: a transcription model they never picked, and a cleanup pass
    whose old default made it a no-op. Anything a user plausibly set on purpose
    (hotkeys, devices, style) is left alone. Each step runs once -- turning the
    cleanup back off afterwards sticks, because the schema has moved on.
    """
    if cfg.schema < 1:
        if cfg.stt.groq_model == "whisper-large-v3-turbo":
            cfg.stt.groq_model = SttConfig().groq_model
        # The pair that made AI cleanup invisible: off, and gated to the rare
        # branch where the rules pass flagged itself uncertain.
        cfg.llm.enabled = True
        cfg.llm.only_when_uncertain = False
        if "groq" not in cfg.llm.backends:
            cfg.llm.backends = ["groq"] + [b for b in cfg.llm.backends if b != "groq"]
        for key, value in LlmConfig().daily_limits.items():
            cfg.llm.daily_limits.setdefault(key, value)
        cfg.schema = 1
    if cfg.schema < 2:
        # Wispr-parity scoring retired gpt-oss-20b as the cleanup default. A
        # config still naming it got it from the old default, not from a
        # choice -- nothing in the UI ever offered a cleanup model.
        if cfg.llm.groq_model == "openai/gpt-oss-20b":
            cfg.llm.groq_model = LlmConfig().groq_model
        cfg.schema = 2
    if cfg.schema < 3:
        # The old Style page held LLM tone presets ("professional", "casual")
        # that asked the model to reword -- which Wispr-parity cleanup and its
        # containment guard now forbid. Carry the one preset that still means
        # something, "casual", into the Flow Styles that replaced them.
        if _legacy_style() == "casual":
            for category, style in list(cfg.formatting.styles.items()):
                if style == "formal":
                    cfg.formatting.styles[category] = "casual"
        cfg.schema = 3


def _legacy_style() -> str:
    try:
        data = json.loads((CONFIG_DIR / "personalization.json").read_text(encoding="utf-8"))
        return str(data.get("style", ""))
    except (OSError, ValueError):
        return ""


def _apply(target, data: dict) -> None:
    """Shallow-merge a parsed JSON dict onto a nested dataclass."""
    for f in fields(target):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(target, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(target, f.name, value)


def api_key(name: str) -> str | None:
    """Read a provider key from the environment.

    Keys are never stored in the config file -- put them in the environment or
    a .env you source yourself.
    """
    return os.environ.get(name) or None
