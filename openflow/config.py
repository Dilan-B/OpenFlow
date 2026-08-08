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


@dataclass(slots=True)
class HotkeyConfig:
    # pynput canonical form. Ctrl+Win is free on Windows: holding Ctrl
    # suppresses the Start menu that a bare Win keyup would open, and unlike
    # Alt+Space it does not collide with the window system menu.
    trigger: str = "<ctrl>+<cmd>"
    # push_to_talk: record while held. toggle: press once to start, again to stop.
    mode: str = "push_to_talk"
    cancel: str = "<esc>"
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
    # Moonshine: opt in by adding "moonshine" to backends. TINY | BASE |
    # SMALL_STREAMING | MEDIUM_STREAMING.
    moonshine_arch: str = "BASE"
    local_model: str = "base.en"   # faster-whisper: base | small | medium (+ .en)
    local_compute_type: str = "int8"
    groq_model: str = "whisper-large-v3-turbo"
    language: str = "en"


@dataclass(slots=True)
class LlmConfig:
    # Master switch for the AI cleanup pass. Measured on this machine:
    # rules 0.1 ms, Gemini flash-lite 585 ms, local Ollama 8B 2,700 ms -- and
    # on ordinary dictation all three produce the same sentence, because the
    # rules pass already implements the PRD's editing spec. Off by default:
    # dictation is a latency product. The Settings toggle turns it on.
    enabled: bool = False
    # "rules" is always the last resort and never fails.
    backends: list[str] = field(default_factory=lambda: ["gemini", "ollama", "rules"])
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    gemini_model: str = "gemini-flash-lite-latest"
    timeout_s: float = 6.0
    # Daily free-tier ceilings; crossing one flips the router to the next
    # backend for the rest of the day (PRD section 4).
    # Groq's published free tier is 2,000 speech-to-text requests/day.
    daily_limits: dict[str, int] = field(
        default_factory=lambda: {"gemini": 1_400, "groq": 2_000}
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
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    log_transcripts: bool = False   # off by default: dictation is sensitive

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
