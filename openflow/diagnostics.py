"""A one-click report of everything worth knowing when something breaks.

A packaged build has no console, so the log file is the only evidence a failure
ever happened -- and nobody is going to find ``~/.openflow/openflow.log`` on
their own, let alone know which parts of it matter. This assembles the report
for them.

Deliberately excluded: transcripts, dictionary entries, snippets, and API keys.
The whole point of the app is that what you say stays on your machine, and a
diagnostics blob that leaks it would be a nasty way to break that promise.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from .config import CONFIG_DIR, Config

LOG_LINES = 60

# Substrings that mark a line as carrying something private. Log lines are
# ours, but huggingface and Qt also write here and their URLs can be long and
# token-bearing.
_REDACT_MARKERS = ("key=", "token=", "authorization", "signature=", "api_key",
                   "&sig=", "x-goog", "bearer ")


def _redact(line: str) -> str:
    lowered = line.lower()
    if any(marker in lowered for marker in _REDACT_MARKERS):
        return line.split(" ", 3)[0] + "  [line redacted -- contained a credential or signed URL]"
    return line


def _version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _tail_log(path: Path, lines: int = LOG_LINES) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [f"(could not read {path}: {exc})"]
    return [_redact(line) for line in content[-lines:]]


def _backend_status(config: Config) -> list[str]:
    """Which engines would actually run, without touching the network."""
    out: list[str] = []
    try:
        from .stt.engines import build_engine

        for name in config.stt.backends:
            try:
                engine = build_engine(name, config)
                ready = "ready" if engine.available() else "unavailable"
                kind = "local" if engine.is_local else "cloud"
            except Exception as exc:
                ready, kind = f"error: {exc}", "?"
            out.append(f"  stt.{name:<18} {ready} ({kind})")
    except Exception as exc:
        out.append(f"  (could not probe STT backends: {exc})")

    try:
        from .stt.providers import available_providers, select_providers

        out.append(f"  onnx providers      {', '.join(available_providers()) or 'none'}")
        out.append(f"  onnx selected       {', '.join(select_providers(config.stt.device))}")
    except Exception as exc:
        out.append(f"  (could not probe execution providers: {exc})")
    return out


def _dependencies() -> list[str]:
    out: list[str] = []
    for module in ("PySide6", "sounddevice", "numpy", "pynput", "pyperclip",
                   "PIL", "onnx_asr", "onnxruntime", "moonshine_voice", "pycaw"):
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installed")
        except ImportError:
            version = "MISSING"
        except Exception as exc:
            version = f"error: {exc}"
        out.append(f"  {module:<18} {version}")
    return out


def collect(config: Config | None = None) -> str:
    """Build the report. Never raises -- a diagnostics tool that crashes while
    you are already debugging something is worse than useless."""
    config = config or Config.load()
    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    lines.append(f"OpenFlow diagnostics (version {_version()})")

    section("System")
    lines.append(f"  platform           {platform.platform()}")
    lines.append(f"  python             {sys.version.split()[0]}")
    lines.append(f"  frozen             {bool(getattr(sys, 'frozen', False))}")
    lines.append(f"  executable         {sys.executable}")
    lines.append(f"  config dir         {CONFIG_DIR}")

    section("Settings")
    lines.append(f"  hotkey             {config.hotkey.trigger} ({config.hotkey.mode})")
    lines.append(f"  stt backends       {', '.join(config.stt.backends)}")
    lines.append(f"  stt device         {config.stt.device}")
    lines.append(f"  language           {config.stt.language or 'auto'}")
    lines.append(f"  llm enabled        {config.llm.enabled}")
    lines.append(f"  llm backends       {', '.join(config.llm.backends)}")
    lines.append(f"  injection          {config.injection.method}")
    lines.append(f"  duck others        {config.audio.duck_others}")
    lines.append(f"  log transcripts    {config.log_transcripts}")
    # Keys are read from the environment and never stored, but say which are
    # present so "why is the cloud path not used" answers itself.
    for env in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        lines.append(f"  {env:<18} {'set' if os.environ.get(env) else 'not set'}")

    section("Backends")
    lines.extend(_backend_status(config))

    section("Dependencies")
    lines.extend(_dependencies())

    section(f"Log (last {LOG_LINES} lines)")
    lines.extend(f"  {line}" for line in _tail_log(CONFIG_DIR / "openflow.log"))

    lines.append("")
    lines.append("Transcripts, dictionary entries, and snippets are deliberately "
                 "not included.")
    return "\n".join(lines)
