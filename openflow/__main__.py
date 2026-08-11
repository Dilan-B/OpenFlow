"""CLI entry point.

    python -m openflow                 # run the app
    python -m openflow --check         # report backend readiness and exit
    python -m openflow --write-config  # materialize the default config file
    python -m openflow --clean "..."   # pipe text through the cleanup chain
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .config import CONFIG_PATH, Config


def _ensure_stdio() -> bool:
    """Give a windowed build somewhere to write. Returns True when there was
    no console to begin with.

    Under pythonw.exe and PyInstaller's --noconsole, sys.stdout and sys.stderr
    are None, and anything that writes to them raises AttributeError from a
    place that never expected to fail: argparse's usage message on a bad
    argument, huggingface_hub's download progress bar while the local STT
    weights load. Point them at the null device so those paths are boring.
    """
    windowed = False
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            windowed = True
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    return windowed


def _configure_logging(verbose: bool, windowed: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    # With no console there is nothing behind sys.stderr but the null device,
    # so a StreamHandler would silently discard every record. Fall back to a
    # rotating file in the config directory.
    if windowed:
        from logging.handlers import RotatingFileHandler

        from .config import CONFIG_DIR

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            CONFIG_DIR / "openflow.log", maxBytes=512_000, backupCount=2,
            encoding="utf-8",
        )
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def _check(config: Config) -> int:
    from .llm.providers import build_provider
    from .llm.quota import QuotaLedger
    from .stt.engines import build_engine

    quota = QuotaLedger()
    print(f"config: {CONFIG_PATH if CONFIG_PATH.exists() else '(defaults)'}")
    print(f"hotkey: {config.hotkey.trigger}  mode={config.hotkey.mode}")

    print("\nspeech-to-text:")
    for name in config.stt.backends:
        try:
            engine = build_engine(name, config)
            ready = "ready" if engine.available() else "unavailable"
            kind = "local" if engine.is_local else "cloud"
        except ValueError:
            ready, kind = "unknown backend", "?"
        suffix = ""
        if kind == "cloud":
            limit = config.llm.daily_limits.get(name)
            audio_limit = config.llm.hourly_audio_seconds.get(name)
            parts = []
            if limit:
                parts.append(f"{quota.used(name)}/{limit} req today")
            if audio_limit:
                parts.append(
                    f"{quota.audio_seconds_used(name):.0f}/{audio_limit} audio-s this hour"
                )
            if parts:
                suffix = "  [" + ", ".join(parts) + "]"
        print(f"  {name:<16} {ready:<14} {kind}{suffix}")

    print("\npost-processing:")
    for name in config.llm.backends:
        if name == "rules":
            print(f"  {'rules':<16} ready (always)")
            continue
        try:
            provider = build_provider(name, config)
            ready = "ready" if provider.available() else "unavailable"
        except ValueError:
            ready = "unknown backend"
        limit = config.llm.daily_limits.get(name)
        used = quota.used(name)
        suffix = f"  [{used}/{limit} today]" if limit else ""
        print(f"  {name:<16} {ready}{suffix}")

    print("\ninput/output:")
    for module, label in (("sounddevice", "microphone"), ("pynput", "hotkeys + typing"),
                          ("PySide6", "window, pill, tray"), ("pyperclip", "paste injection"),
                          ("pycaw", "mute other apps while dictating"),
                          ("onnx_asr", "local STT (parakeet)"),
                          ("faster_whisper", "local STT (whisper fallback)"),
                          ("moonshine_voice", "local STT (moonshine, optional)")):
        try:
            __import__(module)
            status = "ready"
        except ImportError:
            status = "missing (pip install -r requirements.txt)"
        print(f"  {module:<16} {status}   {label}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Before argparse: a usage error in a windowed build writes to sys.stderr,
    # and if that is None the app dies with no window and no message.
    windowed = _ensure_stdio()

    parser = argparse.ArgumentParser(prog="openflow", description="System-wide voice-to-text.")
    parser.add_argument("--check", action="store_true", help="report backend readiness and exit")
    parser.add_argument("--write-config", action="store_true",
                        help=f"write default config to {CONFIG_PATH}")
    parser.add_argument("--clean", metavar="TEXT",
                        help="run text through the cleanup chain and print the result")
    parser.add_argument("--rules-only", action="store_true",
                        help="with --clean, skip the LLM and use the deterministic pass")
    parser.add_argument("--install-shortcuts", action="store_true",
                        help="create Desktop and Start Menu launchers (Windows)")
    parser.add_argument("--minimized", action="store_true",
                        help="start hidden in the system tray")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose, windowed)
    config = Config.load()

    if args.write_config:
        print(f"wrote {config.save()}")
        return 0

    if args.install_shortcuts:
        from .shortcuts import install_shortcuts

        for path in install_shortcuts():
            print(f"created {path}")
        return 0

    if args.minimized:
        config.ui.start_minimized = True

    if args.check:
        return _check(config)

    if args.clean is not None:
        if args.rules_only:
            from .text.cleaner import RuleBasedCleaner

            cleaner = RuleBasedCleaner()
        else:
            from .llm.cleaner import LLMCleaner

            cleaner = LLMCleaner(config)
        result = cleaner.clean(args.clean)
        print(result.text)
        if args.verbose:
            for retraction in result.retractions:
                print(f"  [{retraction.strategy}] dropped {retraction.removed!r} "
                      f"after {retraction.pivot!r}", file=sys.stderr)
        return 0

    from .app import OpenFlowApp

    return OpenFlowApp(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
