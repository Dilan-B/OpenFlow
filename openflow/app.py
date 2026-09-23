"""Application orchestrator -- Qt edition.

Threading model, one direction of data flow:

  Qt main thread   owns the window, pill, and tray; a 16 ms QTimer drains the
                   event queue and advances the pill animation
  pynput thread    hotkey edges + audio ducking (COM objects stay in the
                   apartment they were created in)
  worker thread    transcription + cleanup + injection (the slow path)

Nothing but the Qt thread touches widgets; the worker posts state changes
through a queue, so a slow model can never freeze the pill animation.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from .audio import chime
from .audio.conditioning import condition, is_silent, measure, voiced_seconds
from .audio.ducker import AudioDucker
from .audio.recorder import AudioUnavailable, Recorder
from .capture import Capture
from .commands import EDIT_PROMPT, parse as parse_command
from . import __version__
from .config import CONFIG_DIR, Config, llm_chain, stt_chain
from .context import (
    EMPTY as EMPTY_CONTEXT, ContextReader, repair_names, tag_files, with_own_name,
)
from .corrections import shared as corrections
from .exits import EXIT_NO_HOTKEY, EXIT_NO_MICROPHONE, EXIT_OK
from .formatting import classify, smart_format
from .history import Entry, History
from .input.caret import CaretReader, copy_selection
from .input.hotkeys import HotkeyListener, HotkeyUnavailable
from .input import macos_permissions
from .input.injector import Injector
from .llm.cleaner import LLMCleaner
from .llm.quota import QuotaLedger
from .personalization import shared as personalization
from .profiles import DEFAULT, foreground_window, profile_for
from .stt.engines import SttError, build_engine
from .stt.router import SttRouter
from .text.hallucinations import is_silence_hallucination

log = logging.getLogger(__name__)

FRAME_MS = 16          # ~60 fps pill animation
MIN_AUDIO_S = 0.25     # shorter than this is a mis-press, not speech
# How long undo stays available. It works by sending backspaces, so it is only
# correct while the caret has not moved; past this we refuse rather than risk
# deleting something the user typed.
UNDO_WINDOW_S = 15.0

TRANSFORM_PROMPTS = {
    "formal": "Rewrite the user's text in a clear, professional register. Preserve "
              "every fact and name. Output only the rewritten text.",
    "shorten": "Shorten the user's text to its essentials. Preserve every fact. "
               "Output only the shortened text.",
    "bullets": "Restructure the user's text as a concise bulleted list, one point "
               "per line starting with '- '. Preserve every fact. Output only the list.",
}


class OpenFlowApp:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.load()
        self.quota = QuotaLedger()
        self.history = History(limit=self.config.ui.history_limit)
        self.recorder = Recorder(self.config.audio)
        self.stt = SttRouter(self.config, self.quota)
        self.cleaner = LLMCleaner(self.config, quota=self.quota)
        self.injector = Injector(self.config.injection)
        self.ducker = AudioDucker()
        self.personal = personalization()
        self.corrections = corrections()
        self.capture = Capture(self.config)
        self.hotkeys: HotkeyListener | None = None
        # (raw, cleaned, capture id, monotonic time) of the last insertion.
        self._last_insertion: tuple[str, str, str | None, float] | None = None
        # Executable the text is headed for, sampled when the hotkey goes down.
        self._target_app = ""
        self._target_title = ""
        # Reads the text around the caret in the target app, started with the
        # recording and collected after transcription. See input/caret.py.
        self._caret: CaretReader | None = None
        # What is on screen (names, the open project), gathered while the
        # user speaks. See context.py.
        self._context: ContextReader | None = None
        # Command Mode: the caret read taken when its hotkey went down, which
        # is the only moment the user's selection is still the focused one.
        self._command_caret: CaretReader | None = None

        self.window = None
        self.overlay = None
        self.tray = None
        self.single_instance = None
        self.paused = False
        self._hotkeys_reported_dead = False
        # macOS: waiting on Accessibility / Input Monitoring. See
        # input/macos_permissions.py and _check_permissions.
        self._awaiting_permissions = False
        self._permissions_checked_at = 0.0
        self._events: queue.Queue = queue.Queue()
        self._jobs: queue.Queue = queue.Queue()
        self._stop = threading.Event()

    # -- startup -----------------------------------------------------------
    def run(self) -> int:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        from .ui.main_window import MainWindow
        from .ui.overlay import Overlay
        from .ui.tray import Tray

        qt_app = QApplication.instance() or QApplication([])
        qt_app.setQuitOnLastWindowClosed(False)   # close-to-tray

        # macOS aborts the process if pynput reads the keyboard layout off the
        # main thread; snapshot it here, before the worker and hotkey threads.
        from .input import macos_layout

        macos_layout.install()

        # Claim the lock before opening the microphone or binding the hotkey:
        # a second instance must not touch either. If one is already running it
        # gets raised instead, so clicking the shortcut behaves as expected.
        from .single_instance import SingleInstance

        self.single_instance = SingleInstance()
        if not self.single_instance.acquire():
            log.info("OpenFlow is already running; showing that window instead")
            return EXIT_OK
        self.single_instance.activate_requested.connect(self.show_window)

        # Ask before the main window is built: the dictation page bakes the
        # greeting into a label at construction time.
        if not self.config.onboarded:
            self._ask_for_name()

        self.window = MainWindow(
            self.config, self.history, self.personal,
            callbacks={
                "rebind": self._begin_rebind,
                "mode": self._set_mode,
                "pause": self.toggle_pause,
                "quit": self.quit,
                "hide": self.hide_window,
                "clear_history": self.history.clear,
                "setting": self._apply_setting,
                "api_key": self._set_api_key,
                "transform": self._start_transform,
                "last_text": self._last_text,
                "learn": self._learn_correction,
            },
        )
        self.overlay = Overlay(self.config.ui)
        self.tray = Tray(
            on_show=lambda: self._events.put(("show_window", None)),
            on_toggle_pause=lambda: self._events.put(("toggle_pause", None)),
            on_toggle_texting=lambda: self._events.put(("toggle_texting", None)),
            on_quit=lambda: self._events.put(("quit", None)),
            texting=self.config.formatting.texting,
        )

        try:
            self.recorder.open()
        except AudioUnavailable as exc:
            log.error("microphone unavailable: %s", exc)
            return EXIT_NO_MICROPHONE

        threading.Thread(target=self._worker, name="openflow-worker", daemon=True).start()
        threading.Thread(target=self._warm, name="openflow-warm", daemon=True).start()

        self.hotkeys = HotkeyListener(
            self.config.hotkey,
            on_start=self._on_hotkey_start,
            on_stop=self._on_hotkey_stop,
            on_cancel=self._on_hotkey_cancel,
            on_undo=self._on_hotkey_undo,
            on_command_start=(self._on_command_start
                              if self.config.commands.enabled else None),
            on_command_stop=self._on_command_stop,
        )
        try:
            self.hotkeys.start()
        except HotkeyUnavailable as exc:
            log.error("hotkey listener unavailable: %s", exc)
            return EXIT_NO_HOTKEY

        # Without these the hotkey only works while OpenFlow is focused, and
        # macOS will not list the app to switch on until it has asked.
        if not macos_permissions.trusted():
            self._awaiting_permissions = True
            macos_permissions.clear_stale(
                __version__, CONFIG_DIR / "macos_permissions.json")
            macos_permissions.request()

        self.tray.start()
        self._refresh_engines()
        self.window.set_state("ready")
        if not self.config.ui.start_minimized:
            self.window.show()

        timer = QTimer()
        timer.timeout.connect(self._tick)
        timer.start(FRAME_MS)

        log.info("OpenFlow ready -- %s (%s)",
                 self.config.hotkey.trigger, self.config.hotkey.mode)
        try:
            return qt_app.exec()
        finally:
            self.shutdown()

    # -- corrections -------------------------------------------------------
    def _last_text(self) -> str:
        """What we inserted last, for the "Fix last" button.

        Read from memory rather than history, so correcting works even with
        ``log_transcripts`` off -- the privacy default should not quietly
        disable the feature that makes dictation get better.
        """
        return self._last_insertion[1] if self._last_insertion else ""

    def _learn_correction(self, original: str, corrected: str) -> list:
        """Record a hand-correction and push what it teaches upstream.

        Two destinations, because a correction answers two different questions.
        The store replays the fix on future transcripts; ``promote`` turns a
        single-word name fix into a dictionary term, which rides along in the
        recognition prompt and stops the error being made in the first place.
        """
        learned = self.corrections.learn(original, corrected)
        if not learned:
            return []
        promoted = self.corrections.promote(self.personal)
        if promoted:
            log.info("promoted to dictionary: %s", ", ".join(promoted))
        # Keep "fix last" idempotent: a second pass should diff against what
        # the user just said was right, not against the text they replaced.
        if self._last_insertion and self._last_insertion[1] == original:
            raw, _cleaned, record_id, at = self._last_insertion
            self._last_insertion = (raw, corrected, record_id, at)
        log.info("learned %d correction(s) from a hand edit", len(learned))
        return learned

    # -- undo --------------------------------------------------------------
    def _on_hotkey_undo(self) -> None:
        """Swap the last insertion for the raw transcript.

        Bounded by UNDO_WINDOW_S because replace_last() works by sending
        backspaces: it is only safe while the caret is still sitting where we
        left it. After a few seconds the user has very likely typed, clicked,
        or moved on, and those backspaces would eat their words instead of
        ours. Refusing is the safe failure.
        """
        pending = self._last_insertion
        if pending is None:
            return
        raw, cleaned, record_id, at = pending
        if time.monotonic() - at > UNDO_WINDOW_S:
            log.info("undo ignored: %.0fs since the insertion, too late to be safe",
                     time.monotonic() - at)
            self._flash_error("Too late to undo")
            return
        if raw.strip() == cleaned.strip():
            log.info("undo ignored: cleanup changed nothing")
            return

        try:
            replaced = self.injector.replace_last(raw)
        except Exception as exc:
            log.warning("undo failed: %s", exc)
            self._flash_error("Undo failed")
            return
        if not replaced:
            return

        self._last_insertion = None
        # The user rejecting our cleanup is the most honest label we ever get.
        self.capture.reject(record_id)
        log.info("undone: restored the raw transcript")
        self._events.put(("flash", "undone"))

    def _ask_for_name(self) -> None:
        """First-run greeting prompt. Never fatal: a failure here must not
        cost someone their dictation hotkey."""
        if self.config.ui.start_minimized:
            # Launched to the tray at sign-in -- nobody is watching. Leave
            # onboarded False and ask on the next ordinary launch.
            return
        try:
            from .ui.main_window import _account_name
            from .ui.welcome import ask_for_name

            name = ask_for_name(self.config, suggestion=_account_name())
            log.info("first run: greeting set to %r", name or "(skipped)")
        except Exception as exc:
            log.warning("welcome prompt failed: %s", exc)

    def _warm(self) -> None:
        self.stt.warm()
        self._warm_llm()
        self._events.put(("engines", None))
        self._check_for_updates()

    def _check_for_updates(self) -> None:
        """Runs on the warm-up thread, after the models -- an update notice is
        never worth delaying the first dictation for."""
        try:
            from .updates import check_if_due

            release = check_if_due(self.config)
            if release is None:
                return
            if release.version == self.config.updates.skipped_version:
                return
            log.info("update available: %s", release.version)
            self._events.put(("update", release))
        except Exception as exc:
            log.debug("update check failed: %s", exc)

    def _warm_llm(self) -> None:
        """Load the local model's weights before the first dictation needs
        them. Measured: ~18 s cold for llama3.1:8b, ~2 s once resident."""
        if not self.config.llm.enabled or "ollama" not in self.config.llm.backends:
            return
        try:
            from .llm.providers import OllamaProvider

            provider = OllamaProvider(self.config)
            if provider.available():
                started = time.monotonic()
                provider.warm(self.config.llm.warmup_timeout_s)
                log.info("ollama warm (%s) in %.1fs",
                         provider.model, time.monotonic() - started)
        except Exception as exc:
            log.debug("ollama warm-up skipped: %s", exc)

    def shutdown(self) -> None:
        self._stop.set()
        if self.hotkeys:
            self.hotkeys.stop()
        if self.tray:
            self.tray.stop()
        self.recorder.close()
        self.ducker.restore()   # never leave the user's audio muted
        if self.single_instance is not None:
            self.single_instance.release()

    def quit(self) -> None:
        from PySide6.QtWidgets import QApplication

        self._stop.set()
        QApplication.instance().quit()

    # -- window ------------------------------------------------------------
    def show_window(self) -> None:
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()
        self.window.refresh()

    def hide_window(self) -> None:
        self.window.hide()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        log.info("dictation %s", "paused" if self.paused else "resumed")
        if self.hotkeys:
            self.hotkeys.paused = self.paused
        if self.paused:
            self.recorder.cancel()
            self.ducker.restore()
        state = "paused" if self.paused else "ready"
        self.window.set_state(state)
        self.tray.set_state(state, self.paused)

    def set_texting(self, on: bool) -> None:
        """Texting mode on or off, from Settings or the tray -- kept in step."""
        self.config.formatting.texting = on
        self.config.save()
        log.info("texting mode %s", "on" if on else "off")
        self.window.set_texting(on)
        self.tray.set_texting(on)

    def _set_api_key(self, name: str, value: str | None) -> tuple[bool, str]:
        """Save (or with None, remove) a provider key from the Settings page.
        Returns (ok, message to show). The key itself is never logged."""
        from . import keys

        try:
            if value is None:
                keys.delete(name)
                log.info("removed saved %s", name)
            else:
                keys.save(name, value)
                log.info("saved %s to the %s", name, keys.store_name())
        except ValueError as exc:
            return False, str(exc).capitalize()
        except Exception as exc:
            log.warning("could not save %s: %s", name, exc)
            return False, f"Could not save to the {keys.store_name()}"
        self._events.put(("engines", None))
        return True, ""

    # -- settings ----------------------------------------------------------
    def _apply_setting(self, key: str, value) -> None:
        if key == "launch_at_login":
            from .shortcuts import set_launch_at_login

            try:
                set_launch_at_login(bool(value))
            except Exception as exc:
                log.warning("could not change startup entry: %s", exc)
            return
        if key == "llm_enabled":
            self.config.llm.enabled = bool(value)
            self.config.save()
            self._events.put(("engines", None))
            return
        if key == "display_name":
            from .ui.welcome import clean_name

            self.config.ui.display_name = clean_name(value)
            self.config.save()
            self._events.put(("greeting", None))
            return
        if key == "updates.skipped_version":
            self.config.updates.skipped_version = str(value)
            self.config.save()
            return
        if key == "updates_enabled":
            self.config.updates.check_on_startup = bool(value)
            self.config.save()
            return
        if key == "commands_enabled":
            self.config.commands.enabled = bool(value)
            self.config.save()
            log.info("command mode %s (restart to rebind the hotkey)",
                     "enabled" if value else "disabled")
            return
        if key == "start_sound":
            self.config.audio.start_sound = bool(value)
            self.config.save()
            if value:
                chime.play()    # so you hear what you just turned on
            return
        if key == "texting_mode":
            self.set_texting(bool(value))
            return
        if key == "formatting_smart":
            self.config.formatting.smart = bool(value)
            self.config.save()
            return
        if key == "formatting_context":
            self.config.formatting.context_aware = bool(value)
            self.config.save()
            return
        if key.startswith("formatting.styles."):
            category = key.rsplit(".", 1)[1]
            from .formatting import STYLES_FOR

            if category in STYLES_FOR and value in STYLES_FOR[category]:
                self.config.formatting.styles[category] = str(value)
                self.config.save()
            else:
                log.warning("ignoring style %r for category %r", value, category)
            return
        if key == "profiles_enabled":
            self.config.profiles.enabled = bool(value)
            self.config.save()
            return
        if key == "capture_enabled":
            self.config.capture.enabled = bool(value)
            if not value:
                # Turning it off deletes what was collected. Leaving recordings
                # behind after someone opts out is not a defensible default.
                removed = Capture.purge()
                log.info("capture disabled; removed %d file(s)", removed)
            self.config.save()
            return
        if key == "capture_audio":
            self.config.capture.audio = bool(value)
            self.config.save()
            return
        if key == "models.tier":
            if value not in ("free", "pro"):
                log.warning("ignoring models.tier=%r", value)
                return
            self.config.models.tier = value
            self._models_changed()
            return
        if key == "models.transcription":
            self.config.models.transcription = str(value or "")
            self._models_changed()
            return
        if key == "models.cleanup":
            self.config.models.cleanup = str(value or "")
            self._models_changed()
            return
        if key == "models.groq_paid":
            self.config.models.groq_paid = bool(value)
            self._models_changed()
            return
        if key == "stt.languages":
            codes = [str(c) for c in (value or []) if c and c != "auto"]
            self.config.stt.languages = codes
            self.config.stt.language = codes[0] if codes else ""
            self.config.save()
            log.info("languages: %s", ", ".join(codes) or "detect")
            self._events.put(("engines", None))
            return
        if key == "close_to_tray":
            self.config.ui.close_to_tray = bool(value)
        elif key == "duck_others":
            self.config.audio.duck_others = bool(value)
        elif key == "log_transcripts":
            self.config.log_transcripts = bool(value)
        elif key == "injection.method":
            self.config.injection.method = str(value)
        elif key == "audio.input_device":
            # Reopens the stream and saves on its own, so it returns early.
            self._set_input_device(value)
            return
        else:
            # A switch wired to a key nothing handles would look like it worked
            # and silently do nothing. Say so instead of writing the config.
            log.warning("no handler for setting %r; ignoring", key)
            return
        self.config.save()

    def _models_changed(self) -> None:
        self.config.save()
        log.info("models: tier=%s transcription=%s cleanup=%s groq_paid=%s",
                 self.config.models.tier, self.config.models.transcription or "auto",
                 self.config.models.cleanup or "auto", self.config.models.groq_paid)
        self._events.put(("engines", None))

    def _set_input_device(self, value) -> None:
        """Swap microphones live. The UI hands back None for System default."""
        from .audio.devices import describe

        device = None if value in (None, "") else int(value)
        try:
            self.recorder.reopen(device)
        except Exception as exc:
            log.error("could not open %s: %s", describe(device), exc)
            self.window.set_mic_status(f"Could not open that microphone: {exc}", ok=False)
            self.window.select_input_device(self.config.audio.input_device)
            return
        self.config.audio.input_device = device
        self.config.save()
        log.info("input device: %s", describe(device))
        self.window.set_mic_status(f"Listening on {describe(device)}.", ok=True)

    def _set_mode(self, mode: str) -> None:
        self.config.hotkey.mode = mode
        self.config.save()
        log.info("hotkey mode: %s", mode)

    def _begin_rebind(self) -> None:
        if self.hotkeys is None:
            return

        def done(combo: str) -> None:
            self.hotkeys.rebind(combo)
            self.config.hotkey.trigger = combo
            self.config.save()
            self._events.put(("rebound", combo))

        self.hotkeys.capture(done)

    def _start_transform(self, kind: str) -> None:
        text = self.window.scratch_text()
        if not text:
            self.window.set_transform_status("Scratchpad is empty.")
            return
        self.window.set_transform_status("Working…", busy=True)
        threading.Thread(target=self._run_transform, args=(kind, text),
                         name="openflow-transform", daemon=True).start()

    def _run_transform(self, kind: str, text: str) -> None:
        from .llm.base import ProviderError
        from .llm.providers import build_provider

        prompt = TRANSFORM_PROMPTS.get(kind, TRANSFORM_PROMPTS["formal"])
        for name in llm_chain(self.config):
            if name == "rules":
                continue
            try:
                provider = build_provider(name, self.config)
            except ValueError:
                continue
            if not provider.available():
                continue
            try:
                out = provider.complete(prompt, text, strict=False)
            except ProviderError as exc:
                log.warning("transform via %s failed: %s", name, exc)
                continue
            self._events.put(("transformed", (out, f"Done, via {name}.")))
            return
        self._events.put(
            ("transformed",
             (None, "No AI engine available — add a Gemini key or install Ollama."))
        )

    def _refresh_engines(self) -> None:
        rows: list[tuple[str, str, bool, str]] = []
        for name in stt_chain(self.config):
            try:
                engine = build_engine(name, self.config)
            except ValueError:
                continue
            rows.append(("SPEECH", name, engine.available(),
                         "local" if engine.is_local else "cloud"))
        for name in llm_chain(self.config):
            if name == "rules":
                rows.append(("CLEANUP", "rules", True, "built in"))
                continue
            try:
                from .llm.providers import build_provider

                provider = build_provider(name, self.config)
            except ValueError:
                continue
            rows.append(("CLEANUP", name, provider.available(), ""))
        self.window.set_engines(rows)

    # -- hotkey callbacks (pynput thread; ducker lives here on purpose) ----
    def _on_hotkey_start(self) -> None:
        self.injector.remember_focus()
        # Identify the target app now, while it still has focus. By the time
        # the worker finishes, the foreground window may be ours.
        self._target_app, self._target_title = foreground_window()
        fmt = self.config.formatting
        self._caret = CaretReader().start() if fmt.smart and fmt.context_aware else None
        self._context = (ContextReader(self._target_app, self._target_title).start()
                         if fmt.context_aware else None)
        if self.config.audio.start_sound:
            chime.play()
        self.recorder.start()
        if self.config.audio.duck_others:
            self.ducker.duck()
        self._events.put(("show", "recording"))

    def _on_hotkey_stop(self) -> None:
        audio = self.recorder.stop()
        # Handed to the worker with the audio: by the time it runs, the next
        # hotkey press may already have started a new reader.
        context, self._context = self._context, None
        if self.config.audio.start_sound:
            chime.play_stop()
        # Restore audio the moment the key is released -- the user's music
        # should come back while transcription is still running.
        self.ducker.restore()
        self._events.put(("state", "transcribing"))
        self._jobs.put(("dictation", (audio, context)))

    # -- command mode ------------------------------------------------------
    def _on_command_start(self) -> None:
        """Hold the command combo: record an instruction, not text."""
        self.injector.remember_focus()
        self._target_app, self._target_title = foreground_window()
        # Always read the caret here, whatever the formatting setting says:
        # Command Mode is *about* the selection, not about spacing.
        self._command_caret = CaretReader().start()
        self._context = ContextReader(self._target_app, self._target_title).start()
        if self.config.audio.start_sound:
            chime.play()
        self.recorder.start()
        if self.config.audio.duck_others:
            self.ducker.duck()
        self._events.put(("show", "recording"))

    def _on_command_stop(self) -> None:
        audio = self.recorder.stop()
        context, self._context = self._context, None
        if self.config.audio.start_sound:
            chime.play_stop()
        self.ducker.restore()
        self._events.put(("state", "transcribing"))
        self._jobs.put(("command", (audio, context)))

    def _on_hotkey_cancel(self) -> None:
        self._context = None
        self.recorder.cancel()
        self.ducker.restore()
        self._events.put(("hide", None))

    # -- Qt tick -----------------------------------------------------------
    def _tick(self) -> None:
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            self._handle(kind, payload)

        self._check_hotkeys_alive()
        self._check_permissions()

        if self.window.isVisible():
            self.window.push_input_level(self.recorder.level)

        if self.overlay.state == "recording":
            self.overlay.push_level(self.recorder.level)
            if self.recorder.over_limit:
                log.warning("recording hit max_seconds; finishing early")
                self._on_hotkey_stop()
        self.overlay.tick()

    def _check_permissions(self) -> None:
        """Once macOS grants access, restart the listener so the hotkey works
        in every app straight away -- pynput decides what its event tap can
        see when it starts, so the running one would stay blind."""
        if not self._awaiting_permissions or self.hotkeys is None:
            return
        now = time.monotonic()
        if now - self._permissions_checked_at < 1.0:
            return
        self._permissions_checked_at = now
        if not macos_permissions.trusted():
            return
        self._awaiting_permissions = False
        log.info("permissions granted; restarting the hotkey listener")
        self.hotkeys.stop()
        try:
            self.hotkeys.start()
        except HotkeyUnavailable as exc:
            log.error("hotkey listener unavailable: %s", exc)

    def _check_hotkeys_alive(self) -> None:
        """A dead pynput thread is indistinguishable from an idle app: no
        error, no keys. Say so once, rather than letting it look healthy."""
        if self._hotkeys_reported_dead or self.hotkeys is None:
            return
        if self.hotkeys.alive:
            return
        self._hotkeys_reported_dead = True
        log.error("hotkey listener stopped; dictation will not respond until restart")
        self.window.set_state("error")

    def _handle(self, kind: str, payload) -> None:
        if kind == "show":
            self.overlay.show_pill(payload)
            self.window.set_state(payload)
            self.tray.set_state(payload, self.paused)
        elif kind == "state":
            self.overlay.set_state(payload)
            self.window.set_state(payload)
            self.tray.set_state(payload, self.paused)
        elif kind == "hide":
            self.overlay.hide_pill()
            state = "paused" if self.paused else "ready"
            self.window.set_state(state)
            self.tray.set_state(state, self.paused)
        elif kind == "done":
            self.window.refresh()
        elif kind == "scratch":
            self.window.append_scratch(payload)
        elif kind == "transformed":
            text, message = payload
            if text is not None:
                self.window.replace_scratch(text)
            self.window.set_transform_status(message, busy=False)
        elif kind == "rebound":
            self.window.capture_finished(payload)
        elif kind == "flash":
            # A short colored pulse on the pill. The overlay renders bars, not
            # text, so the state colour is the whole message.
            self.overlay.show_pill(str(payload))
            self.overlay.set_state(str(payload))
            threading.Timer(1.2, lambda: self._events.put(("hide", None))).start()
        elif kind == "update":
            self.window.show_update(payload)
        elif kind == "engines":
            self._refresh_engines()
        elif kind == "greeting":
            self.window.refresh_greeting()
        elif kind == "show_window":
            self.show_window()
        elif kind == "toggle_pause":
            self.toggle_pause()
        elif kind == "toggle_texting":
            self.set_texting(not self.config.formatting.texting)
        elif kind == "quit":
            self.quit()

    # -- worker ------------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            kind, payload = job if isinstance(job, tuple) else ("dictation", job)
            audio, reader = payload if isinstance(payload, tuple) else (payload, None)
            try:
                if kind == "command":
                    self._process_command(audio, reader)
                else:
                    self._process(audio, reader)
            except Exception:
                log.exception("%s failed", kind)
                self._flash_error("Failed")

    def _process(self, audio, reader: ContextReader | None = None) -> None:
        rate = self.config.audio.sample_rate
        if audio is None or len(audio) < rate * MIN_AUDIO_S:
            self._events.put(("hide", None))
            return

        started = time.perf_counter()
        duration_s = len(audio) / rate
        # Measure before conditioning: condition() normalizes to a fixed peak,
        # after which every clip looks equally loud and the level tells you
        # nothing about whether anyone actually spoke.
        rms_before, peak_before = measure(audio)
        voiced_before = voiced_seconds(audio, rate)
        if is_silent(audio, rate):
            # A stray tap of the hotkey, or a dead microphone. Log the levels:
            # if this ever fires on real speech, the numbers say so immediately.
            log.info("no signal (rms=%.6f peak=%.6f, %.1fs); discarding",
                     rms_before, peak_before, duration_s)
            self._events.put(("hide", None))
            return
        audio = condition(audio, rate)
        scratch = self.window.scratch_mode
        context = EMPTY_CONTEXT if scratch or reader is None else reader.result()
        context = with_own_name(context, self.config.ui.display_name)
        if context.names or context.files:
            log.info("context: %d names on screen, %d project files, %d identifiers",
                     len(context.names), len(context.files), len(context.identifiers))
        try:
            transcript = self.stt.transcribe(audio, rate, context=context)
        except SttError as exc:
            log.error("transcription failed: %s", exc)
            self._flash_error("No transcription")
            return

        if not transcript.text.strip():
            self._events.put(("hide", None))
            return

        # Whisper-family models answer silence with subtitle boilerplate --
        # "Thank you.", "Thanks for watching!". Pasting that into someone's
        # document is worse than pasting nothing.
        if is_silence_hallucination(transcript.text, duration_s, rms_before,
                                    voiced_before):
            log.info("discarding likely silence hallucination %r "
                     "(%.1fs, rms=%.4f, voiced=%.2fs)", transcript.text.strip(),
                     duration_s, rms_before, voiced_before)
            self._events.put(("hide", None))
            return

        fmt = self.config.formatting
        kind = classify("" if scratch else self._target_app,
                        "" if scratch else self._target_title, fmt.apps)
        result = self.cleaner.clean(transcript.text, context=context,
                                    category=kind.category, language=transcript.language)
        final = self.personal.apply(result.text)
        # Names as the screen spells them: "sidney" -> "Sydney" when Sydney is
        # in the thread being replied to.
        final = repair_names(final, context.names)
        # Replay what the user has already taught us. Last, so a correction
        # always wins: it is the one edit we know this speaker made by hand.
        final = self.corrections.apply(final)
        if context.is_ide and context.files:
            final = tag_files(final, context.file_names())

        # Lay the words out for where they land: lists, digits, the app
        # category's Flow Style, and the text already around the caret.
        profile = (profile_for(self._target_app, self.config.profiles.apps)
                   if self.config.profiles.enabled and not scratch else DEFAULT)
        style = fmt.styles.get(kind.category, "formal")
        caret = None if scratch or self._caret is None else self._caret.result()
        shaped = smart_format(
            final, kind=kind, style=style, profile=profile, context=caret,
            smart=fmt.smart, protected_terms=tuple(self.personal.dictionary),
            texting=fmt.texting,
        )
        if shaped != final:
            log.info("formatted for %s: %s, %s style, profile %s, caret context %s",
                     self._target_app or "unknown app", kind.category, style,
                     profile.name, "yes" if caret else "no")
        inserted = shaped
        # What the user reads back in history and "Fix last": the words, not
        # the spacing that fitted them to the caret.
        final = shaped.strip()

        self._events.put(("state", "injecting"))
        if scratch:
            self._events.put(("scratch", final))
        else:
            self.injector.inject(inserted)
        self._events.put(("hide", None))

        total_ms = (time.perf_counter() - started) * 1000

        record_id = self.capture.record(
            raw=transcript.text, cleaned=final,
            stt_engine=transcript.engine, clean_engine=result.engine,
            duration_s=duration_s, rms=rms_before, latency_ms=total_ms,
            strategies=[r.strategy for r in result.retractions],
            fillers_removed=result.fillers_removed,
            repetitions_collapsed=result.repetitions_collapsed,
            uncertain=result.uncertain,
            audio=audio, sample_rate=rate,
        )
        # Held for the undo hotkey. The raw transcript is what we restore, so a
        # bad cleanup is one keystroke away from the words actually spoken.
        self._last_insertion = (transcript.text, final, record_id, time.monotonic())
        self.history.add(
            Entry(
                at=time.time(),
                words=len(final.split()),
                chars=len(final),
                duration_s=duration_s,
                latency_ms=total_ms,
                stt_engine=transcript.engine,
                clean_engine=result.engine,
                retractions=len(result.retractions),
                text=final if self.config.log_transcripts else "",
            ),
            dict_fixes=self.personal.last_fixes + result.autofixes,
        )
        self._events.put(("done", None))

        log.info(
            "dictated %d chars in %.0f ms (stt=%s %.0f ms, clean=%s %.0f ms, %d retractions)",
            len(final), total_ms, transcript.engine, transcript.latency_ms,
            result.engine, result.latency_ms, len(result.retractions),
        )

    def _process_command(self, audio, reader: ContextReader | None = None) -> None:
        """Run a spoken Command Mode instruction.

        Deliberately narrow: an edit rewrites the selection, a search opens a
        browser, anything else does nothing but say so. Words spoken here never
        reach a search engine unless the user named one -- see commands.py.
        """
        rate = self.config.audio.sample_rate
        if audio is None or len(audio) < rate * MIN_AUDIO_S:
            self._events.put(("hide", None))
            return

        audio = condition(audio, rate)
        context = reader.result() if reader is not None else EMPTY_CONTEXT
        try:
            transcript = self.stt.transcribe(audio, rate, context=context)
        except SttError as exc:
            log.error("command transcription failed: %s", exc)
            self._flash_error("No transcription")
            return

        spoken = self.personal.apply(transcript.text.strip())
        if not spoken:
            self._events.put(("hide", None))
            return

        caret = self._command_caret.result(0.6) if self._command_caret else None
        selection = caret.selected if caret and caret.has_selection else ""
        if not selection and self.config.commands.clipboard_fallback:
            # Electron apps expose no text to accessibility, and that is where
            # most selections live. Copying is the only way to see them.
            selection = copy_selection()

        command = parse_command(spoken, selection=selection)
        log.info("command mode heard %r -> %s", spoken, command.kind)

        if command.kind == "search":
            import webbrowser

            log.info("opening %s", command.url)
            webbrowser.open(command.url)
            self._events.put(("flash", "injecting"))
            return

        if command.kind == "edit":
            if not selection.strip():
                log.info("command needs a selection; nothing was selected")
                self._flash_error("Select some text first")
                return
            self._run_command_edit(command.instruction, selection)
            return

        # Wispr does nothing here. Saying so is the difference between "that
        # was not a command" and "the hotkey is broken".
        log.info("not a command: needs \"hey flow ...\" or a search engine")
        self._flash_error("Not a command")

    def _run_command_edit(self, instruction: str, selection: str) -> None:
        from .llm.base import ProviderError
        from .llm.providers import build_provider

        self._events.put(("state", "transcribing"))
        user = f"INSTRUCTION: {instruction}\n\nTEXT:\n{selection}"
        for name in llm_chain(self.config):
            if name == "rules":
                continue
            try:
                provider = build_provider(name, self.config)
            except ValueError:
                continue
            if not provider.available():
                continue
            try:
                # strict=False: this is a rewrite the user asked for, so the
                # containment guard that protects dictation would reject it.
                out = provider.complete(EDIT_PROMPT, user, strict=False)
            except ProviderError as exc:
                log.warning("command edit via %s failed: %s", name, exc)
                continue
            self._events.put(("state", "injecting"))
            # Typing over a selection replaces it, which is the whole gesture.
            self.injector.inject(out)
            self._events.put(("hide", None))
            log.info("command edited %d chars via %s", len(selection), name)
            return

        self._flash_error("No AI engine")

    def _flash_error(self, message: str) -> None:
        self._events.put(("state", "error"))
        threading.Timer(1.6, lambda: self._events.put(("hide", None))).start()
