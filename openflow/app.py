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

from .audio.conditioning import condition, is_silent, measure
from .audio.ducker import AudioDucker
from .audio.recorder import AudioUnavailable, Recorder
from .config import Config
from .history import Entry, History
from .input.hotkeys import HotkeyListener, HotkeyUnavailable
from .input.injector import Injector
from .llm.cleaner import LLMCleaner
from .llm.quota import QuotaLedger
from .personalization import shared as personalization
from .stt.engines import SttError, build_engine
from .stt.router import SttRouter

log = logging.getLogger(__name__)

FRAME_MS = 16          # ~60 fps pill animation
MIN_AUDIO_S = 0.25     # shorter than this is a mis-press, not speech

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
        self.hotkeys: HotkeyListener | None = None

        self.window = None
        self.overlay = None
        self.tray = None
        self.single_instance = None
        self.paused = False
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

        # Claim the lock before opening the microphone or binding the hotkey:
        # a second instance must not touch either. If one is already running it
        # gets raised instead, so clicking the shortcut behaves as expected.
        from .single_instance import SingleInstance

        self.single_instance = SingleInstance()
        if not self.single_instance.acquire():
            log.info("OpenFlow is already running; showing that window instead")
            return 0
        self.single_instance.activate_requested.connect(self.show_window)

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
                "transform": self._start_transform,
            },
        )
        self.overlay = Overlay(self.config.ui)
        self.tray = Tray(
            on_show=lambda: self._events.put(("show_window", None)),
            on_toggle_pause=lambda: self._events.put(("toggle_pause", None)),
            on_quit=lambda: self._events.put(("quit", None)),
        )

        try:
            self.recorder.open()
        except AudioUnavailable as exc:
            log.error("microphone unavailable: %s", exc)
            return 2

        threading.Thread(target=self._worker, name="openflow-worker", daemon=True).start()
        threading.Thread(target=self._warm, name="openflow-warm", daemon=True).start()

        self.hotkeys = HotkeyListener(
            self.config.hotkey,
            on_start=self._on_hotkey_start,
            on_stop=self._on_hotkey_stop,
            on_cancel=self._on_hotkey_cancel,
        )
        try:
            self.hotkeys.start()
        except HotkeyUnavailable as exc:
            log.error("hotkey listener unavailable: %s", exc)
            return 2

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

    def _warm(self) -> None:
        self.stt.warm()
        self._warm_llm()
        self._events.put(("engines", None))

    def _warm_llm(self) -> None:
        """Load the local model's weights before the first dictation needs
        them -- a cold Ollama start costs about six seconds."""
        if not self.config.llm.enabled or "ollama" not in self.config.llm.backends:
            return
        try:
            from .llm.providers import OllamaProvider

            provider = OllamaProvider(self.config)
            if provider.available():
                provider.complete("Reply with the single word: ok", "ok", strict=False)
                log.info("ollama warm (%s)", provider.model)
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
        if self.hotkeys:
            self.hotkeys.paused = self.paused
        if self.paused:
            self.recorder.cancel()
            self.ducker.restore()
        state = "paused" if self.paused else "ready"
        self.window.set_state(state)
        self.tray.set_state(state, self.paused)

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
        if key == "close_to_tray":
            self.config.ui.close_to_tray = bool(value)
        elif key == "duck_others":
            self.config.audio.duck_others = bool(value)
        elif key == "log_transcripts":
            self.config.log_transcripts = bool(value)
        elif key == "injection.method":
            self.config.injection.method = str(value)
        self.config.save()

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
        for name in self.config.llm.backends:
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
        for name in self.config.stt.backends:
            try:
                engine = build_engine(name, self.config)
            except ValueError:
                continue
            rows.append(("SPEECH", name, engine.available(),
                         "local" if engine.is_local else "cloud"))
        for name in self.config.llm.backends:
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
        self.recorder.start()
        if self.config.audio.duck_others:
            self.ducker.duck()
        self._events.put(("show", "recording"))

    def _on_hotkey_stop(self) -> None:
        audio = self.recorder.stop()
        # Restore audio the moment the key is released -- the user's music
        # should come back while transcription is still running.
        self.ducker.restore()
        self._events.put(("state", "transcribing"))
        self._jobs.put(audio)

    def _on_hotkey_cancel(self) -> None:
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

        if self.overlay.state == "recording":
            self.overlay.push_level(self.recorder.level)
            if self.recorder.over_limit:
                log.warning("recording hit max_seconds; finishing early")
                self._on_hotkey_stop()
        self.overlay.tick()

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
        elif kind == "engines":
            self._refresh_engines()
        elif kind == "show_window":
            self.show_window()
        elif kind == "toggle_pause":
            self.toggle_pause()
        elif kind == "quit":
            self.quit()

    # -- worker ------------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                audio = self._jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._process(audio)
            except Exception:
                log.exception("dictation failed")
                self._flash_error("Failed")

    def _process(self, audio) -> None:
        rate = self.config.audio.sample_rate
        if audio is None or len(audio) < rate * MIN_AUDIO_S:
            self._events.put(("hide", None))
            return

        started = time.perf_counter()
        duration_s = len(audio) / rate
        if is_silent(audio, rate):
            # A stray tap of the hotkey, or a dead microphone. Log the levels:
            # if this ever fires on real speech, the numbers say so immediately.
            rms, peak = measure(audio)
            log.info("no signal (rms=%.6f peak=%.6f, %.1fs); discarding",
                     rms, peak, duration_s)
            self._events.put(("hide", None))
            return
        audio = condition(audio, rate)
        try:
            transcript = self.stt.transcribe(audio, rate)
        except SttError as exc:
            log.error("transcription failed: %s", exc)
            self._flash_error("No transcription")
            return

        if not transcript.text.strip():
            self._events.put(("hide", None))
            return

        result = self.cleaner.clean(transcript.text)
        final = self.personal.apply(result.text)
        self._events.put(("state", "injecting"))
        if self.window.scratch_mode:
            self._events.put(("scratch", final))
        else:
            self.injector.inject(final)
        self._events.put(("hide", None))

        total_ms = (time.perf_counter() - started) * 1000
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

    def _flash_error(self, message: str) -> None:
        self._events.put(("state", "error"))
        threading.Timer(1.6, lambda: self._events.put(("hide", None))).start()
