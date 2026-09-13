"""Offline speech output.

Uses the speech engine already present in Windows (SAPI5) via pyttsx3, so there is
no speech model to ship, no network call, and no added NPU load. A critical utterance
interrupts whatever is currently being spoken — if the user is about to walk into a
car, the sentence about the chair no longer matters.
"""

from __future__ import annotations

import logging
import queue
import threading

from .config import SpeechConfig
from .policy import Utterance

log = logging.getLogger(__name__)


class Speaker:
    """Serial speech channel with interrupt support, driven off a worker thread."""

    def __init__(self, config: SpeechConfig):
        self.config = config
        self._queue: queue.Queue[Utterance | None] = queue.Queue(maxsize=4)
        self._engine = self._make_engine()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="drishti-speech", daemon=True)
        self._thread.start()

    def _make_engine(self):
        try:
            import pyttsx3
        except ImportError:
            log.warning("pyttsx3 not installed — speech will be printed, not spoken")
            return None
        engine = pyttsx3.init()
        engine.setProperty("rate", self.config.rate_wpm)
        engine.setProperty("volume", self.config.volume)
        self._select_voice(engine)
        return engine

    def _select_voice(self, engine) -> None:
        """Pick an installed system voice matching the configured locale.

        SAPI enumerates whatever voices are present on the machine, which is what
        makes localisation a deployment choice rather than a code change: install a
        Hindi voice through Windows language settings, set the locale, and Drishti
        speaks Hindi. If no matching voice is installed we fall back to the system
        default and say so in the log, rather than failing to speak at all.
        """
        wanted = self.config.voice_hint.split("-")[0].lower()
        try:
            voices = engine.getProperty("voices") or []
        except Exception as exc:  # noqa: BLE001 - driver differences across platforms
            log.warning("could not enumerate voices: %s", exc)
            return

        for voice in voices:
            haystack = " ".join(
                str(part).lower()
                for part in (getattr(voice, "id", ""), getattr(voice, "name", ""),
                             *(getattr(voice, "languages", None) or []))
            )
            if wanted in haystack:
                engine.setProperty("voice", voice.id)
                log.info("speech voice: %s (%s)", getattr(voice, "name", voice.id), wanted)
                return

        log.warning(
            "no installed voice matches locale %r — using the system default. "
            "Install a matching voice via Windows language settings.", wanted
        )

    def say(self, utterance: Utterance) -> None:
        """Queue an utterance. Critical ones clear the queue first."""
        if utterance.interrupt:
            self._drain()
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:  # noqa: BLE001 - stop() is best-effort across drivers
                    pass
        try:
            self._queue.put_nowait(utterance)
        except queue.Full:
            # Backed up: the newest information is the most useful, so drop the oldest.
            self._drain()
            self._queue.put_nowait(utterance)

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                return
            if self._engine is None:
                print(f"[speech] {item.text}")
                continue
            try:
                self._engine.say(item.text)
                self._engine.runAndWait()
            except Exception as exc:  # noqa: BLE001
                log.error("speech synthesis failed: %s", exc)

    def close(self) -> None:
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=2.0)
