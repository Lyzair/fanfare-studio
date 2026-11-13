"""Mock backend: a short sine beep per request. For tests and client development."""

from __future__ import annotations

import math

from .. import LANGS
from ..voice_config import Settings
from ..wav import encode_wav
from .base import UnknownVoiceError, Voice

SAMPLE_RATE = 16000


class MockBackend:
    name = "mock"
    runtime = "none (mock)"

    def __init__(self) -> None:
        self._voices = {f"mock-{lang}": Voice(f"mock-{lang}", lang, f"Mock beep ({lang})") for lang in LANGS}
        self._settings: dict[str, Settings] = {}
        self.calls: list[tuple[str, Settings]] = []  # (voice_id, settings) per synthesis, for tests

    def voices(self) -> list[Voice]:
        return list(self._voices.values())

    def settings(self, voice_id: str) -> Settings:
        return self._settings.get(voice_id, Settings())

    def add_voice(self, voice: Voice, settings: Settings) -> None:
        self._voices[voice.id] = voice
        self._settings[voice.id] = settings

    def remove_voice(self, voice_id: str) -> None:
        self._voices.pop(voice_id, None)
        self._settings.pop(voice_id, None)

    def synthesize(self, text: str, voice_id: str, settings: Settings | None = None) -> bytes:
        if voice_id not in self._voices:
            raise UnknownVoiceError(voice_id)
        self.calls.append((voice_id, settings or self.settings(voice_id)))
        # Length grows with the text so clients can see different clips; capped at 3 s.
        seconds = min(3.0, 0.2 + 0.02 * len(text))
        n = int(SAMPLE_RATE * seconds)
        return encode_wav(
            (0.2 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE) for i in range(n)), SAMPLE_RATE
        )
