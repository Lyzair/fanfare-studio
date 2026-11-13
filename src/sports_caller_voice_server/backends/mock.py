"""Mock backend: a short sine beep per request. For tests and client development."""

from __future__ import annotations

import math

from .. import LANGS
from ..wav import encode_wav
from .base import UnknownVoiceError, Voice

SAMPLE_RATE = 16000


class MockBackend:
    name = "mock"
    runtime = "none (mock)"

    def __init__(self) -> None:
        self._voices = [Voice(f"mock-{lang}", lang, f"Mock beep ({lang})") for lang in LANGS]
        self._ids = {v.id for v in self._voices}

    def voices(self) -> list[Voice]:
        return list(self._voices)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        if voice_id not in self._ids:
            raise UnknownVoiceError(voice_id)
        # Length grows with the text so clients can see different clips; capped at 3 s.
        seconds = min(3.0, 0.2 + 0.02 * len(text))
        n = int(SAMPLE_RATE * seconds)
        return encode_wav(
            (0.2 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE) for i in range(n)), SAMPLE_RATE
        )
