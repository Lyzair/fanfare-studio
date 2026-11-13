from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Voice:
    id: str
    lang: str  # one of LANGS
    label: str


class UnknownVoiceError(KeyError):
    """The requested voice id is not offered by the backend (HTTP 404)."""


class BackendError(RuntimeError):
    """The model failed to synthesize (HTTP 500)."""


class Backend(Protocol):
    name: str
    runtime: str  # human-readable device / execution provider actually in use

    def voices(self) -> list[Voice]: ...

    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Return a complete WAV file. Raise UnknownVoiceError or BackendError."""
        ...
