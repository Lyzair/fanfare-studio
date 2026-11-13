"""Mandatory PerTh watermark on every synthesized clip.

Resemble's PerTh watermarker (``resemble-perth``, a required dependency) embeds an inaudible
marker that identifies a clip as AI-generated. It works on arbitrary speech audio, so it is applied
to *every* backend's output (Chatterbox ONNX/torch, Piper, ...), not only to Chatterbox.

If the watermarker cannot be loaded the server refuses to start (``WatermarkError``): there is
deliberately no "run without watermark" mode.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Protocol

log = logging.getLogger("fanfare_studio")


class WatermarkError(RuntimeError):
    """The watermarker could not be loaded or applied; the server must not emit unmarked audio."""


class _PerthLike(Protocol):
    def apply_watermark(self, wav, sample_rate: int): ...  # noqa: ANN001


class Watermarker:
    def __init__(self, impl: _PerthLike | None = None) -> None:
        if impl is None:
            try:
                import perth

                impl = perth.PerthImplicitWatermarker()
            except Exception as e:  # noqa: BLE001 - any failure must stop the server
                raise WatermarkError(
                    "Cannot load the PerTh watermarker (resemble-perth), and Fanfare Studio never "
                    "produces unwatermarked audio. Reinstall the dependencies (uv sync) or see the "
                    f"README. Cause: {type(e).__name__}: {e}"
                ) from e
        self._impl = impl

    def apply_array(self, wav, sample_rate: int):  # noqa: ANN001, ANN201
        """Watermark a float32 mono numpy array."""
        try:
            return self._impl.apply_watermark(wav, sample_rate=sample_rate)
        except Exception as e:  # noqa: BLE001
            raise WatermarkError(f"watermarking failed: {type(e).__name__}: {e}") from e

    def apply_wav(self, data: bytes) -> bytes:
        """Watermark a mono 16-bit PCM WAV file (as returned by the backends)."""
        import numpy as np

        from .wav import encode_numpy

        with wave.open(io.BytesIO(data), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise WatermarkError("backend returned a WAV that is not mono 16-bit PCM")
            sr = w.getframerate()
            pcm = w.readframes(w.getnframes())
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return encode_numpy(self.apply_array(x, sr), sr)


class WatermarkedBackend:
    """Wraps a backend so every ``synthesize`` result is watermarked. Other attributes pass through."""

    def __init__(self, backend, watermarker: Watermarker) -> None:  # noqa: ANN001
        self._backend = backend
        self._wm = watermarker
        self.watermarked_calls = 0

    def __getattr__(self, name: str):  # noqa: ANN204
        return getattr(self._backend, name)

    def synthesize(self, text: str, voice_id: str, *args, **kwargs) -> bytes:  # noqa: ANN002, ANN003
        from .backends.base import BackendError

        wav = self._backend.synthesize(text, voice_id, *args, **kwargs)
        try:
            out = self._wm.apply_wav(wav)
        except WatermarkError as e:
            raise BackendError(str(e)) from e
        self.watermarked_calls += 1
        return out
