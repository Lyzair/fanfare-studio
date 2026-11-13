"""Minimal WAV encoding (mono, 16-bit PCM) without third-party dependencies."""

from __future__ import annotations

import io
import struct
import wave
from collections.abc import Iterable


def encode_wav(samples: Iterable[float], sample_rate: int) -> bytes:
    """Encode float samples in [-1, 1] as mono 16-bit PCM WAV bytes."""
    frames = bytearray()
    for s in samples:
        s = max(-1.0, min(1.0, float(s)))
        frames += struct.pack("<h", int(round(s * 32767)))
    return encode_pcm16(bytes(frames), sample_rate)


def encode_pcm16(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw little-endian mono 16-bit PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def encode_numpy(audio, sample_rate: int) -> bytes:  # type: ignore[no-untyped-def]
    """Encode a float numpy array (any shape, flattened) as mono 16-bit PCM WAV."""
    import numpy as np

    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    a = np.clip(a, -1.0, 1.0)
    return encode_pcm16((a * 32767.0).round().astype("<i2").tobytes(), sample_rate)
