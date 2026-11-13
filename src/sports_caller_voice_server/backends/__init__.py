"""TTS backends. Import heavy dependencies lazily so ``mock`` works without them."""

from __future__ import annotations

from .base import Backend, BackendError, UnknownVoiceError, Voice

BACKENDS = ("mock", "piper", "chatterbox", "chatterbox-onnx")


def create_backend(name: str, **opts) -> Backend:  # type: ignore[no-untyped-def]
    if name == "mock":
        from .mock import MockBackend

        return MockBackend()
    if name == "piper":
        from .piper_backend import PiperBackend

        return PiperBackend(voices_dir=opts["voices_dir"], ep=opts["ep"])
    if name == "chatterbox":
        from .chatterbox_backend import ChatterboxBackend

        return ChatterboxBackend(device=opts["device"])
    if name == "chatterbox-onnx":
        from .chatterbox_onnx_backend import ChatterboxOnnxBackend

        return ChatterboxOnnxBackend(ep=opts["ep"])
    raise ValueError(f"unknown backend {name!r}")


__all__ = ["BACKENDS", "Backend", "BackendError", "UnknownVoiceError", "Voice", "create_backend"]
