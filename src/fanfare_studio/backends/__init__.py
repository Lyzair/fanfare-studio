"""TTS backends. Import heavy dependencies lazily so ``mock`` works without them."""

from __future__ import annotations

from .base import Backend, BackendError, UnknownVoiceError, Voice

BACKENDS = ("chatterbox-onnx", "chatterbox", "piper", "mock")
DEFAULT_BACKEND = "chatterbox-onnx"  # Chatterbox, 4-bit ONNX LM, CPU EP: best-sounding path


def create_backend(name: str, **opts) -> Backend:  # type: ignore[no-untyped-def]
    if name == "mock":
        from .mock import MockBackend

        return MockBackend()
    if name == "piper":
        from .piper_backend import PiperBackend

        return PiperBackend(voices_dir=opts["voices_dir"], ep=opts["ep"])
    if name == "chatterbox":
        from .chatterbox_backend import ChatterboxBackend

        return ChatterboxBackend(
            device=opts["device"],
            voices_config=opts.get("voices_config"),
            math_sdp=opts.get("math_sdp", False),
        )
    if name == "chatterbox-onnx":
        from .chatterbox_onnx_backend import ChatterboxOnnxBackend

        return ChatterboxOnnxBackend(
            ep=opts["ep"],
            variant=opts.get("onnx_variant", "q4"),
            voices_config=opts.get("voices_config"),
        )
    raise ValueError(f"unknown backend {name!r}")


__all__ = ["BACKENDS", "DEFAULT_BACKEND", "Backend", "BackendError", "UnknownVoiceError", "Voice", "create_backend"]
