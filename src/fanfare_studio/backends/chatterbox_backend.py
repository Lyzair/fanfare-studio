"""Chatterbox Multilingual (Resemble AI, MIT) on PyTorch: CUDA, ROCm, MPS or CPU.

Covers 9 of the 10 Fanfare languages (no Ukrainian). Uses the built-in
default speaker; every clip carries Resemble's inaudible PerTh watermark.
"""

from __future__ import annotations

import logging
import time

from ..runtime import torch_device
from ..wav import encode_numpy
from .base import BackendError, UnknownVoiceError, Voice

log = logging.getLogger("fanfare_studio")

_LANG_NAMES = {
    "de": "German", "en": "English", "fr": "French", "es": "Spanish", "it": "Italian",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "ar": "Arabic",
}  # fmt: skip
VOICES = [Voice(f"{lang}-high", lang, f"{name} (high quality)") for lang, name in _LANG_NAMES.items()]


class ChatterboxBackend:
    name = "chatterbox"

    def __init__(self, device: str = "auto", exaggeration: float = 0.5, cfg_weight: float = 0.5) -> None:
        dev, desc = torch_device(device)
        self.runtime = desc
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        log.info("chatterbox: loading model on %s", desc)
        t0 = time.perf_counter()
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._model = ChatterboxMultilingualTTS.from_pretrained(device=dev)
        log.info("chatterbox: model ready in %.1fs", time.perf_counter() - t0)
        self._voices = {v.id: v for v in VOICES}

    def voices(self) -> list[Voice]:
        return list(self._voices.values())

    def synthesize(self, text: str, voice_id: str) -> bytes:
        voice = self._voices.get(voice_id)
        if voice is None:
            raise UnknownVoiceError(voice_id)
        try:
            import torch

            with torch.inference_mode():
                wav = self._model.generate(
                    text,
                    language_id=voice.lang,
                    exaggeration=self.exaggeration,
                    cfg_weight=self.cfg_weight,
                )
            return encode_numpy(wav.detach().cpu().float().numpy(), self._model.sr)
        except Exception as e:
            raise BackendError(str(e)) from e
