"""Piper voices (ONNX) through ONNX Runtime: the "works everywhere" tier.

Loads every ``*.onnx`` (+ ``.onnx.json``) in ``--voices-dir``. The voice id is the
file stem (e.g. ``de_DE-thorsten-high``), the language is its prefix. Voices whose
language is not one of the app's LANGS are skipped.

Engine licence: ``piper-tts`` (OHF-Voice/piper1-gpl) is GPL-3.0; it is an optional,
separately installed dependency. Each voice has its own licence (see its MODEL_CARD).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .. import LANGS
from ..runtime import pick_providers
from ..wav import encode_pcm16
from .base import BackendError, UnknownVoiceError, Voice

log = logging.getLogger("fanfare_studio")

_LANG_NAMES = {"de": "German", "en": "English", "fr": "French", "es": "Spanish", "it": "Italian",
               "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "uk": "Ukrainian", "ar": "Arabic"}


class PiperBackend:
    name = "piper"

    def __init__(self, voices_dir: str, ep: str = "auto") -> None:
        import onnxruntime as ort
        from piper import PiperVoice
        from piper.config import PiperConfig

        providers = pick_providers(ep, ort.get_available_providers())
        self._models: dict[str, object] = {}
        self._voices: dict[str, Voice] = {}
        active = "CPUExecutionProvider"
        for model in sorted(Path(voices_dir).glob("*.onnx")):
            lang = model.stem.split("_", 1)[0].split("-", 1)[0]
            if lang not in LANGS:
                continue
            cfg = json.loads(Path(f"{model}.json").read_text(encoding="utf-8"))
            session = ort.InferenceSession(str(model), providers=providers)
            active = session.get_providers()[0]
            self._models[model.stem] = PiperVoice(config=PiperConfig.from_dict(cfg), session=session)
            parts = model.stem.split("-")
            speaker, quality = (parts[1], parts[-1]) if len(parts) >= 3 else (model.stem, "")
            self._voices[model.stem] = Voice(
                model.stem, lang, f"{_LANG_NAMES[lang]} – {speaker} (Piper {quality})".strip()
            )
        if not self._voices:
            raise RuntimeError(f"no Piper voices (*.onnx + *.onnx.json) found in {voices_dir}")
        self.runtime = f"ONNX Runtime {ort.__version__} {active}"
        log.info("piper: requested ep=%s, active provider %s", ep, active)

    def voices(self) -> list[Voice]:
        return list(self._voices.values())

    def synthesize(self, text: str, voice_id: str) -> bytes:
        voice = self._models.get(voice_id)
        if voice is None:
            raise UnknownVoiceError(voice_id)
        try:
            chunks = list(voice.synthesize(text))  # type: ignore[attr-defined]
            if not chunks:
                raise BackendError("no audio produced")
            pcm = b"".join(c.audio_int16_bytes for c in chunks)
            return encode_pcm16(pcm, chunks[0].sample_rate)
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(str(e)) from e
