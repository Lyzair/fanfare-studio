"""Chatterbox Multilingual (Resemble AI, MIT) on PyTorch — EXPERIMENTAL.

CUDA, ROCm, MPS or CPU. Listening tests found it worse than the 4-bit ONNX backend (see
docs/model-choice.md); kept for comparison and for GPUs. Same per-voice settings as the
ONNX backend (voice_config.py). Every clip carries Resemble's PerTh watermark.
"""

from __future__ import annotations

import logging
import time

from ..runtime import torch_device
from ..voice_config import Settings, VoiceEntry, load_voices
from ..wav import encode_numpy
from .base import BackendError, UnknownVoiceError, Voice

log = logging.getLogger("fanfare_studio")


class ChatterboxBackend:
    name = "chatterbox"

    def __init__(self, device: str = "auto", voices_config: str | None = None, math_sdp: bool = False) -> None:
        import torch

        dev, desc = torch_device(device)
        if math_sdp and dev == "cuda":
            # Diagnostic: disable fused attention kernels (flash / mem-efficient), keep math SDP.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            desc += ", math SDP only"
        self.runtime = desc
        log.info("chatterbox: loading model on %s", desc)
        t0 = time.perf_counter()
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._model = ChatterboxMultilingualTTS.from_pretrained(device=dev)
        self._builtin_conds = self._model.conds
        self._ref_conds: dict[str, object] = {}
        log.info("chatterbox: model ready in %.1fs", time.perf_counter() - t0)
        self._entries: dict[str, VoiceEntry] = {e.voice.id: e for e in load_voices(voices_config)}

    def voices(self) -> list[Voice]:
        return [e.voice for e in self._entries.values()]

    def settings(self, voice_id: str) -> Settings:
        return self._entries[voice_id].settings

    def synthesize(self, text: str, voice_id: str, settings: Settings | None = None) -> bytes:
        entry = self._entries.get(voice_id)
        if entry is None:
            raise UnknownVoiceError(voice_id)
        s = settings or entry.settings
        try:
            import torch

            m = self._model
            if s.reference:
                if s.reference not in self._ref_conds:
                    m.prepare_conditionals(s.reference, exaggeration=s.exaggeration)
                    self._ref_conds[s.reference] = m.conds
                m.conds = self._ref_conds[s.reference]
            else:
                m.conds = self._builtin_conds
            if s.seed is not None:
                torch.manual_seed(s.seed)
            with torch.inference_mode():
                wav = m.generate(
                    text,
                    language_id=entry.voice.lang,
                    exaggeration=s.exaggeration,
                    cfg_weight=s.cfg_weight,
                    temperature=max(s.temperature, 1e-3),  # torch path always samples
                    repetition_penalty=s.repetition_penalty,
                    min_p=s.min_p,
                    top_p=s.top_p,
                )
            return encode_numpy(wav.detach().cpu().float().numpy(), m.sr)
        except Exception as e:
            raise BackendError(str(e)) from e
