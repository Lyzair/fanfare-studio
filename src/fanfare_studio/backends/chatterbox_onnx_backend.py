"""Chatterbox Multilingual via ONNX Runtime (community export, MIT).

Portable path for machines without a CUDA/ROCm/MPS PyTorch: CPU everywhere, and
DirectML / OpenVINO / CoreML / QNN / Vitis AI through ONNX Runtime execution
providers (``--ep``). Adapted from the reference script on the model card of
https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX (MIT).

Differences from the PyTorch backend: greedy decoding without classifier-free
guidance, so prosody can be flatter; PerTh watermark only if ``perth`` is
installed.
"""

from __future__ import annotations

import logging
import time

from ..runtime import pick_providers
from ..wav import encode_numpy
from .base import BackendError, UnknownVoiceError, Voice
from .chatterbox_backend import VOICES

log = logging.getLogger("fanfare_studio")

MODEL_ID = "onnx-community/chatterbox-multilingual-ONNX"
SAMPLE_RATE = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM = 30, 16, 64
LM_VARIANTS = {"fp32": "language_model", "fp16": "language_model_fp16", "q4": "language_model_q4"}


class ChatterboxOnnxBackend:
    name = "chatterbox-onnx"

    def __init__(self, ep: str = "auto", variant: str = "q4", max_new_tokens: int = 800) -> None:
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self._np = np
        self.max_new_tokens = max_new_tokens
        providers = pick_providers(ep, ort.get_available_providers())

        def fetch(stem: str) -> str:
            path = hf_hub_download(MODEL_ID, f"{stem}.onnx", subfolder="onnx")
            hf_hub_download(MODEL_ID, f"{stem}.onnx_data", subfolder="onnx")
            return path

        def session(stem: str):  # type: ignore[no-untyped-def]
            return ort.InferenceSession(fetch(stem), providers=providers)

        self._speech_encoder = session("speech_encoder")
        self._embed = session("embed_tokens")
        self._lm = session(LM_VARIANTS[variant])
        self._decoder = session("conditional_decoder")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        active = self._lm.get_providers()[0]
        self.runtime = f"ONNX Runtime {ort.__version__} {active} (lm={variant})"
        log.info("chatterbox-onnx: requested ep=%s, active provider %s", ep, active)

        # Speaker conditioning from the bundled default voice; computed once.
        import librosa

        ref, _ = librosa.load(hf_hub_download(MODEL_ID, "default_voice.wav"), sr=SAMPLE_RATE)
        self._cond = self._speech_encoder.run(None, {"audio_values": ref[np.newaxis, :].astype(np.float32)})

        try:
            import perth

            self._watermarker = perth.PerthImplicitWatermarker()
        except Exception:  # noqa: BLE001 - optional
            self._watermarker = None
        self._voices = {v.id: v for v in VOICES}

    def voices(self) -> list[Voice]:
        return list(self._voices.values())

    def synthesize(self, text: str, voice_id: str) -> bytes:
        voice = self._voices.get(voice_id)
        if voice is None:
            raise UnknownVoiceError(voice_id)
        np = self._np
        try:
            t0 = time.perf_counter()
            cond_emb, prompt_token, spk_emb, spk_feat = self._cond
            ids = self._tokenizer(f"[{voice.lang}]{text}", return_tensors="np")["input_ids"].astype(np.int64)
            pos = np.where(ids >= START_SPEECH_TOKEN, 0, np.arange(ids.shape[1])[np.newaxis, :] - 1)
            feed = {"input_ids": ids, "position_ids": pos.astype(np.int64),
                    "exaggeration": np.array([0.5], dtype=np.float32)}
            generated = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
            past = None
            mask = None
            for i in range(self.max_new_tokens):
                emb = self._embed.run(None, feed)[0]
                if past is None:
                    emb = np.concatenate((cond_emb, emb), axis=1)
                    past = {f"past_key_values.{layer}.{kv}": np.zeros([1, NUM_KV_HEADS, 0, HEAD_DIM], dtype=np.float32)
                            for layer in range(NUM_LAYERS) for kv in ("key", "value")}
                    mask = np.ones((1, emb.shape[1]), dtype=np.int64)
                logits, *present = self._lm.run(None, dict(inputs_embeds=emb, attention_mask=mask, **past))
                logits = logits[:, -1, :].astype(np.float32)
                # repetition penalty 1.2 (as in the reference script)
                prev = np.take_along_axis(logits, generated, axis=1)
                np.put_along_axis(logits, generated, np.where(prev < 0, prev * 1.2, prev / 1.2), axis=1)
                nxt = np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)
                generated = np.concatenate((generated, nxt), axis=-1)
                if int(nxt[0, 0]) == STOP_SPEECH_TOKEN:
                    break
                feed["input_ids"] = nxt
                feed["position_ids"] = np.full((1, 1), i + 1, dtype=np.int64)
                mask = np.concatenate([mask, np.ones((1, 1), dtype=np.int64)], axis=1)
                for j, key in enumerate(past):
                    past[key] = present[j]
            tokens = generated[:, 1:]
            if tokens.shape[1] and int(tokens[0, -1]) == STOP_SPEECH_TOKEN:
                tokens = tokens[:, :-1]
            tokens = np.concatenate([prompt_token, tokens], axis=1)
            wav = self._decoder.run(None, {"speech_tokens": tokens, "speaker_embeddings": spk_emb,
                                           "speaker_features": spk_feat})[0]
            wav = np.squeeze(wav, axis=0)
            if self._watermarker is not None:
                wav = self._watermarker.apply_watermark(wav, sample_rate=SAMPLE_RATE)
            log.info("chatterbox-onnx: %d speech tokens in %.1fs", tokens.shape[1], time.perf_counter() - t0)
            return encode_numpy(wav, SAMPLE_RATE)
        except Exception as e:
            raise BackendError(str(e)) from e
