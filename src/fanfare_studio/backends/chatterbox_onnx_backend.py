"""Chatterbox Multilingual via ONNX Runtime (community export, MIT). Default backend.

Portable: CPU everywhere, and DirectML / OpenVINO / CoreML / QNN / Vitis AI through ONNX
Runtime execution providers (``--ep``). Generation loop adapted from the reference script
on https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX (MIT), extended to
match the PyTorch package: same text normalisation (punctuation clean-up, lower-case, NFKD —
the reference script skipped this and fed upper-case tokens the model never saw in
training), temperature / min-p / top-p sampling with a seed, classifier-free guidance, and
a per-voice reference clip (see voice_config.py).
"""

from __future__ import annotations

import logging
import time
import unicodedata

from ..runtime import pick_providers
from ..voice_config import Settings, VoiceEntry, load_voices
from ..wav import encode_numpy
from .base import BackendError, UnknownVoiceError, Voice

log = logging.getLogger("fanfare_studio")

MODEL_ID = "onnx-community/chatterbox-multilingual-ONNX"
SAMPLE_RATE = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM = 30, 16, 64
LM_VARIANTS = {"fp32": "language_model", "fp16": "language_model_fp16", "q4": "language_model_q4"}

_PUNC = [("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "), ("—", "-"),
         ("–", "-"), (" ,", ","), ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")]  # fmt: skip


def normalize_text(text: str) -> str:
    """Same clean-up as chatterbox.mtl_tts.punc_norm + MTLTokenizer.preprocess_text."""
    text = " ".join(text.split())
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    for old, new in _PUNC:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not text.endswith((".", "!", "?", "-", ",", "、", "，", "。", "？", "！")):
        text += "."
    return unicodedata.normalize("NFKD", text.lower())


class ChatterboxOnnxBackend:
    name = "chatterbox-onnx"

    def __init__(
        self,
        ep: str = "auto",
        variant: str = "q4",
        voices_config: str | None = None,
        normalize: bool = True,
    ) -> None:
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self._np = np
        self.normalize = normalize
        providers = pick_providers(ep, ort.get_available_providers())

        def session(stem: str):  # type: ignore[no-untyped-def]
            path = hf_hub_download(MODEL_ID, f"{stem}.onnx", subfolder="onnx")
            hf_hub_download(MODEL_ID, f"{stem}.onnx_data", subfolder="onnx")
            return ort.InferenceSession(path, providers=providers)

        self._speech_encoder = session("speech_encoder")
        self._embed = session("embed_tokens")
        self._lm = session(LM_VARIANTS[variant])
        self._decoder = session("conditional_decoder")
        # tokenizer.json directly (works with HF_HUB_OFFLINE=1, unlike AutoTokenizer)
        self._tokenizer = Tokenizer.from_file(hf_hub_download(MODEL_ID, "tokenizer.json"))
        self._default_ref = hf_hub_download(MODEL_ID, "default_voice.wav")
        active = self._lm.get_providers()[0]
        self.runtime = f"ONNX Runtime {ort.__version__} {active} (lm={variant})"
        log.info("chatterbox-onnx: requested ep=%s, active provider %s", ep, active)

        self._entries: dict[str, VoiceEntry] = {e.voice.id: e for e in load_voices(voices_config)}
        self._conds: dict[str, tuple] = {}
        for e in self._entries.values():
            self._conditioning(e.settings.reference)  # load every reference once at start-up
            if e.settings.reference:
                log.info("voice %s: reference %s", e.voice.id, e.settings.reference)

    def voices(self) -> list[Voice]:
        return [e.voice for e in self._entries.values()]

    def settings(self, voice_id: str) -> Settings:
        return self._entries[voice_id].settings

    def add_voice(self, voice: Voice, settings: Settings) -> None:
        self._conditioning(settings.reference)  # fail early on a bad reference
        self._entries[voice.id] = VoiceEntry(voice, settings)

    def remove_voice(self, voice_id: str) -> None:
        entry = self._entries.pop(voice_id, None)
        if entry and entry.settings.reference:
            self._conds.pop(entry.settings.reference, None)

    def _conditioning(self, reference: str | None) -> tuple:
        key = reference or self._default_ref
        if key not in self._conds:
            import librosa

            np = self._np
            ref, _ = librosa.load(key, sr=SAMPLE_RATE)
            self._conds[key] = tuple(
                self._speech_encoder.run(None, {"audio_values": ref[np.newaxis, :].astype(np.float32)})
            )
        return self._conds[key]

    def synthesize(self, text: str, voice_id: str, settings: Settings | None = None) -> bytes:
        entry = self._entries.get(voice_id)
        if entry is None:
            raise UnknownVoiceError(voice_id)
        try:
            wav = self._generate(text, entry.voice.lang, settings or entry.settings)
            return encode_numpy(wav, SAMPLE_RATE)
        except Exception as e:
            raise BackendError(str(e)) from e

    def _generate(self, text: str, lang: str, s: Settings):  # type: ignore[no-untyped-def]
        np = self._np
        t0 = time.perf_counter()
        rng = np.random.default_rng(s.seed)
        cond_emb, prompt_token, spk_emb, spk_feat = self._conditioning(s.reference)
        prepared = f"[{lang}]{normalize_text(text) if self.normalize else text}"
        ids = np.array([self._tokenizer.encode(prepared).ids], dtype=np.int64)
        log.debug("chatterbox-onnx: language token [%s], %d input tokens", lang, ids.shape[1])
        pos = np.where(ids >= START_SPEECH_TOKEN, 0, np.arange(ids.shape[1])[np.newaxis, :] - 1)
        feed = {"input_ids": ids, "position_ids": pos.astype(np.int64),
                "exaggeration": np.array([s.exaggeration], dtype=np.float32)}
        use_cfg = s.cfg_weight > 0
        batch = 2 if use_cfg else 1

        emb = self._embed.run(None, feed)[0]
        emb = np.concatenate((cond_emb, emb), axis=1)
        if use_cfg:
            # Unconditional branch as in T3: same speaker conditioning, text embeddings zeroed,
            # trailing start-of-speech tokens kept.
            uncond = emb.copy()
            n_text = int((ids[0] < START_SPEECH_TOKEN).sum())
            c = cond_emb.shape[1]
            uncond[:, c : c + n_text, :] = 0.0
            emb = np.concatenate((emb, uncond), axis=0)
        past = {f"past_key_values.{layer}.{kv}": np.zeros([batch, NUM_KV_HEADS, 0, HEAD_DIM], dtype=np.float32)
                for layer in range(NUM_LAYERS) for kv in ("key", "value")}  # fmt: skip
        mask = np.ones((batch, emb.shape[1]), dtype=np.int64)
        generated = [START_SPEECH_TOKEN]
        # Speech tokens run at 25/s and speech at >= ~8 characters/s, so 3 tokens per character
        # (+2 s) is generous. Without this cap a run that never emits stop (seen with greedy
        # decoding and a cloned voice) babbles until max_new_tokens (32 s of audio).
        limit = min(s.max_new_tokens, 50 + 3 * len(text))
        for i in range(limit):
            logits, *present = self._lm.run(None, dict(inputs_embeds=emb, attention_mask=mask, **past))
            logits = logits[:, -1, :].astype(np.float64)
            step = logits[0] + s.cfg_weight * (logits[0] - logits[1]) if use_cfg else logits[0].copy()
            prev = np.array(sorted(set(generated)))
            vals = step[prev]
            step[prev] = np.where(vals < 0, vals * s.repetition_penalty, vals / s.repetition_penalty)
            nxt = self._pick(step, s, rng)
            generated.append(nxt)
            if nxt == STOP_SPEECH_TOKEN:
                break
            feed["input_ids"] = np.array([[nxt]], dtype=np.int64)
            feed["position_ids"] = np.full((1, 1), i + 1, dtype=np.int64)
            emb = self._embed.run(None, feed)[0]
            if use_cfg:
                emb = np.concatenate((emb, emb), axis=0)
            mask = np.concatenate([mask, np.ones((batch, 1), dtype=np.int64)], axis=1)
            for j, key in enumerate(past):
                past[key] = present[j]
        speech = [t for t in generated[1:] if t < START_SPEECH_TOKEN]
        tokens = np.concatenate([prompt_token, np.array([speech], dtype=np.int64)], axis=1)
        wav = self._decoder.run(None, {"speech_tokens": tokens, "speaker_embeddings": spk_emb,
                                       "speaker_features": spk_feat})[0]  # fmt: skip
        wav = np.squeeze(wav, axis=0)
        log.info("chatterbox-onnx: %d speech tokens in %.1fs", len(speech), time.perf_counter() - t0)
        return wav

    def _pick(self, logits, s: Settings, rng) -> int:  # type: ignore[no-untyped-def]
        np = self._np
        if s.temperature <= 0:
            return int(np.argmax(logits))
        logits = logits / s.temperature
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        if s.min_p > 0:
            probs = np.where(probs < s.min_p * probs.max(), 0.0, probs)
        if s.top_p < 1.0:
            order = np.argsort(-probs)
            cum = np.cumsum(probs[order])
            probs[order[cum - probs[order] > s.top_p]] = 0.0
        probs /= probs.sum()
        return int(rng.choice(len(probs), p=probs))
