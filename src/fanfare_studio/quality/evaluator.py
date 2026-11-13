"""Model-based clip evaluation (faster-whisper + Silero VAD, UTMOSv2, SpeechBrain ECAPA).

Install with ``uv sync --extra onnx --extra eval --extra cpu`` (or rocm/cuda). Every model is
loaded lazily and only once; a metric whose model is unavailable is reported as None.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from .metrics import ClipScore, cer, prefix_check, wer, word_diff

log = logging.getLogger("fanfare_studio")
SR = 16000


class Evaluator:
    def __init__(self, whisper_model: str = "small", device: str = "auto", judge=None) -> None:  # type: ignore[no-untyped-def]
        self.whisper_model = whisper_model
        self.device = device
        self.judge = judge
        self._whisper = self._utmos = self._ecapa = None
        self._ref_emb: dict[str, object] = {}

    # --- lazy models -----------------------------------------------------------
    def _torch_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def whisper(self):  # type: ignore[no-untyped-def]
        if self._whisper is None:
            from faster_whisper import WhisperModel

            self._whisper = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
        return self._whisper

    def utmos(self):  # type: ignore[no-untyped-def]
        if self._utmos is None:
            import utmosv2

            self._utmos = utmosv2.create_model(pretrained=True, device=self._torch_device())
        return self._utmos

    def ecapa(self):  # type: ignore[no-untyped-def]
        if self._ecapa is None:
            from platformdirs import user_cache_dir
            from speechbrain.inference.speaker import EncoderClassifier

            self._ecapa = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=f"{user_cache_dir('fanfare-studio')}/spkrec-ecapa-voxceleb",
                run_opts={"device": self._torch_device()},
            )
        return self._ecapa

    # --- metrics -----------------------------------------------------------------
    @staticmethod
    def load(path: str):  # type: ignore[no-untyped-def]
        import librosa

        audio, _ = librosa.load(path, sr=SR, mono=True)
        return audio

    def speech_segments(self, audio) -> list[tuple[float, float]]:  # type: ignore[no-untyped-def]
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        segs = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=150, speech_pad_ms=0))
        return [(s["start"] / SR, s["end"] / SR) for s in segs]

    def transcribe(self, audio, lang: str):  # type: ignore[no-untyped-def]
        m = self.whisper()
        segments, _ = m.transcribe(audio, language=lang, word_timestamps=True, beam_size=5,
                                   condition_on_previous_text=False, vad_filter=False)
        words = [w for s in segments for w in (s.words or [])]
        text = "".join(w.word for w in words).strip()
        return text, words

    def language_prob(self, audio, lang: str) -> float:  # type: ignore[no-untyped-def]
        _, _, probs = self.whisper().detect_language(audio)
        return float(dict(probs).get(lang, 0.0))

    def mos(self, audio) -> float:  # type: ignore[no-untyped-def]
        return float(self.utmos().predict(data=audio, sr=SR, verbose=False))

    def embedding(self, audio):  # type: ignore[no-untyped-def]
        import torch

        with torch.no_grad():
            e = self.ecapa().encode_batch(torch.from_numpy(audio).unsqueeze(0))
        return torch.nn.functional.normalize(e.reshape(-1), dim=0).cpu()

    def speaker_similarity(self, audio, reference: str) -> float:  # type: ignore[no-untyped-def]
        if reference not in self._ref_emb:
            self._ref_emb[reference] = self.embedding(self.load(reference))
        return float((self.embedding(audio) * self._ref_emb[reference]).sum())

    def evaluate(self, path: str, text: str, lang: str, reference: str | None = None) -> ClipScore:
        audio = self.load(path)
        s = ClipScore(path=path, duration_s=round(len(audio) / SR, 2))
        s.transcript, words = self.transcribe(audio, lang)
        s.wer, s.cer = round(wer(text, s.transcript, lang), 3), round(cer(text, s.transcript, lang), 3)
        s.extra_words, s.missing_words, leading = word_diff(text, s.transcript, lang)
        first_word = words[0].start if words else None
        lead, trail, s.prefix_flag, s.prefix_reason = prefix_check(
            self.speech_segments(audio), first_word, leading, len(audio) / SR
        )
        s.lead_s, s.trail_s = round(lead, 2), round(trail, 2)
        for name, fn in (("lang_prob", lambda: self.language_prob(audio, lang)),
                         ("mos", lambda: self.mos(audio)),
                         ("spk_sim", (lambda: self.speaker_similarity(audio, reference)) if reference else None)):
            if fn is None:
                continue
            try:
                setattr(s, name, round(fn(), 3))
            except Exception as e:  # noqa: BLE001 - a missing optional model must not stop eval
                log.warning("eval: %s unavailable (%s)", name, e)
        if self.judge is not None:
            s.judge = self.judge.rate(path, text, lang)
        return s


def as_dict(s: ClipScore) -> dict:
    return asdict(s)
