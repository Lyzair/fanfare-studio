"""Club reference voices: "your club's own announcer voice".

Stored per user, outside the repository::

    <user data dir>/fanfare-studio/references/<id>/reference.wav   cleaned clip, 24 kHz mono 16-bit
    <user data dir>/fanfare-studio/references/<id>/meta.json       {id, label, lang, created, consent, consent_text}

IDs are random (``club-<8 hex>``) so that names never appear in paths or access logs.
A voice is personal data: adding one requires explicit consent, everything stays on this
computer, and deleting removes the files.
"""

from __future__ import annotations

import io
import json
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .voice_config import CHATTERBOX_LANGS

SAMPLE_RATE = 24000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_INPUT_SECONDS = 60.0
KEEP_SECONDS = 30.0
MIN_SPEECH_SECONDS = 5.0
MAX_NAME_CHARS = 40
TARGET_RMS_DBFS = -20.0
CONSENT_TEXT = (
    "I confirm that the person whose voice this is has agreed that it is stored on this computer "
    "and used to generate announcements, and that they can ask for it to be deleted at any time."
)


class ReferenceError(ValueError):
    """Invalid input; the message is safe to show to the user (HTTP 400)."""


@dataclass
class Processed:
    wav: bytes
    speech_seconds: float
    warnings: list[str] = field(default_factory=list)
    decoder: str = ""


def references_dir() -> Path:
    from .quality.prefs import data_dir

    return data_dir() / "references"


def _decode(data: bytes):  # type: ignore[no-untyped-def]
    """→ (mono float32 samples at SAMPLE_RATE, decoder name). ffmpeg if present, else soundfile."""
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-t", str(MAX_INPUT_SECONDS + 1), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
            input=data, capture_output=True, timeout=60, check=False,
        )  # fmt: skip
        if proc.returncode != 0 or not proc.stdout:
            raise ReferenceError("could not decode the audio (supported: wav, webm, ogg, mp3, m4a, flac)")
        return np.frombuffer(proc.stdout, dtype="<f4").copy(), "ffmpeg"
    import soundfile as sf

    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:
        raise ReferenceError("could not decode the audio (without ffmpeg only wav/flac/ogg are supported)") from e
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32), "soundfile"


def _speech_mask(audio, frame: int):  # type: ignore[no-untyped-def]
    """Energy-based speech/non-speech per frame (relative to the loudest frames)."""
    import numpy as np

    n = len(audio) // frame
    if n == 0:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    ref = np.percentile(db, 95)
    mask = db > max(ref - 35.0, -80.0)
    # pauses shorter than 300 ms between words count as speech
    gap_frames = int(0.3 * SAMPLE_RATE / frame)
    idx = np.flatnonzero(mask)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap_frames:
            mask[a:b] = True
    return mask


def process(data: bytes) -> Processed:
    """Decode, mono, resample, trim silence, loudness-normalise, cap at 30 s, quality-check."""
    import numpy as np

    from .wav import encode_numpy

    if len(data) > MAX_UPLOAD_BYTES:
        raise ReferenceError("audio larger than 10 MB")
    audio, decoder = _decode(data)
    if len(audio) / SAMPLE_RATE > MAX_INPUT_SECONDS:
        raise ReferenceError("recording longer than 60 seconds")
    warnings: list[str] = []
    if audio.size and np.mean(np.abs(audio) >= 0.999) > 0.001:
        warnings.append("the recording is clipped (too loud) - record a bit further from the microphone")
    frame = SAMPLE_RATE // 50  # 20 ms
    mask = _speech_mask(audio, frame)
    if not mask.any():
        raise ReferenceError("no speech found in the recording")
    first, last = int(np.argmax(mask)), len(mask) - int(np.argmax(mask[::-1]))
    pad = 5  # keep 100 ms around speech
    audio = audio[max(0, (first - pad) * frame) : min(len(audio), (last + pad) * frame)]
    audio = audio[: int(KEEP_SECONDS * SAMPLE_RATE)]
    speech_seconds = float(_speech_mask(audio, frame).sum() * frame / SAMPLE_RATE)
    if speech_seconds < MIN_SPEECH_SECONDS:
        raise ReferenceError(
            f"only {speech_seconds:.1f} s of speech - record at least {MIN_SPEECH_SECONDS:.0f} s (15-20 s is best)"
        )
    rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-12)
    if rms_db < -40:
        warnings.append("the recording is very quiet - move closer to the microphone")
    gain = 10 ** ((TARGET_RMS_DBFS - rms_db) / 20)
    peak = float(np.max(np.abs(audio))) or 1.0
    gain = min(gain, 0.95 / peak)  # no clipping from normalisation
    audio = (audio * gain).astype(np.float32)
    return Processed(encode_numpy(audio, SAMPLE_RATE), round(speech_seconds, 1), warnings, decoder)


@dataclass
class Reference:
    id: str
    label: str
    lang: str
    created: str
    path: Path  # reference.wav

    def meta(self) -> dict:
        return {"id": self.id, "label": self.label, "lang": self.lang, "created": self.created}


def validate(name: str, lang: str, consent: bool) -> str:
    if not consent:
        raise ReferenceError(
            "consent required: only add a voice if its owner agreed that it is stored on this computer "
            "and used for announcements (and can ask for deletion)"
        )
    name = " ".join((name or "").split())
    if not name:
        raise ReferenceError("name is required")
    if len(name) > MAX_NAME_CHARS:
        raise ReferenceError(f"name longer than {MAX_NAME_CHARS} characters")
    if lang not in CHATTERBOX_LANGS:
        raise ReferenceError(f"language must be one of: {', '.join(CHATTERBOX_LANGS)}")
    return name


def add(name: str, lang: str, data: bytes, consent: bool, root: Path | None = None) -> tuple[Reference, list[str]]:
    name = validate(name, lang, consent)
    processed = process(data)
    root = root or references_dir()
    rid = f"club-{secrets.token_hex(4)}"
    folder = root / rid
    folder.mkdir(parents=True)
    (folder / "reference.wav").write_bytes(processed.wav)
    created = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta = {"id": rid, "label": name, "lang": lang, "created": created, "consent": True,
            "consent_text": CONSENT_TEXT, "speech_seconds": processed.speech_seconds,
            "decoder": processed.decoder}  # fmt: skip
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return Reference(rid, name, lang, created, folder / "reference.wav"), processed.warnings


def list_all(root: Path | None = None) -> list[Reference]:
    root = root or references_dir()
    out = []
    for meta_file in sorted(root.glob("club-*/meta.json")):
        try:
            m = json.loads(meta_file.read_text(encoding="utf-8"))
            if m.get("consent") is True and (meta_file.parent / "reference.wav").is_file():
                out.append(Reference(m["id"], m["label"], m["lang"], m["created"], meta_file.parent / "reference.wav"))
        except (OSError, ValueError, KeyError):
            continue
    return out


def delete(rid: str, root: Path | None = None) -> bool:
    root = root or references_dir()
    if not rid.startswith("club-") or "/" in rid or "\\" in rid or ".." in rid:
        return False
    folder = root / rid
    if not folder.is_dir():
        return False
    shutil.rmtree(folder)
    return True
