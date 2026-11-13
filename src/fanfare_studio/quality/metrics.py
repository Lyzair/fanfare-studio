"""Pure metric helpers (no model dependencies)."""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

LEAD_TRAIL_LIMIT_S = 0.4


def normalize_words(text: str) -> list[str]:
    """Lower-case, strip punctuation, NFC; digits stay digits."""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s']|_", " ", text)
    return text.split()


def _edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_words(ref), normalize_words(hyp)
    return _edit_distance(r, h) / max(1, len(r))


def cer(ref: str, hyp: str) -> float:
    r, h = " ".join(normalize_words(ref)), " ".join(normalize_words(hyp))
    return _edit_distance(list(r), list(h)) / max(1, len(r))


def word_diff(ref: str, hyp: str) -> tuple[list[str], list[str], list[str]]:
    """(extra words in hyp, missing words from ref, extra words BEFORE the first matched word)."""
    r, h = normalize_words(ref), normalize_words(hyp)
    extra, missing, leading = [], [], []
    seen_match = False
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if op == "equal":
            seen_match = True
            continue
        if op in ("replace", "delete"):
            missing += r[i1:i2]
        if op in ("replace", "insert"):
            extra += h[j1:j2]
            if not seen_match and op == "insert":
                leading += h[j1:j2]
    return extra, missing, leading


@dataclass
class ClipScore:
    path: str
    transcript: str = ""
    wer: float | None = None
    cer: float | None = None
    extra_words: list[str] = field(default_factory=list)
    missing_words: list[str] = field(default_factory=list)
    lead_s: float | None = None  # non-speech before first VAD speech segment
    trail_s: float | None = None
    prefix_flag: bool = False
    prefix_reason: str = ""
    lang_prob: float | None = None  # Whisper P(target language): accent proxy
    mos: float | None = None  # UTMOSv2 predicted MOS (1-5)
    spk_sim: float | None = None  # ECAPA cosine to --reference
    duration_s: float | None = None
    judge: dict | None = None


def prefix_check(
    speech: list[tuple[float, float]],
    first_word_start: float | None,
    leading_extra: list[str],
    duration: float,
) -> tuple[float, float, bool, str]:
    """Leading/trailing non-speech and the "garbage prefix" flag.

    speech: VAD speech segments (start, end) in seconds. first_word_start: start of the first
    ASR word that matches the intended text (None if no match).
    """
    if not speech:
        return duration, duration, True, "no speech detected"
    lead = speech[0][0]
    trail = max(0.0, duration - speech[-1][1])
    reasons = []
    if lead > LEAD_TRAIL_LIMIT_S:
        reasons.append(f"{lead:.2f}s leading non-speech")
    if leading_extra:
        reasons.append(f"extra words before the text: {' '.join(leading_extra)}")
    if first_word_start is not None and speech[0][0] < first_word_start - 0.3:
        # speech-like audio well before the first intended word (e.g. babble ASR ignored)
        if speech[0][1] < first_word_start - 0.1:
            reasons.append(f"speech segment at {speech[0][0]:.2f}s before first word at {first_word_start:.2f}s")
    return lead, trail, bool(reasons), "; ".join(reasons)
