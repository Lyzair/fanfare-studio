"""Combined score, hard filters and preference learning (pairwise logistic / Bradley-Terry)."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

# Features are oriented so that larger = better.
FEATURES = ("asr", "lang", "mos", "spk", "edges")
DEFAULT_WEIGHTS = {"asr": 0.35, "lang": 0.15, "mos": 0.35, "spk": 0.10, "edges": 0.05}
MIN_PICKS = 5


def features(score: dict) -> dict[str, float]:
    """Map raw metrics (ClipScore fields or their per-config means) to [~0, 1] features."""
    cer = score.get("cer")
    lead, trail = score.get("lead_s") or 0.0, score.get("trail_s") or 0.0
    mos = score.get("mos")
    return {
        "asr": 1.0 - min(1.0, cer) if cer is not None else 0.0,
        "lang": score.get("lang_prob") or 0.0,
        "mos": (mos - 1.0) / 4.0 if mos is not None else 0.0,
        "spk": max(0.0, score.get("spk_sim") or 0.0),
        "edges": 1.0 - min(1.0, max(0.0, lead - 0.2) + max(0.0, trail - 0.4)),
    }


def combined(score: dict, weights: dict[str, float]) -> float:
    f = features(score)
    return sum(weights.get(k, 0.0) * f[k] for k in FEATURES)


def rejected(score: dict, max_cer: float = 0.15) -> str:
    """Reason for a hard rejection, or ''."""
    if score.get("cer") is None:
        return "no transcript"
    if score["cer"] > max_cer:
        return f"CER {score['cer']:.2f} > {max_cer}"
    if score.get("prefix_flag"):
        return f"prefix: {score.get('prefix_reason', '')}"
    return ""


def data_dir() -> Path:
    env = os.environ.get("FANFARE_STUDIO_DATA")
    if env:
        return Path(env)
    from platformdirs import user_data_dir

    return Path(user_data_dir("fanfare-studio", appauthor=False))


def preferences_path() -> Path:
    return data_dir() / "preferences.jsonl"


def record_preferences(pairs: list[tuple[dict, dict]], meta: dict) -> int:
    """Append (winner_features, loser_features) pairs; returns number written."""
    path = preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for win, lose in pairs:
            fh.write(json.dumps({"ts": time.time(), **meta, "winner": win, "loser": lose}) + "\n")
    return len(pairs)


def load_pairs(path: Path | None = None) -> list[tuple[dict, dict]]:
    path = path or preferences_path()
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out.append((rec["winner"], rec["loser"]))
    return out


def fit_weights(
    pairs: list[tuple[dict, dict]],
    prior: dict[str, float] | None = None,
    l2: float = 1.0,
    steps: int = 2000,
    lr: float = 0.5,
) -> dict[str, float]:
    """Bradley-Terry / pairwise logistic regression on feature differences.

    P(winner beats loser) = sigmoid(w · (f_w - f_l)), L2-regularised towards the prior
    weights (so few picks only nudge the defaults). Returns weights with the prior's L1 norm,
    clipped at 0 (a metric can't count against quality). Fewer than MIN_PICKS pairs → prior.
    """
    prior = dict(prior or DEFAULT_WEIGHTS)
    if len(pairs) < MIN_PICKS:
        return prior
    diffs = [[w[k] - l[k] for k in FEATURES] for w, l in pairs]
    scale = sum(abs(v) for v in prior.values()) or 1.0
    w = [prior[k] / scale * len(FEATURES) for k in FEATURES]
    p0 = list(w)
    n = len(diffs)
    for _ in range(steps):
        grad = [l2 * (wi - pi) / n for wi, pi in zip(w, p0)]
        for d in diffs:
            z = sum(wi * di for wi, di in zip(w, d))
            g = -1.0 / (1.0 + math.exp(z))  # d/dz of -log sigmoid(z)
            for i, di in enumerate(d):
                grad[i] += g * di / n
        w = [wi - lr * gi for wi, gi in zip(w, grad)]
    w = [max(0.0, wi) for wi in w]
    total = sum(w) or 1.0
    return {k: round(wi / total * scale, 4) for k, wi in zip(FEATURES, w)}
