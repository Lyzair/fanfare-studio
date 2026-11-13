"""Evaluation metrics, tuning loop (mock backend + mocked scorer) and preference learning."""

from __future__ import annotations

import csv
import json
import random
import wave
from pathlib import Path

import pytest

from fanfare_studio.backends.mock import MockBackend
from fanfare_studio.quality import prefs
from fanfare_studio.quality.judge import from_config, is_local_url, LocalJudge
from fanfare_studio.quality.metrics import cer, normalize_words, prefix_check, wer, word_diff
from fanfare_studio.quality.tune import prefer, run_tune

REPO = Path(__file__).resolve().parents[1]
TEXT = "Und jetzt, mit der Nummer 7: Lea! Viel Spaß und gib alles!"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FANFARE_STUDIO_DATA", str(tmp_path / "data"))


# --- metrics --------------------------------------------------------------------
def test_normalisation_wer_cer():
    assert normalize_words("Lea! Viel Spaß, und: GIB alles.") == ["lea", "viel", "spaß", "und", "gib", "alles"]
    assert wer(TEXT, "und jetzt mit der nummer 7 lea viel spaß und gib alles") == 0
    assert cer(TEXT, "Und jetzt mit der Nummer 7, Lea. Viel Spaß und gib alles!") == 0
    assert wer("a b c d", "a x c d") == 0.25
    assert cer("wearing number 7", "Wearing number seven.", "en") == 0
    assert cer("mit der Nummer 7", "mit der Nummer sieben", "de") == 0
    assert cer("mit der Nummer 7", "mit der Nummer sieben") > 0


def test_word_diff_finds_garbage_prefix():
    extra, missing, leading = word_diff(TEXT, "Mein Name Die und jetzt mit der Nummer 7 Lea viel Spaß und gib alles")
    assert leading == ["mein", "name", "die"] and extra == leading and missing == []
    extra, missing, leading = word_diff(TEXT, "Und jetzt mit der Nummer 7 Lea viel Spaß. And, Jonik.")
    assert leading == [] and extra[-2:] == ["and", "jonik"] and "alles" in missing


def test_prefix_check():
    assert prefix_check([(0.05, 3.0)], 0.1, [], 3.4)[2] is False
    lead, trail, flag, why = prefix_check([(1.0, 3.0)], 1.05, [], 3.2)
    assert flag and lead == 1.0 and "leading non-speech" in why
    assert prefix_check([(0.05, 0.6), (1.2, 3.0)], 1.25, [], 3.1)[2] is True  # babble before first word
    assert prefix_check([(0.05, 3.0)], 0.1, ["mein"], 3.1)[2] is True
    assert prefix_check([], None, [], 2.0)[2] is True


def test_real_vad_detects_silence_prefix(tmp_path):
    """TTS-free fixture: the CC0 Thorsten clip with 1 s of silence prepended; transcript mocked."""
    pytest.importorskip("faster_whisper")
    librosa = pytest.importorskip("librosa")
    import numpy as np

    from fanfare_studio.quality.evaluator import SR, Evaluator

    speech, _ = librosa.load(REPO / "voices" / "de-thorsten.wav", sr=SR)
    ev = Evaluator()
    segs = ev.speech_segments(np.concatenate([np.zeros(SR, dtype=np.float32), speech]))
    lead, _, flag, why = prefix_check(segs, None, [], 1 + len(speech) / SR)
    assert 0.8 < lead < 1.3 and flag and "leading" in why
    segs = ev.speech_segments(speech)
    assert segs[0][0] < 0.4


def test_evaluate_with_mocked_models(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    from fanfare_studio.quality.evaluator import SR, Evaluator
    from fanfare_studio.wav import encode_numpy

    path = tmp_path / "c.wav"
    path.write_bytes(encode_numpy(np.zeros(SR * 2, dtype=np.float32), SR))
    ev = Evaluator()

    class W:
        def __init__(self, word, start):
            self.word, self.start = word, start

    monkeypatch.setattr(ev, "transcribe", lambda a, l: ("Mein Und jetzt", [W(" Mein", 0.1), W(" Und", 0.6)]))
    monkeypatch.setattr(ev, "speech_segments", lambda a: [(0.1, 1.9)])
    monkeypatch.setattr(ev, "language_prob", lambda a, l: 0.9)
    monkeypatch.setattr(ev, "mos", lambda a: 3.5)
    s = ev.evaluate(str(path), "Und jetzt", "de")
    assert s.prefix_flag and s.extra_words == ["mein"] and s.lang_prob == 0.9 and s.mos == 3.5
    assert s.spk_sim is None and s.duration_s == 2.0
    assert prefs.rejected(vars(s)).startswith("CER")  # 'mein ' adds > 15 % characters


# --- combined score + filters -------------------------------------------------------
def test_combined_and_rejected():
    good = {"cer": 0.0, "lang_prob": 1.0, "mos": 4.0, "spk_sim": 0.8, "lead_s": 0.1, "trail_s": 0.3}
    worse = {**good, "mos": 2.5}
    w = prefs.DEFAULT_WEIGHTS
    assert prefs.combined(good, w) > prefs.combined(worse, w)
    assert prefs.rejected(good) == ""
    assert "CER" in prefs.rejected({**good, "cer": 0.2})
    assert "prefix" in prefs.rejected({**good, "prefix_flag": True, "prefix_reason": "x"})


# --- preference learning -------------------------------------------------------------
def _rand_features(rng):
    return {k: rng.random() for k in prefs.FEATURES}


def test_fit_weights_defaults_with_few_pairs():
    rng = random.Random(0)
    pairs = [(_rand_features(rng), _rand_features(rng)) for _ in range(prefs.MIN_PICKS - 1)]
    assert prefs.fit_weights(pairs) == prefs.DEFAULT_WEIGHTS


def test_fit_weights_learns_hidden_preference():
    """Synthetic listener who only cares about MOS: the MOS weight must grow, others shrink."""
    rng = random.Random(1)
    pairs = []
    for _ in range(200):
        a, b = _rand_features(rng), _rand_features(rng)
        pairs.append((a, b) if a["mos"] > b["mos"] else (b, a))
    w = prefs.fit_weights(pairs, l2=0.1)
    assert w["mos"] > prefs.DEFAULT_WEIGHTS["mos"] + 0.2
    assert w["mos"] == max(w.values())
    assert abs(sum(w.values()) - sum(prefs.DEFAULT_WEIGHTS.values())) < 1e-3


# --- tune loop -------------------------------------------------------------------
def fake_scorer(calls):
    """Score depends on settings recorded by the mock backend: best near cfg 0.5, ex 0.8, t 0.6."""

    def score(path, text, lang, ref):
        s = calls[-1][1]
        q = 1 - abs(s.cfg_weight - 0.5) - abs(s.exaggeration - 0.8) - abs(s.temperature - 0.6)
        return {"cer": 0.3 if s.temperature > 0.75 else 0.0,  # high temperature "mumbles" → rejected
                "lang_prob": 1.0, "mos": 1 + 4 * max(0.0, q), "spk_sim": 0.5,
                "lead_s": 0.1, "trail_s": 0.2, "prefix_flag": False, "transcript": text}  # fmt: skip

    return score


def test_tune_with_mock_backend(tmp_path):
    backend = MockBackend()
    run = run_tune(backend, fake_scorer(backend.calls), lang="de", sentences=["Eins.", "Zwei drei."],
                   budget=20, out_root=tmp_path / "tune-out", voice_id="mock-de", references=[None],
                   weights=prefs.DEFAULT_WEIGHTS)  # fmt: skip
    assert len(backend.calls) <= 20 and len(backend.calls) % 2 == 0
    clips = sorted((run / "clips").glob("*.wav"))
    assert len(clips) == len(backend.calls)
    rows = list(csv.DictReader((run / "scores.csv").open(encoding="utf-8")))
    assert len(rows) == len(clips) and {"cer", "mos", "combined", "rejected"} <= set(rows[0])
    board = (run / "leaderboard.md").read_text(encoding="utf-8")
    assert "| 1 | " in board and "CER 0.30" in board  # rejected configs listed with the reason
    listen = sorted(p.name for p in (run / "listen").iterdir())
    assert listen == ["A.wav", "B.wav", "C.wav"]
    mapping = json.loads((run / "mapping.json").read_text())
    best = mapping["letters"]["A"]["config"]
    assert best["temperature"] <= 0.75  # rejected configs never reach the listening set
    assert all(set(v["features"]) == set(prefs.FEATURES) for v in mapping["letters"].values())
    with wave.open(str(run / "listen" / "A.wav")) as w:  # both sentences + a gap
        assert w.getnframes() > 2 * 16000 * 0.2

    # preference recording + refit
    assert prefer(run, pick="B") == 2
    assert prefer(run, rank=["C", "A", "B"]) == 3
    assert len(prefs.load_pairs()) == 5
    assert prefs.preferences_path().is_file()
    with pytest.raises(ValueError):
        prefer(run, pick="D")


def test_tune_budget_too_small(tmp_path):
    with pytest.raises(ValueError):
        run_tune(MockBackend(), lambda *a: {}, lang="de", sentences=["a", "b", "c"], budget=2,
                 out_root=tmp_path, voice_id="mock-de", references=[None], weights=prefs.DEFAULT_WEIGHTS)


# --- judge ----------------------------------------------------------------------
def test_judge_is_optional_and_local_only():
    assert from_config({}) is None
    assert from_config({"judge": {"enabled": False, "url": "https://api.example.com"}}) is None
    assert is_local_url("http://127.0.0.1:8000/v1") and is_local_url("http://localhost:1")
    for bad in ("https://api.openai.com/v1", "http://localhost.evil.com/v1", "http://u:p@localhost/v1",
                "file:///etc/passwd", "http://192.168.1.2:8000"):
        assert not is_local_url(bad)
        with pytest.raises(ValueError):
            LocalJudge(bad, "m")
    assert LocalJudge("http://127.0.0.1:8000/v1/", "m").url == "http://127.0.0.1:8000/v1"
    with pytest.raises(ValueError):
        LocalJudge("http://127.0.0.1:8000", "m")  # must end with /v1
