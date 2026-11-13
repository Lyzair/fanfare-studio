from __future__ import annotations

from pathlib import Path

import pytest

from fanfare_studio.backends.chatterbox_onnx_backend import normalize_text
from fanfare_studio.voice_config import load_voices

REPO = Path(__file__).resolve().parents[1]


def test_builtin_voices_without_config():
    entries = load_voices(None)
    assert {e.voice.id for e in entries} == {f"{l}-high" for l in "de en fr es it nl pl tr ar".split()}
    assert all(e.settings.reference is None for e in entries)


def test_repo_voices_toml_gives_german_a_native_reference():
    entries = {e.voice.id: e for e in load_voices(str(REPO / "voices.toml"))}
    de = entries["de-high"].settings
    assert de.reference and Path(de.reference).name == "de-thorsten.wav" and Path(de.reference).is_file()
    assert entries["en-high"].settings.reference is None
    assert de.seed is not None


def test_config_overrides_and_validation(tmp_path):
    (tmp_path / "r.wav").write_bytes(b"RIFF")
    cfg = tmp_path / "v.toml"
    cfg.write_text('[defaults]\ntemperature = 0.3\n[voices.de-club]\nlang = "de"\nlabel = "Club"\n'
                   'reference = "r.wav"\nexaggeration = 0.9\n')
    entries = {e.voice.id: e for e in load_voices(str(cfg))}
    club = entries["de-club"]
    assert (club.voice.lang, club.voice.label) == ("de", "Club")
    assert club.settings.reference == str(tmp_path / "r.wav")
    assert (club.settings.temperature, club.settings.exaggeration) == (0.3, 0.9)
    assert entries["en-high"].settings.temperature == 0.3

    cfg.write_text('[voices.de-high]\nloudness = 3\n')
    with pytest.raises(ValueError):
        load_voices(str(cfg))
    cfg.write_text('[voices.x]\nlang = "uk"\n')
    with pytest.raises(ValueError):
        load_voices(str(cfg))
    cfg.write_text('[voices.de-high]\nreference = "missing.wav"\n')
    with pytest.raises(FileNotFoundError):
        load_voices(str(cfg))


def test_normalize_text_matches_torch_preprocessing():
    # lower-case + NFKD + punctuation clean-up like chatterbox punc_norm/MTLTokenizer
    assert normalize_text("Und jetzt, mit der Nummer 7: Lea!") == "und jetzt, mit der nummer 7, lea!"
    assert normalize_text("hallo  welt") == "hallo welt."
    assert normalize_text("Spaß!") == "spaß!"
    assert normalize_text("Ä") == "ä"
