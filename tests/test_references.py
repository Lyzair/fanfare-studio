"""Club reference voices: processing, HTTP endpoints (mock backend) and CLI."""

from __future__ import annotations

import io
import json
import logging
import threading
import uuid
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("soundfile")

from fanfare_studio import references as refs  # noqa: E402
from fanfare_studio.__main__ import main  # noqa: E402
from fanfare_studio.backends.mock import MockBackend  # noqa: E402
from fanfare_studio.server import VoiceServer  # noqa: E402

from .test_protocol import request, synth  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPEECH = (REPO / "voices" / "de-thorsten.wav").read_bytes()  # 6.6 s CC0 speech


def short_clip(seconds: float = 2.0) -> bytes:
    with wave.open(io.BytesIO(SPEECH)) as w:
        params, frames = w.getparams(), w.readframes(int(w.getframerate() * seconds))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return buf.getvalue()


def multipart(fields: dict[str, str | bytes]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        if isinstance(v, bytes):
            out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"rec.wav\"\r\n"
                    "Content-Type: audio/wav\r\n\r\n").encode() + v + b"\r\n"  # fmt: skip
        else:
            out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    return out + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def post_ref(server, **fields):
    body, ctype = multipart(fields)
    return request(server, "POST", "/references", body, {"Content-Type": ctype})


@pytest.fixture
def mock_server():
    srv = VoiceServer(MockBackend(), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


# --- processing -------------------------------------------------------------------
def test_process_trims_normalises_and_resamples():
    silence = np.zeros(24000, dtype=np.float32)
    import soundfile as sf

    speech, sr = sf.read(io.BytesIO(SPEECH), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, np.concatenate([silence, speech * 0.05, silence]), sr, format="WAV")
    p = refs.process(buf.getvalue())
    with wave.open(io.BytesIO(p.wav)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, 24000, 2)
        dur = w.getnframes() / 24000
    assert dur < len(speech) / sr + 0.5  # 2 s of silence trimmed
    assert p.speech_seconds >= 5
    assert p.decoder in ("ffmpeg", "soundfile")


def test_process_rejects_short_and_warns_quiet():
    with pytest.raises(refs.ReferenceError, match="speech"):
        refs.process(short_clip(2.0))
    import soundfile as sf

    speech, sr = sf.read(io.BytesIO(SPEECH), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, speech * 0.003, sr, format="WAV")
    assert any("quiet" in w for w in refs.process(buf.getvalue()).warnings)
    buf = io.BytesIO()
    sf.write(buf, np.clip(speech * 20, -1, 1), sr, format="WAV")
    assert any("clipped" in w for w in refs.process(buf.getvalue()).warnings)


# --- HTTP --------------------------------------------------------------------------
def test_consent_missing_is_400(mock_server):
    for consent in ("false", ""):
        resp, body = post_ref(mock_server, audio=SPEECH, name="Coach", lang="de", consent=consent)
        assert resp.status == 400 and b"consent" in body
        assert resp.getheader("Access-Control-Allow-Origin") == "*"
    assert not list(refs.references_dir().glob("club-*")) if refs.references_dir().exists() else True


def test_bad_fields_are_400(mock_server):
    assert post_ref(mock_server, audio=short_clip(), name="Coach", lang="de", consent="true")[0].status == 400
    assert post_ref(mock_server, audio=SPEECH, name="x" * 41, lang="de", consent="true")[0].status == 400
    assert post_ref(mock_server, audio=SPEECH, name="Coach", lang="uk", consent="true")[0].status == 400
    assert post_ref(mock_server, name="Coach", lang="de", consent="true")[0].status == 400
    assert post_ref(mock_server, audio=b"not audio at all", name="C", lang="de", consent="true")[0].status == 400


def test_oversize_is_413(mock_server):
    resp, _ = request(mock_server, "POST", "/references", b"--x--",
                      {"Content-Type": "multipart/form-data; boundary=x", "Content-Length": str(11 * 1024 * 1024)})
    assert resp.status == 413
    assert resp.getheader("Access-Control-Allow-Origin") == "*"


def test_add_synthesize_delete_roundtrip(mock_server, caplog):
    with caplog.at_level(logging.DEBUG):
        resp, body = post_ref(mock_server, audio=SPEECH, name="Trainer Zyxwvut", lang="de", consent="true")
    assert resp.status == 201 and resp.getheader("Content-Type") == "application/json"
    assert resp.getheader("Access-Control-Allow-Origin") == "*"
    data = json.loads(body)
    assert set(data) == {"id", "lang", "label", "warnings"} and data["id"].startswith("club-")
    assert data["lang"] == "de" and data["label"] == "Trainer Zyxwvut" and data["warnings"] == []
    assert "Zyxwvut" not in caplog.text  # names are never logged
    rid = data["id"]
    folder = refs.references_dir() / rid
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["consent"] is True and meta["consent_text"] and (folder / "reference.wav").is_file()

    voices = json.loads(request(mock_server, "GET", "/voices")[1])
    assert {"id": rid, "lang": "de", "label": "Trainer Zyxwvut"} in voices
    assert synth(mock_server, "Hallo", voice=rid)[0].status == 200
    voice_id, settings = mock_server.backend.calls[-1]
    assert voice_id == rid and settings.reference == str(folder / "reference.wav")
    assert settings.cfg_weight == 0.5 and settings.exaggeration == 0.5

    resp, body = request(mock_server, "DELETE", f"/references/{rid}")
    assert resp.status == 204 and body == b""
    assert resp.getheader("Access-Control-Allow-Origin") == "*"
    assert not folder.exists()
    assert rid not in {v["id"] for v in json.loads(request(mock_server, "GET", "/voices")[1])}
    assert synth(mock_server, "Hallo", voice=rid)[0].status == 404
    assert request(mock_server, "DELETE", f"/references/{rid}")[0].status == 404
    assert request(mock_server, "DELETE", "/references/../../etc")[0].status == 404


def test_preflight_allows_delete(mock_server):
    resp, _ = request(mock_server, "OPTIONS", "/references/club-1", headers={"Access-Control-Request-Method": "DELETE"})
    assert resp.status == 204 and "DELETE" in resp.getheader("Access-Control-Allow-Methods")
    assert resp.getheader("Access-Control-Allow-Private-Network") == "true"


def test_club_voices_load_at_startup():
    ref, _ = refs.add("Stadionsprecher", "de", SPEECH, consent=True)
    srv = VoiceServer(MockBackend(), port=0)
    try:
        assert ref.id in {v.id for v in srv.backend.voices()}
    finally:
        srv.server_close()


# --- CLI ---------------------------------------------------------------------------
def test_cli_add_list_delete(tmp_path, capsys):
    rec = tmp_path / "rec.wav"
    rec.write_bytes(SPEECH)
    assert main(["voice", "add", "--name", "Coach", "--lang", "de", "--file", str(rec)]) == 2
    assert "consent" in capsys.readouterr().out
    assert not refs.list_all()
    assert main(["voice", "add", "--name", "Coach", "--lang", "de", "--file", str(rec), "--consent"]) == 0
    out = capsys.readouterr().out
    rid = out.split()[1]
    assert rid.startswith("club-")
    assert main(["voice", "list"]) == 0
    assert rid in capsys.readouterr().out
    assert main(["voice", "delete", rid]) == 0
    assert not refs.list_all()
    assert main(["voice", "delete", rid]) == 1


def test_soundfile_fallback_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(refs.shutil, "which", lambda name: None)
    p = refs.process(SPEECH)
    assert p.decoder == "soundfile" and p.speech_seconds >= 5
    with pytest.raises(refs.ReferenceError, match="without ffmpeg"):
        refs.process(b"\x1aE\xdf\xa3 webm-ish bytes")
