"""Protocol tests against the mock backend (no GPU, no model download)."""

from __future__ import annotations

import http.client
import io
import json
import logging
import wave

import pytest

from fanfare_studio import LANGS
from fanfare_studio.backends.base import BackendError
from fanfare_studio.server import HOST, MAX_TEXT_CHARS, VoiceServer


ORIGIN = "http://localhost:5173"  # allowed by default (local dev); browsers always send Origin


def request(server, method, path, body=None, headers=None):
    """Like a browser page on ORIGIN; pass headers={"Origin": None} for a non-browser client."""
    conn = http.client.HTTPConnection(*server.server_address[:2], timeout=10)
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    h = {"Content-Type": "application/json"} if data is not None else {}
    h["Origin"] = ORIGIN
    h.update(headers or {})
    h = {k: v for k, v in h.items() if v is not None}
    conn.request(method, path, body=data, headers=h)
    resp = conn.getresponse()
    payload = resp.read()
    conn.close()
    return resp, payload


def synth(server, text, voice="mock-de"):
    return request(server, "POST", "/synthesize", {"text": text, "voice": voice})


# --- binding -----------------------------------------------------------------
def test_binds_loopback_only(server):
    assert HOST == "127.0.0.1"
    assert server.server_address[0] == "127.0.0.1"


def test_default_port_is_5123():
    from fanfare_studio.__main__ import main  # noqa: F401
    from fanfare_studio.server import DEFAULT_PORT

    assert DEFAULT_PORT == 5123


# --- GET /voices -------------------------------------------------------------
def test_voices(server):
    resp, body = request(server, "GET", "/voices")
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "application/json"
    assert resp.getheader("Access-Control-Allow-Origin") == ORIGIN
    voices = json.loads(body)
    assert isinstance(voices, list) and voices
    for v in voices:
        assert set(v) == {"id", "lang", "label"}
        assert all(isinstance(v[k], str) and v[k] for k in v)
        assert v["lang"] in LANGS
    assert {v["lang"] for v in voices} == set(LANGS)


# --- POST /synthesize ----------------------------------------------------------
def test_synthesize_returns_wav(server):
    resp, body = synth(server, "Und jetzt, mit der Nummer 7: Lea!")
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "audio/wav"
    assert resp.getheader("Access-Control-Allow-Origin") == ORIGIN
    assert int(resp.getheader("Content-Length")) == len(body)
    with wave.open(io.BytesIO(body)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() > 0


def test_text_of_exactly_max_length_is_accepted(server):
    resp, _ = synth(server, "a" * MAX_TEXT_CHARS)
    assert resp.status == 200


@pytest.mark.parametrize(
    "body",
    [
        {"text": "", "voice": "mock-de"},
        {"text": "   ", "voice": "mock-de"},
        {"voice": "mock-de"},
        {"text": 7, "voice": "mock-de"},
        {"text": "a" * (MAX_TEXT_CHARS + 1), "voice": "mock-de"},
        {"text": "hallo"},
        ["not", "an", "object"],
        b"{not json",
    ],
    ids=["empty", "blank", "missing-text", "non-string", "too-long", "missing-voice", "array", "bad-json"],
)
def test_bad_requests_are_400_text_plain(server, body):
    resp, payload = request(server, "POST", "/synthesize", body)
    assert resp.status == 400
    assert resp.getheader("Content-Type").startswith("text/plain")
    assert resp.getheader("Access-Control-Allow-Origin") == ORIGIN
    assert payload.strip()


def test_unknown_voice_is_404_text_plain(server):
    resp, payload = synth(server, "hallo", voice="nope")
    assert resp.status == 404
    assert resp.getheader("Content-Type").startswith("text/plain")
    assert resp.getheader("Access-Control-Allow-Origin") == ORIGIN
    assert b"nope" in payload


def test_model_failure_is_500_text_plain(server, monkeypatch):
    def boom(text, voice_id):
        raise BackendError("GPU on fire")

    monkeypatch.setattr(server.backend, "synthesize", boom)
    resp, payload = synth(server, "hallo")
    assert resp.status == 500
    assert resp.getheader("Content-Type").startswith("text/plain")
    assert resp.getheader("Access-Control-Allow-Origin") == ORIGIN


def test_unexpected_exception_is_500(server, monkeypatch):
    monkeypatch.setattr(server.backend, "synthesize", lambda t, v: 1 / 0)
    resp, _ = synth(server, "hallo")
    assert resp.status == 500


def test_oversized_body_is_rejected(server):
    # Declared size is checked before reading the body.
    resp, _ = request(server, "POST", "/synthesize", b"{}", {"Content-Length": str(128 * 1024)})
    assert resp.status == 413
    assert resp.getheader("Access-Control-Allow-Origin") == ORIGIN


def test_unknown_route_is_404_with_cors(server):
    for method, path in [("GET", "/nope"), ("POST", "/voices")]:
        resp, _ = request(server, method, path, {} if method == "POST" else None)
        assert resp.status == 404
        assert resp.getheader("Access-Control-Allow-Origin") == ORIGIN


def test_unsupported_method_has_cors(server):
    resp, _ = request(server, "PUT", "/voices")
    assert resp.status == 501
    assert resp.getheader("Access-Control-Allow-Origin") == ORIGIN


# --- CORS preflight + Private Network Access ----------------------------------
@pytest.mark.parametrize("path", ["/voices", "/synthesize"])
def test_preflight(server, path):
    resp, body = request(
        server,
        "OPTIONS",
        path,
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert resp.status == 204
    assert body == b""
    assert resp.getheader("Access-Control-Allow-Origin") == ORIGIN
    assert resp.getheader("Access-Control-Allow-Methods") == "GET, POST, DELETE"
    assert resp.getheader("Access-Control-Allow-Headers") == "Content-Type, X-Fanfare"
    assert resp.getheader("Access-Control-Allow-Private-Network") == "true"


# --- privacy: request text never logged ----------------------------------------
def test_request_text_is_never_logged(server, caplog, capsys):
    secret = "Geheimer Spielername Zyxwvut"
    with caplog.at_level(logging.DEBUG):
        synth(server, secret)
        synth(server, secret, voice="unknown-voice-xyz")
        synth(server, secret + "x" * MAX_TEXT_CHARS)
        request(server, "GET", "/voices?name=" + secret.replace(" ", "+"))
    out = capsys.readouterr()
    everything = caplog.text + out.out + out.err
    assert "Zyxwvut" not in everything
    # the access log has method, path, status and duration
    assert any(
        r.getMessage().startswith("POST /synthesize 200 ") and r.getMessage().endswith("ms")
        for r in caplog.records
    )
    assert any(r.getMessage().startswith("GET /voices 200 ") for r in caplog.records)


def test_synthesis_is_serialized(monkeypatch):
    """The server lock makes concurrent requests run one at a time."""
    import threading
    import time

    from fanfare_studio.backends.mock import MockBackend

    backend = MockBackend()
    active, peak = 0, 0
    guard = threading.Lock()
    real = backend.synthesize

    def slow(text, voice_id):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return real(text, voice_id)

    backend.synthesize = slow  # type: ignore[method-assign]
    srv = VoiceServer(backend, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        threads = [threading.Thread(target=synth, args=(srv, "hi")) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        srv.shutdown()
        srv.server_close()
    assert peak == 1
