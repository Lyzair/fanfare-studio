from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fanfare_studio.backends.mock import MockBackend
from fanfare_studio.quality.judge import NOT_ASSESSABLE, TOKEN_ENV, LocalJudge

TOKEN = "sk-unsloth-test"


class FakeUnsloth(ThreadingHTTPServer):
    def __init__(self, accept_audio: bool, token: str = TOKEN) -> None:
        super().__init__(("127.0.0.1", 0), _H)
        self.accept_audio, self.token = accept_audio, token
        self.log: list[tuple[str, str, dict]] = []  # (method, path, body-ish)
        self.auth_seen: set[str | None] = set()


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:  # noqa: ANN002
        pass

    def _reply(self, status: int, obj) -> None:  # noqa: ANN001
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        s = self.server
        s.auth_seen.add(self.headers.get("Authorization"))
        if self.path.startswith("/v1") and self.headers.get("Authorization") != f"Bearer {s.token}":
            return self._reply(401, {"error": "unauthorized"})
        if self.path == "/v1/models":
            s.log.append(("GET", self.path, {}))
            return self._reply(200, {"data": [{"id": "gemma-audio"}]})
        if self.path == "/openapi.json":
            return self._reply(200, {"paths": {"/v1/chat/completions": {}, "/v1/audio/transcriptions": {}}})
        if self.path == "/v1/chat/completions":
            body = json.loads(raw)
            parts = body["messages"][0]["content"]
            has_audio = isinstance(parts, list) and any(p["type"] == "input_audio" for p in parts)
            s.log.append(("POST", self.path, {"audio": has_audio}))
            if has_audio and not s.accept_audio:
                return self._reply(400, {"error": "audio input not supported"})
            if has_audio and "Rate 1-5" not in parts[0]["text"]:
                return self._reply(200, {"choices": [{"message": {"content": "ok"}}]})
            text = parts if isinstance(parts, str) else parts[0]["text"]
            ans = '{"native": 4, "exact": 5, "announcer": 3, "quality": 4, "comment": "good"}' \
                if has_audio else '{"exact": 5, "comment": "match"}'
            return self._reply(200, {"choices": [{"message": {"content": ans}}]})
        if self.path == "/v1/audio/transcriptions":
            s.log.append(("POST", self.path, {"multipart": b"filename=" in raw}))
            return self._reply(200, {"text": "Tor fuer Bayern"})
        self._reply(404, {})

    do_GET = do_POST = _handle  # noqa: N815


@pytest.fixture
def fake(request):  # noqa: ANN001, ANN201
    srv = FakeUnsloth(accept_audio=request.param)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _clip(tmp_path):  # noqa: ANN001, ANN202
    p = tmp_path / "c.wav"
    p.write_bytes(MockBackend().synthesize("Tor", "mock-de"))
    return str(p)


def _judge(srv, model="default"):  # noqa: ANN001, ANN202
    return LocalJudge(f"http://127.0.0.1:{srv.server_address[1]}/v1", model)


@pytest.mark.parametrize("fake", [True], indirect=True)
def test_audio_mode(fake, tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    j = _judge(fake, "gemma-audio")
    r = j.rate(_clip(tmp_path), "Tor", "de")
    assert j.mode == "audio" and r["mode"] == "audio" and r["native"] == 4
    j.rate(_clip(tmp_path), "Tor", "de")
    # detection ran once (1 models GET + 1 probe), then 2 ratings: cached
    assert sum(1 for m, p, _ in fake.log if p == "/v1/models") == 1
    assert sum(1 for m, p, _ in fake.log if p == "/v1/chat/completions") == 3
    assert fake.auth_seen == {f"Bearer {TOKEN}"}


@pytest.mark.parametrize("fake", [False], indirect=True)
def test_transcription_fallback_mode(fake, tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    j = _judge(fake)
    r = j.rate(_clip(tmp_path), "Tor fuer Bayern", "de")
    assert j.mode == "transcribe" and r["mode"] == "transcribe"
    assert r["transcript"] == "Tor fuer Bayern" and r["exact"] == 5
    assert r["native"] is None and r["comment"] == NOT_ASSESSABLE
    assert ("POST", "/v1/audio/transcriptions", {"multipart": True}) in fake.log


@pytest.mark.parametrize("fake", [True], indirect=True)
def test_wrong_or_missing_token_is_a_readable_error(fake, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(TOKEN_ENV, "sk-unsloth-wrong")
    j = _judge(fake)
    with pytest.raises(Exception, match=TOKEN_ENV):
        j.detect()
    r = j.rate(_clip(tmp_path), "Tor", "de")
    assert TOKEN_ENV in r["error"]
    assert "sk-unsloth-wrong" not in caplog.text and "sk-unsloth-wrong" not in r["error"]
    monkeypatch.delenv(TOKEN_ENV)
    assert TOKEN_ENV in j.rate(_clip(tmp_path), "Tor", "de")["error"]


@pytest.mark.parametrize("fake", [True], indirect=True)
def test_unknown_model_lists_available(fake, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    with pytest.raises(Exception, match="gemma-audio"):
        _judge(fake, "nope").detect()


def test_url_and_model_validation():
    for bad in ("http://localhost:8888", "http://localhost:8888/v2", "https://api.openai.com/v1"):
        with pytest.raises(ValueError):
            LocalJudge(bad, "default")
    with pytest.raises(ValueError):
        LocalJudge("http://localhost:8888/v1", "")
