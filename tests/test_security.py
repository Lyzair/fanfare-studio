"""Origin allow-list, first-use approval plumbing, X-Fanfare on write endpoints."""

from __future__ import annotations

import threading

import pytest

from fanfare_studio import origins
from fanfare_studio.__main__ import main
from fanfare_studio.backends.mock import MockBackend
from fanfare_studio.origins import OriginError, OriginPolicy, normalize
from fanfare_studio.server import VoiceServer

from .test_protocol import request

PAGES = "https://fanfare.example.org"
EVIL = "https://evil.example.com"


def make_server(**kw):
    srv = VoiceServer(MockBackend(), port=0, **kw)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def server():
    srv = make_server()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_normalize():
    assert normalize("HTTPS://Fanfare.Example.org:443/") == PAGES
    assert normalize("http://localhost:5173") == "http://localhost:5173"
    for bad in ("null", "file:///x", "https://a.b/path", "https://u:p@a.b", "ftp://a.b", "https://a.b:*"):
        with pytest.raises(OriginError):
            normalize(bad)


def test_policy_matching():
    p = OriginPolicy(extra=[PAGES])
    assert p.allowed("http://localhost:3000") and p.allowed("http://127.0.0.1:8080") and p.allowed("http://localhost")
    assert p.allowed(PAGES) and p.allowed(PAGES + ":443")
    assert not p.allowed(PAGES + ":8443")  # exact port for configured entries
    assert not p.allowed("http://fanfare.example.org")  # exact scheme
    assert not p.allowed("https://localhost:3000")  # defaults are http only
    assert not p.allowed("http://localhost.evil.com:3000")
    assert not p.allowed("http://192.168.1.5:5123")
    assert not p.allowed("null")
    assert not p.allowed(EVIL)


def test_allowed_origin_is_echoed_with_vary(server):
    resp, _ = request(server, "GET", "/voices", headers={"Origin": "http://127.0.0.1:4173"})
    assert resp.status == 200
    assert resp.getheader("Access-Control-Allow-Origin") == "http://127.0.0.1:4173"
    assert "Origin" in resp.getheader("Vary")


def test_unknown_origin_gets_403_without_cors(server):
    for method, path, body in (("GET", "/voices", None), ("POST", "/synthesize", {"text": "x", "voice": "mock-de"}),
                               ("DELETE", "/references/club-1", None)):  # fmt: skip
        resp, payload = request(server, method, path, body, headers={"Origin": EVIL, "X-Fanfare": "1"})
        assert resp.status == 403
        assert resp.getheader("Access-Control-Allow-Origin") is None
        assert resp.getheader("Content-Type").startswith("text/plain")
        assert b"fanfare-studio allow " + EVIL.encode() in payload
    assert server.backend.calls == []


def test_preflight_from_unknown_origin_has_no_allow_headers(server):
    resp, _ = request(server, "OPTIONS", "/references", headers={
        "Origin": EVIL, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "x-fanfare", "Access-Control-Request-Private-Network": "true"})  # fmt: skip
    assert resp.status == 403
    for h in ("Access-Control-Allow-Origin", "Access-Control-Allow-Methods",
              "Access-Control-Allow-Headers", "Access-Control-Allow-Private-Network"):  # fmt: skip
        assert resp.getheader(h) is None


def test_requests_without_origin_are_allowed(server):
    resp, _ = request(server, "GET", "/voices", headers={"Origin": None})
    assert resp.status == 200 and resp.getheader("Access-Control-Allow-Origin") is None
    resp, _ = request(server, "POST", "/synthesize", {"text": "hi", "voice": "mock-de"}, headers={"Origin": None})
    assert resp.status == 200


def test_write_endpoints_require_x_fanfare(server):
    resp, payload = request(server, "POST", "/references", b"--x--", {"Content-Type": "multipart/form-data; boundary=x"})
    assert resp.status == 400 and b"X-Fanfare" in payload
    resp, payload = request(server, "DELETE", "/references/club-12345678")
    assert resp.status == 400 and b"X-Fanfare" in payload
    resp, _ = request(server, "DELETE", "/references/club-12345678", headers={"X-Fanfare": "1"})
    assert resp.status == 404


def test_cli_allow_persists_and_server_picks_it_up(capsys):
    assert main(["allow", PAGES + "/"]) == 0
    assert origins.load_saved() == [PAGES]
    assert main(["allowed"]) == 0
    out = capsys.readouterr().out
    assert PAGES in out and "http://localhost:*" in out
    srv = make_server()
    try:
        resp, _ = request(srv, "GET", "/voices", headers={"Origin": PAGES})
        assert resp.status == 200 and resp.getheader("Access-Control-Allow-Origin") == PAGES
    finally:
        srv.shutdown()
        srv.server_close()
    assert main(["allow", "https://x.example:*"]) == 1  # no wildcards for user entries
    assert main(["disallow", PAGES]) == 0
    assert origins.load_saved() == []
    assert main(["disallow", PAGES]) == 1


def test_prompt_answer_yes_allows_and_persists(monkeypatch):
    """First-use approval: the 403'd origin is queued for one prompt; "y" persists it."""
    answers = iter(["y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    srv = make_server(prompt_unknown_origins=True)
    try:
        assert request(srv, "GET", "/voices", headers={"Origin": PAGES})[0].status == 403
        for _ in range(100):
            if PAGES in origins.load_saved():
                break
            threading.Event().wait(0.02)
        assert origins.load_saved() == [PAGES]
        assert request(srv, "GET", "/voices", headers={"Origin": PAGES})[0].status == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_prompt_answer_no_never_allows(monkeypatch):
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "")
    srv = make_server(prompt_unknown_origins=True)
    try:
        for _ in range(3):
            assert request(srv, "GET", "/voices", headers={"Origin": EVIL})[0].status == 403
        threading.Event().wait(0.2)
        assert len(asked) == 1 and EVIL in asked[0]  # asked once, not per request
        assert origins.load_saved() == []
    finally:
        srv.shutdown()
        srv.server_close()
