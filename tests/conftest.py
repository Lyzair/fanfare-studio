from __future__ import annotations

import threading

import pytest

from fanfare_studio.backends.mock import MockBackend
from fanfare_studio.server import VoiceServer


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Never touch the real per-user data dir (club voices, preferences) in tests."""
    monkeypatch.setenv("FANFARE_STUDIO_DATA", str(tmp_path / "fanfare-data"))
    monkeypatch.setenv("FANFARE_STUDIO_CONFIG_DIR", str(tmp_path / "fanfare-config"))


@pytest.fixture
def server():
    srv = VoiceServer(MockBackend(), port=0)  # port 0: pick a free port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def base(server) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"
