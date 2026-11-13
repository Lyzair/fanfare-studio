from __future__ import annotations

import threading

import pytest

from fanfare_studio.backends.mock import MockBackend
from fanfare_studio.server import VoiceServer


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
