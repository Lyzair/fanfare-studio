from __future__ import annotations

import io
import json
import threading
import urllib.request
import wave

import numpy as np
import pytest

from fanfare_studio.backends import create_backend
from fanfare_studio.backends.base import BackendError
from fanfare_studio.backends.mock import MockBackend
from fanfare_studio.server import VoiceServer
from fanfare_studio.watermark import WatermarkError, WatermarkedBackend, Watermarker


class FakePerth:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def apply_watermark(self, wav, sample_rate):
        self.calls.append((len(wav), sample_rate))
        return np.asarray(wav, dtype=np.float32) * 0.5  # audible marker for the test


class BrokenPerth:
    def apply_watermark(self, wav, sample_rate):
        raise RuntimeError("boom")


def _peak(wav: bytes) -> int:
    with wave.open(io.BytesIO(wav)) as w:
        return int(np.abs(np.frombuffer(w.readframes(w.getnframes()), "<i2")).max())


def test_backend_output_is_watermarked():
    fake = FakePerth()
    b = WatermarkedBackend(MockBackend(), Watermarker(fake))
    raw = MockBackend().synthesize("hello", "mock-en")
    out = b.synthesize("hello", "mock-en")
    assert fake.calls and fake.calls[0][1] == 16000
    assert _peak(out) < _peak(raw) * 0.6
    assert b.watermarked_calls == 1
    assert b.name == "mock" and b.voices()  # pass-through


def test_http_path_goes_through_watermark():
    fake = FakePerth()
    srv = VoiceServer(WatermarkedBackend(MockBackend(), Watermarker(fake)), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        host, port = srv.server_address[:2]
        req = urllib.request.Request(
            f"http://{host}:{port}/synthesize",
            data=json.dumps({"voice": "mock-en", "text": "hi"}).encode(),
            headers={"Content-Type": "application/json", "X-Fanfare": "1"},
        )
        with urllib.request.urlopen(req) as r:
            assert r.headers["Content-Type"] == "audio/wav"
        assert len(fake.calls) == 1
    finally:
        srv.shutdown()
        srv.server_close()


def test_watermark_failure_is_an_error_not_unmarked_audio():
    b = WatermarkedBackend(MockBackend(), Watermarker(BrokenPerth()))
    with pytest.raises(BackendError):
        b.synthesize("hello", "mock-en")


def test_create_backend_wraps_every_backend_and_refuses_without_watermarker(monkeypatch):
    fake = FakePerth()
    b = create_backend("mock", watermarker=Watermarker(fake))
    b.synthesize("x", "mock-de")
    assert fake.calls

    import builtins

    real_import = builtins.__import__

    def no_perth(name, *a, **k):
        if name == "perth":
            raise ImportError("no perth")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_perth)
    with pytest.raises(WatermarkError, match="never produces unwatermarked"):
        create_backend("mock")


def test_real_perth_marks_speechlike_audio():
    pytest.importorskip("perth")
    wm = Watermarker()
    raw = MockBackend().synthesize("a longer sentence for the mark", "mock-en")
    out = wm.apply_wav(raw)
    assert out != raw
