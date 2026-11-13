"""HTTP server implementing the Fanfare local voice protocol.

Stdlib only (http.server.ThreadingHTTPServer): the protocol has two routes, so a
framework would add install weight and attack surface without saving code.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .backends.base import Backend, BackendError, UnknownVoiceError

HOST = "127.0.0.1"  # never 0.0.0.0: the server must not be reachable from the network
DEFAULT_PORT = 5123
MAX_TEXT_CHARS = 1000
MAX_BODY_BYTES = 64 * 1024

log = logging.getLogger("fanfare_studio")


class _Handler(BaseHTTPRequestHandler):
    server: VoiceServer  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"
    server_version = "FanfareStudio/0.1"
    sys_version = ""

    def version_string(self) -> str:
        return self.server_version

    # --- helpers -----------------------------------------------------------
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self._status = status
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _text(self, status: int, message: str) -> None:
        self._send(status, (message + "\n").encode(), "text/plain; charset=utf-8")

    def _path(self) -> str:
        return urlsplit(self.path).path

    # --- logging: method, path, status, duration only — never bodies/queries ----
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # silence BaseHTTPRequestHandler's default access log (it includes the raw request line)

    def log_error(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def handle_one_request(self) -> None:
        self._status = 0
        start = time.perf_counter()
        super().handle_one_request()
        if self._status and getattr(self, "command", None):
            log.info(
                "%s %s %d %.0fms",
                self.command,
                self._path(),
                self._status,
                (time.perf_counter() - start) * 1000,
            )

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        # Used by BaseHTTPRequestHandler for malformed requests / unsupported methods.
        self.close_connection = True
        self._text(code, message or HTTPStatus(code).phrase)

    # --- routes --------------------------------------------------------------
    def do_OPTIONS(self) -> None:  # noqa: N802
        self._status = 204
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self._path()
        if path == "/voices":
            body = json.dumps(
                [{"id": v.id, "lang": v.lang, "label": v.label} for v in self.server.backend.voices()]
            ).encode()
            self._send(200, body, "application/json")
        elif path in ("/", "/health"):
            self._text(200, f"Fanfare Studio ({self.server.backend.name}) ok")
        else:
            self._text(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if self._path() != "/synthesize":
            self._drain()
            self._text(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            self._text(411, "Content-Length required")
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self.close_connection = True
            self._text(413, "request body too large")
            return
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._text(400, "body must be JSON: {\"text\": \"...\", \"voice\": \"...\"}")
            return
        if not isinstance(data, dict):
            self._text(400, "body must be a JSON object")
            return
        text, voice = data.get("text"), data.get("voice")
        if not isinstance(voice, str) or not voice:
            self._text(400, "missing \"voice\"")
            return
        if not isinstance(text, str) or not text.strip():
            self._text(400, "missing or empty \"text\"")
            return
        if len(text) > MAX_TEXT_CHARS:
            self._text(400, f"text longer than {MAX_TEXT_CHARS} characters")
            return
        try:
            wav = self.server.synthesize(text, voice)
        except UnknownVoiceError:
            self._text(404, f"unknown voice: {voice}")
            return
        except BackendError as e:
            log.error("synthesis failed: %s", type(e).__name__)
            self._text(500, "synthesis failed")
            return
        except Exception as e:  # noqa: BLE001 - any model crash is a 500, never a dropped socket
            log.error("synthesis crashed: %s", type(e).__name__)
            self._text(500, "synthesis failed")
            return
        self._send(200, wav, "audio/wav")

    def _drain(self) -> None:
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        if 0 < n <= MAX_BODY_BYTES:
            self.rfile.read(n)
        elif n:
            self.close_connection = True


class VoiceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, backend: Backend, port: int = DEFAULT_PORT) -> None:
        self.backend = backend
        # One synthesis at a time: GPU memory is limited and models are not thread-safe.
        self._synth_lock = threading.Lock()
        super().__init__((HOST, port), _Handler)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        with self._synth_lock:
            return self.backend.synthesize(text, voice_id)
