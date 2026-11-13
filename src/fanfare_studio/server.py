"""HTTP server implementing the Fanfare local voice protocol.

Stdlib only (http.server.ThreadingHTTPServer): the protocol has a handful of routes, so a
framework would add install weight and attack surface without saving code.

Routes: GET /voices, POST /synthesize, POST /references (multipart), DELETE /references/{id}.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from email.parser import BytesParser
from email.policy import HTTP as HTTP_POLICY
from pathlib import Path

from . import references as refs
from .origins import OriginPolicy
from .backends.base import Backend, BackendError, UnknownVoiceError, Voice

HOST = "127.0.0.1"  # never 0.0.0.0: the server must not be reachable from the network
DEFAULT_PORT = 5123
MAX_TEXT_CHARS = 1000
MAX_BODY_BYTES = 64 * 1024
MAX_REFERENCE_BODY_BYTES = refs.MAX_UPLOAD_BYTES + 64 * 1024  # audio + form fields

log = logging.getLogger("fanfare_studio")


class _Handler(BaseHTTPRequestHandler):
    server: VoiceServer  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"
    server_version = "FanfareStudio/0.1"
    sys_version = ""

    def version_string(self) -> str:
        return self.server_version

    # --- helpers -----------------------------------------------------------
    _acao: str | None = None

    def _cors(self) -> None:
        # Echo only an allowed origin (never "*"); requests without Origin get no CORS header.
        if self._acao:
            self.send_header("Access-Control-Allow-Origin", self._acao)
        self.send_header("Vary", "Origin")

    def _origin_ok(self) -> bool:
        """Allow-list check for browser requests. Sends 403 (without CORS headers) if not allowed."""
        self._acao = None
        origin = self.headers.get("Origin")
        if origin is None:
            return True  # not a web page (curl, CLI)
        if self.server.origins.allowed(origin):
            self._acao = origin
            return True
        self.server.unknown_origin(origin)
        self._drain()
        self._text(403, f"This website is not allowed to use Fanfare Studio. Allow it with: fanfare-studio allow {origin}")
        return False

    def _require_fanfare_header(self) -> bool:
        # Forces a CORS preflight for write requests (multipart/form-data alone is a "simple"
        # request a foreign page could send without one).
        if self.headers.get("X-Fanfare") == "1":
            return True
        self._drain()
        self._text(400, "missing header X-Fanfare: 1")
        return False

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
        if getattr(self, "headers", None) is not None:
            origin = self.headers.get("Origin")
            self._acao = origin if origin and self.server.origins.allowed(origin) else None
        self._text(code, message or HTTPStatus(code).phrase)

    # --- routes --------------------------------------------------------------
    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._origin_ok():
            return
        self._status = 204
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Fanfare")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if not self._origin_ok():
            return
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

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._origin_ok():
            return
        path = self._path()
        if path.startswith("/references/") and self.server.references_enabled:
            if not self._require_fanfare_header():
                return
            self._drain()
            if self.server.delete_reference(path.removeprefix("/references/")):
                self._send(204, b"", "text/plain; charset=utf-8")
            else:
                self._text(404, "unknown reference")
            return
        self._drain()
        self._text(404, "not found")

    def _references_post(self) -> None:
        ctype = self.headers.get("Content-Type", "")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            self._text(411, "Content-Length required")
            return
        if length < 0 or length > MAX_REFERENCE_BODY_BYTES:
            self.close_connection = True
            self._text(413, "upload larger than 10 MB")
            return
        if not ctype.lower().startswith("multipart/form-data"):
            self._drain()
            self._text(400, "expected multipart/form-data with fields audio, name, lang, consent")
            return
        body = self.rfile.read(length)
        msg = BytesParser(policy=HTTP_POLICY).parsebytes(
            b"Content-Type: " + ctype.encode("latin-1") + b"\r\n\r\n" + body
        )
        fields: dict[str, bytes] = {}
        if msg.is_multipart():
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name:
                    fields[str(name)] = part.get_payload(decode=True) or b""
        audio = fields.get("audio")
        if not audio:
            self._text(400, "missing audio")
            return
        if len(audio) > refs.MAX_UPLOAD_BYTES:
            self._text(413, "audio larger than 10 MB")
            return
        consent = fields.get("consent", b"").decode("utf-8", "replace").strip().lower() == "true"
        name = fields.get("name", b"").decode("utf-8", "replace")
        lang = fields.get("lang", b"").decode("utf-8", "replace").strip()
        try:
            ref, warnings = self.server.add_reference(name, lang, audio, consent)
        except refs.ReferenceError as e:
            self._text(400, str(e))
            return
        except Exception as e:  # noqa: BLE001
            log.error("adding reference failed: %s", type(e).__name__)
            self._text(500, "could not add the voice")
            return
        body_out = json.dumps({"id": ref.id, "lang": ref.lang, "label": ref.label, "warnings": warnings})
        self._send(201, body_out.encode(), "application/json")

    def do_POST(self) -> None:  # noqa: N802
        if not self._origin_ok():
            return
        if self._path() == "/references" and self.server.references_enabled:
            if self._require_fanfare_header():
                self._references_post()
            return
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

    def __init__(
        self,
        backend: Backend,
        port: int = DEFAULT_PORT,
        references_root: Path | None = None,
        origins: OriginPolicy | None = None,
        prompt_unknown_origins: bool = False,
    ) -> None:
        self.backend = backend
        self.origins = origins or OriginPolicy()
        self._prompter = _OriginPrompter(self.origins) if prompt_unknown_origins else None
        # One synthesis at a time: GPU memory is limited and models are not thread-safe.
        self._synth_lock = threading.Lock()
        # Club reference voices need a backend that clones voices (chatterbox*, mock).
        self.references_enabled = hasattr(backend, "add_voice")
        self.references_root = references_root
        super().__init__((HOST, port), _Handler)
        if self.references_enabled:
            for ref in refs.list_all(references_root):
                try:
                    self._register(ref)
                except Exception as e:  # noqa: BLE001 - one broken folder must not stop the server
                    log.warning("skipping club voice %s: %s", ref.id, type(e).__name__)

    def unknown_origin(self, origin: str) -> None:
        log.warning("blocked request from website %s (not in the allow-list)", origin)
        if self._prompter is not None:
            self._prompter.ask(origin)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        with self._synth_lock:
            return self.backend.synthesize(text, voice_id)

    def _register(self, ref: refs.Reference) -> None:
        import dataclasses

        try:
            base = self.backend.settings(f"{ref.lang}-high")  # type: ignore[attr-defined]
        except KeyError:
            base = self.backend.settings(ref.id)  # type: ignore[attr-defined]
        settings = dataclasses.replace(base, reference=str(ref.path))
        self.backend.add_voice(Voice(ref.id, ref.lang, ref.label), settings)  # type: ignore[attr-defined]

    def add_reference(self, name: str, lang: str, audio: bytes, consent: bool):  # type: ignore[no-untyped-def]
        ref, warnings = refs.add(name, lang, audio, consent, self.references_root)
        try:
            with self._synth_lock:
                self._register(ref)
        except Exception:
            refs.delete(ref.id, self.references_root)
            raise
        return ref, warnings

    def delete_reference(self, rid: str) -> bool:
        with self._synth_lock:
            if not refs.delete(rid, self.references_root):
                return False
            self.backend.remove_voice(rid)  # type: ignore[attr-defined]
            return True


class _OriginPrompter:
    """Asks once per unknown origin on the terminal, in a background thread (never blocks
    a request; the triggering request already got 403). Never allows without an explicit "y"."""

    def __init__(self, policy: OriginPolicy) -> None:
        import queue

        self.policy = policy
        self._asked: set[str] = set()
        self._queue: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def ask(self, origin: str) -> None:
        if origin not in self._asked:
            self._asked.add(origin)
            self._queue.put(origin)

    def _run(self) -> None:
        while True:
            origin = self._queue.get()
            try:
                answer = input(f"\nWebsite {origin} wants to use Fanfare Studio. Allow? [y/N] ")
            except EOFError:
                return
            if answer.strip().lower() in ("y", "yes"):
                try:
                    saved = self.policy.allow(origin)
                    print(f"allowed {saved} (saved to {self.policy.path})", flush=True)
                except ValueError as e:
                    print(f"not allowed: {e}", flush=True)
            else:
                print(f"not allowed: {origin}", flush=True)
