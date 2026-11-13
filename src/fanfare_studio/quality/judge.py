"""Optional LLM judge via a LOCAL Unsloth instance (OpenAI-compatible API).

Disabled unless configured. Start the server with ``unsloth run --model <audio-capable GGUF>``;
it prints the port (default ``http://localhost:8888``). Config (eval.toml)::

    [judge]
    enabled = true
    url   = "http://localhost:8888/v1"   # must be localhost / 127.0.0.1 / [::1] and end with /v1
    model = "default"                    # or an id from GET /v1/models

The API key (``sk-unsloth-...``) is read ONLY from the environment variable
``UNSLOTH_STUDIO_AUTH_TOKEN`` and sent as ``Authorization: Bearer ...``. It is never read from a
config file and never logged.

Audio input to chat completions is not documented by Unsloth, so on first use the judge probes:

1. ``GET {url}/models`` (401 -> readable hint; also validates ``judge.model``),
2. ``GET {server}/openapi.json`` (which routes exist),
3. one chat completion with an ``input_audio`` part (base64 wav) on a 1 s test clip.

If the server accepts the audio: mode ``audio`` (full rubric incl. accent). Otherwise the judge
falls back to ``POST {url}/audio/transcriptions`` plus a text-only rubric (mode ``transcribe``;
accent is reported as not assessable). The detected mode is cached for the run.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import struct
import urllib.error
import urllib.request
import uuid
import wave
from urllib.parse import urlsplit

log = logging.getLogger("fanfare_studio")

TOKEN_ENV = "UNSLOTH_STUDIO_AUTH_TOKEN"
NOT_ASSESSABLE = "accent not assessable in fallback mode"

RUBRIC = (
    "You are rating a text-to-speech clip for a sports stadium announcement in language '{lang}'. "
    "Intended text: \"{text}\". Rate 1-5 each: native (sounds like a native {lang} speaker, no "
    "foreign accent), exact (says exactly the intended text, no extra or missing words), announcer "
    "(energetic, clear announcer style), quality (clean audio, no artefacts). Answer ONLY with JSON: "
    '{{"native": n, "exact": n, "announcer": n, "quality": n, "comment": "..."}}'
)
TEXT_RUBRIC = (
    "A text-to-speech clip in language '{lang}' was automatically transcribed. Intended text: "
    "\"{text}\". Transcript: \"{transcript}\". Rate 1-5: exact (the transcript matches the intended "
    "text, no extra or missing words). You cannot hear the audio, so do not judge accent or quality. "
    'Answer ONLY with JSON: {{"exact": n, "comment": "..."}}'
)


class JudgeSetupError(RuntimeError):
    """Readable configuration / connection problem (shown to the user as is)."""


def is_local_url(url: str) -> bool:
    try:
        u = urlsplit(url)
        u.port  # noqa: B018 - validates the port
    except ValueError:
        return False
    return u.scheme in ("http", "https") and u.hostname in ("localhost", "127.0.0.1", "::1") \
        and not u.username and not u.password


def _test_clip(seconds: float = 1.0, sr: int = 16000) -> bytes:
    n = int(sr * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * i / sr))) for i in range(n)))
    return buf.getvalue()


def _json_from(content: str) -> dict:
    start, end = content.find("{"), content.rfind("}")
    return json.loads(content[start : end + 1])


class LocalJudge:
    def __init__(self, url: str, model: str, timeout: float = 120.0) -> None:
        if not is_local_url(url):
            raise ValueError(f"judge.url must be a localhost URL (localhost, 127.0.0.1 or [::1]), got {url!r}")
        url = url.rstrip("/")
        if not url.endswith("/v1"):
            raise ValueError(f"judge.url must end with /v1 (e.g. http://localhost:8888/v1), got {url!r}")
        if not model or not isinstance(model, str):
            raise ValueError("judge.model must be a model id from GET /v1/models, or 'default'")
        self.url = url
        self.root = url[: -len("/v1")]  # server root, for /openapi.json
        self.model = model
        self.timeout = timeout
        self.mode: str | None = None  # "audio" | "transcribe"; detected on first use, cached
        self._warned = False

    # --- HTTP -----------------------------------------------------------------
    def _headers(self, extra: dict | None = None) -> dict:
        h = dict(extra or {})
        token = os.environ.get(TOKEN_ENV)  # the only place the key is read; never logged
        if token:
            h["Authorization"] = f"Bearer {token}"
        return h

    def _request(self, method: str, url: str, body: bytes | None = None, ctype: str | None = None):  # noqa: ANN202
        headers = self._headers({"Content-Type": ctype} if ctype else None)
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except (urllib.error.URLError, OSError) as e:
            raise JudgeSetupError(
                f"cannot reach the Unsloth server at {self.url} ({type(e).__name__}); "
                "start it with `unsloth run --model <audio-capable GGUF>` and check judge.url"
            ) from e

    def _chat(self, content, max_tokens: int | None = None) -> tuple[int, str]:  # noqa: ANN001
        body = {"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": content}]}
        if max_tokens:
            body["max_tokens"] = max_tokens
        status, raw = self._request("POST", f"{self.url}/chat/completions", json.dumps(body).encode(),
                                    "application/json")
        if status == 401:
            raise self._auth_error()
        if status != 200:
            return status, raw.decode("utf-8", "replace")[:300]
        try:
            return 200, json.loads(raw)["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            return 502, "unexpected response shape"

    @staticmethod
    def _auth_error() -> JudgeSetupError:
        return JudgeSetupError(
            f"Unsloth rejected the API key (401). Check {TOKEN_ENV}: it must hold the "
            "sk-unsloth-... key of the running instance."
        )

    # --- first-use detection ------------------------------------------------------
    def detect(self) -> str:
        if self.mode:
            return self.mode
        status, raw = self._request("GET", f"{self.url}/models")
        if status == 401:
            raise self._auth_error()
        if status != 200:
            raise JudgeSetupError(f"GET {self.url}/models returned HTTP {status}; is this an Unsloth /v1 URL?")
        try:
            ids = [m["id"] for m in json.loads(raw)["data"]]
        except (ValueError, KeyError, TypeError) as e:
            raise JudgeSetupError(f"GET {self.url}/models did not return an OpenAI model list") from e
        if self.model != "default" and self.model not in ids:
            raise JudgeSetupError(f"judge.model {self.model!r} not served; available: {', '.join(ids) or 'none'}")

        has_transcriptions = None
        status, raw = self._request("GET", f"{self.root}/openapi.json")
        if status == 200:
            try:
                paths = json.loads(raw).get("paths", {})
                has_transcriptions = any("audio/transcriptions" in p for p in paths)
            except (ValueError, AttributeError):
                pass
        status, _ = self._chat([
            {"type": "text", "text": "Reply with the single word: ok"},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(_test_clip()).decode(), "format": "wav"}},
        ], max_tokens=8)
        if status == 200:
            self.mode = "audio"
        else:
            if has_transcriptions is False:
                raise JudgeSetupError(
                    f"server rejects audio chat input (HTTP {status}) and has no /audio/transcriptions route; "
                    "load an audio-capable model (unsloth run --model <audio-capable GGUF>)"
                )
            self.mode = "transcribe"
        log.info("judge: mode %s (model %s)", self.mode, self.model)
        return self.mode

    # --- rating ---------------------------------------------------------------------
    def rate(self, path: str, text: str, lang: str) -> dict:
        try:
            mode = self.detect()
            with open(path, "rb") as f:
                wav = f.read()
            if mode == "audio":
                status, content = self._chat([
                    {"type": "text", "text": RUBRIC.format(lang=lang, text=text)},
                    {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
                ])
                if status != 200:
                    return {"mode": mode, "error": f"HTTP {status}"}
                return {"mode": mode, **_json_from(content)}
            return self._rate_transcribed(wav, text, lang)
        except JudgeSetupError as e:
            if not self._warned:  # once per run, readable
                self._warned = True
                log.warning("judge: %s", e)
            return {"error": str(e)}
        except Exception as e:  # noqa: BLE001 - a flaky judge must not stop an eval run
            return {"error": type(e).__name__}

    def _rate_transcribed(self, wav: bytes, text: str, lang: str) -> dict:
        boundary = uuid.uuid4().hex
        parts = b""
        for k, v in (("model", self.model), ("language", lang), ("response_format", "json")):
            parts += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n').encode()
        parts += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
                  "Content-Type: audio/wav\r\n\r\n").encode() + wav + f"\r\n--{boundary}--\r\n".encode()
        status, raw = self._request("POST", f"{self.url}/audio/transcriptions", parts,
                                    f"multipart/form-data; boundary={boundary}")
        if status == 401:
            raise self._auth_error()
        if status != 200:
            return {"mode": "transcribe", "error": f"transcription HTTP {status}", "comment": NOT_ASSESSABLE}
        transcript = json.loads(raw).get("text", "")
        out: dict = {"mode": "transcribe", "transcript": transcript, "native": None, "announcer": None,
                     "quality": None, "comment": NOT_ASSESSABLE}
        status, content = self._chat(TEXT_RUBRIC.format(lang=lang, text=text, transcript=transcript))
        if status == 200:
            try:
                out["exact"] = _json_from(content).get("exact")
            except ValueError:
                pass
        return out


def from_config(cfg: dict) -> LocalJudge | None:
    j = cfg.get("judge", {})
    if not j.get("enabled"):
        return None
    return LocalJudge(j["url"], j.get("model", "default"))
