"""Optional LLM judge via a LOCAL OpenAI-compatible endpoint (e.g. Qwen2.5-Omni behind vLLM).

Disabled unless configured; only http(s)://localhost / 127.0.0.1 / [::1] URLs are accepted, so
clips never leave the machine.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from urllib.parse import urlsplit

RUBRIC = (
    "You are rating a text-to-speech clip for a sports stadium announcement in language '{lang}'. "
    "Intended text: \"{text}\". Rate 1-5 each: native (sounds like a native {lang} speaker, no "
    "foreign accent), exact (says exactly the intended text, no extra or missing words), announcer "
    "(energetic, clear announcer style), quality (clean audio, no artefacts). Answer ONLY with JSON: "
    '{{"native": n, "exact": n, "announcer": n, "quality": n, "comment": "..."}}'
)


def is_local_url(url: str) -> bool:
    try:
        u = urlsplit(url)
    except ValueError:
        return False
    return u.scheme in ("http", "https") and u.hostname in ("localhost", "127.0.0.1", "::1") \
        and not u.username and not u.password


class LocalJudge:
    def __init__(self, url: str, model: str, timeout: float = 120.0) -> None:
        if not is_local_url(url):
            raise ValueError(f"judge.url must be a localhost URL, got {url!r}")
        self.url = url.rstrip("/") + "/chat/completions" if not url.rstrip("/").endswith("/chat/completions") else url
        self.model = model
        self.timeout = timeout

    def rate(self, path: str, text: str, lang: str) -> dict:
        audio = base64.b64encode(open(path, "rb").read()).decode()
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": RUBRIC.format(lang=lang, text=text)},
                {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
            ]}],
        }  # fmt: skip
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                content = json.load(r)["choices"][0]["message"]["content"]
            start, end = content.find("{"), content.rfind("}")
            return json.loads(content[start : end + 1])
        except Exception as e:  # noqa: BLE001
            return {"error": type(e).__name__}


def from_config(cfg: dict) -> LocalJudge | None:
    j = cfg.get("judge", {})
    if not j.get("enabled"):
        return None
    return LocalJudge(j["url"], j["model"])
