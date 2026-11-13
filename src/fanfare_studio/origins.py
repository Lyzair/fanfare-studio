"""Which websites may use Fanfare Studio (browser Origin allow-list).

The server binds to 127.0.0.1, but every website open in the user's browser can still send
requests to it. So every request with an ``Origin`` header must come from an allowed origin:

* built-in: ``http://localhost:*`` and ``http://127.0.0.1:*`` (local development / preview;
  the port wildcard exists only for these two),
* the user's list in ``<user config dir>/fanfare-studio/allowed-origins.toml``
  (``fanfare-studio allow <origin>``, or "y" at the server's first-use prompt),
* ``--allow-origin`` on the command line and ``FANFARE_ALLOWED_ORIGINS`` (comma-separated), this run
  only (the env var is for containers, where there is no interactive prompt).

Entries match exactly on scheme + host + port. Requests without ``Origin`` (curl, CLI tools)
are not from a web page and are allowed.
"""

from __future__ import annotations

import os
import re
import threading
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_PATTERNS = ("http://localhost:*", "http://127.0.0.1:*")
_DEFAULT_PORTS = {"http": 80, "https": 443}


class OriginError(ValueError):
    pass


def normalize(origin: str) -> str:
    """'HTTPS://Example.org:443/' → 'https://example.org'. Raises OriginError if not an origin."""
    origin = origin.strip()
    try:
        u = urlsplit(origin)
        port = u.port
    except ValueError as e:
        raise OriginError(f"not a valid origin: {origin!r}") from e
    if u.scheme not in _DEFAULT_PORTS or not u.hostname:
        raise OriginError(f"origin must be http(s)://host[:port], got {origin!r}")
    if u.username or u.password or u.query or u.fragment or u.path not in ("", "/"):
        raise OriginError(f"origin must not contain a path, query or credentials: {origin!r}")
    host = u.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[u.scheme]:
        return f"{u.scheme}://{host}"
    return f"{u.scheme}://{host}:{port}"


def _matches_default(origin: str) -> bool:
    return re.fullmatch(r"http://(localhost|127\.0\.0\.1)(:\d{1,5})?", origin) is not None


def config_dir() -> Path:
    env = os.environ.get("FANFARE_STUDIO_CONFIG_DIR")
    if env:
        return Path(env)
    from platformdirs import user_config_dir

    return Path(user_config_dir("fanfare-studio", appauthor=False))


def config_path() -> Path:
    return config_dir() / "allowed-origins.toml"


def load_saved(path: Path | None = None) -> list[str]:
    path = path or config_path()
    if not path.is_file():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out = []
    for o in data.get("allowed_origins", []):
        try:
            out.append(normalize(o))
        except OriginError:
            continue
    return out


def _write(origins: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f'  "{o}",' for o in sorted(set(origins)))
    path.write_text(
        "# Websites allowed to use Fanfare Studio (managed by `fanfare-studio allow|disallow`).\n"
        f"allowed_origins = [\n{body}\n]\n",
        encoding="utf-8",
    )


def save(origin: str, path: Path | None = None) -> str:
    o = normalize(origin)
    path = path or config_path()
    saved = load_saved(path)
    if o not in saved:
        _write(saved + [o], path)
    return o


def remove(origin: str, path: Path | None = None) -> bool:
    o = normalize(origin)
    path = path or config_path()
    saved = load_saved(path)
    if o not in saved:
        return False
    _write([x for x in saved if x != o], path)
    return True


def from_env() -> list[str]:
    """Origins from FANFARE_ALLOWED_ORIGINS (comma-separated, exact match rules, no wildcards)."""
    return [o for o in (os.environ.get("FANFARE_ALLOWED_ORIGINS") or "").split(",") if o.strip()]


class OriginPolicy:
    def __init__(self, extra: list[str] | None = None, path: Path | None = None) -> None:
        self.path = path or config_path()
        self.extra = {normalize(o) for o in (extra or [])}
        self._lock = threading.Lock()
        self._saved = set(load_saved(self.path))

    def allowed(self, origin: str) -> bool:
        try:
            o = normalize(origin)
        except OriginError:
            return False  # includes "null" (sandboxed iframes, file://)
        if _matches_default(o):
            return True
        with self._lock:
            return o in self.extra or o in self._saved

    def allow(self, origin: str) -> str:
        o = save(origin, self.path)
        with self._lock:
            self._saved.add(o)
        return o

    def listing(self) -> list[str]:
        return [*DEFAULT_PATTERNS, *sorted(self._saved), *sorted(self.extra - self._saved)]
