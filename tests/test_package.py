"""The release package: what goes in, and that it starts from the unpacked folder."""
from __future__ import annotations

import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import package  # noqa: E402  (scripts/package.py)


def names_and_modes() -> dict[str, int]:
    return {name: mode for _, name, mode in package.file_list(ROOT)}


def test_package_holds_what_running_needs():
    names = names_and_modes()
    for name in (
        "fanfare-studio/src/fanfare_studio/server.py",
        "fanfare-studio/pyproject.toml",
        "fanfare-studio/uv.lock",
        "fanfare-studio/README.md",
        "fanfare-studio/LICENSE",
        "fanfare-studio/voices.toml",
        "fanfare-studio/voices/de-thorsten.wav",
        "fanfare-studio/docs/model-choice.md",
        "fanfare-studio/docs/manual/en.md",
        "fanfare-studio/docs/manual/de.md",
        "fanfare-studio/start-studio.sh",
        "fanfare-studio/start-studio.command",
        "fanfare-studio/start-studio.bat",
    ):
        assert name in names, name


def test_package_leaves_out_development_files():
    for name in names_and_modes():
        rel = name.removeprefix("fanfare-studio/")
        assert not rel.startswith(("tests/", "scripts/", "samples/", "models/", "tune-out/", ".")), name
        assert rel not in ("eval.toml", "Dockerfile", ".dockerignore"), name
        assert not rel.startswith("compose"), name
        assert "__pycache__" not in name, name


def test_start_files_are_executable():
    modes = names_and_modes()
    assert modes["fanfare-studio/start-studio.sh"] == 0o755
    assert modes["fanfare-studio/start-studio.command"] == 0o755
    assert modes["fanfare-studio/README.md"] == 0o644


def test_missing_manual_is_a_clear_error(tmp_path):
    (tmp_path / "src").mkdir()
    with pytest.raises(SystemExit, match="docs/manual/en.md"):
        package.file_list(tmp_path)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="runs start-studio.sh")
def test_unpacked_package_starts_from_another_folder(tmp_path):
    out = tmp_path / "out"
    archives = package.build(ROOT, out)
    tgz = next(a for a in archives if a.name.endswith(".tar.gz"))
    with tarfile.open(tgz) as t:
        t.extractall(tmp_path / "unpacked", filter="tar")
    folder = tmp_path / "unpacked" / "fanfare-studio"
    port = _free_port()
    # started from tmp_path, not from inside the folder: the start file must find voices.toml itself
    proc = subprocess.Popen(
        [str(folder / "start-studio.sh"), "--backend", "mock", "--port", str(port)],
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            assert proc.poll() is None, proc.stdout.read()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/voices", timeout=2) as r:
                    assert r.status == 200
                    return
            except OSError:
                time.sleep(1)
        pytest.fail("server did not answer /voices")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def test_program_version_matches_pyproject():
    from fanfare_studio import __version__

    assert __version__ == package.version(ROOT)


@pytest.mark.parametrize("lang", ["en", "de"])
def test_manual_mentions(lang):
    text = (ROOT / "docs" / "manual" / f"{lang}.md").read_text(encoding="utf-8")
    for needle in ("start-studio", "lyzair.github.io", "--consent", "compose.rocm.yaml", "compose.cuda.yaml",
                   "ghcr.io/lyzair/fanfare-studio", "127.0.0.1:5123:5123"):
        assert needle in text, needle
