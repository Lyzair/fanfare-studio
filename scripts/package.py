"""Build the release package: release/fanfare-studio-<version>.zip and .tar.gz.

Only what running Fanfare Studio needs (source, lock file, voice settings, reference voices,
manuals, licence) plus the start files; tests, dev tools and Docker files stay out. Every archive
unpacks into fanfare-studio/. Usage: uv run python scripts/package.py [--out release]
"""
from __future__ import annotations

import argparse
import tarfile
import tomllib
import zipfile
from pathlib import Path

TOP = "fanfare-studio"
PACKAGE_FILES = [
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "voices.toml",
    "docs/model-choice.md",
    "docs/manual/en.md",
    "docs/manual/de.md",
]
PACKAGE_DIRS = ["src", "voices"]
START_FILES = {"start-studio.sh": 0o755, "start-studio.command": 0o755, "start-studio.bat": 0o644}


def file_list(root: Path) -> list[tuple[Path, str, int]]:
    """[(source, archive name, unix mode)]; exits naming every required file that is missing."""
    required = [root / f for f in PACKAGE_FILES] + [root / "start" / f for f in START_FILES]
    missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing:
        raise SystemExit("package: missing " + ", ".join(missing))
    files = [(root / f, f"{TOP}/{f}", 0o644) for f in PACKAGE_FILES]
    for d in PACKAGE_DIRS:
        for p in sorted((root / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                files.append((p, f"{TOP}/{p.relative_to(root).as_posix()}", 0o644))
    files += [(root / "start" / f, f"{TOP}/{f}", mode) for f, mode in START_FILES.items()]
    return files


def version(root: Path) -> str:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def build(root: Path, out: Path) -> list[Path]:
    files = file_list(root)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{TOP}-{version(root)}"
    zpath, tpath = base.with_name(base.name + ".zip"), base.with_name(base.name + ".tar.gz")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, name, mode in files:
            info = zipfile.ZipInfo.from_file(src, name)
            info.external_attr = (0o100000 | mode) << 16  # keeps the executable bit on unzip
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(src, "rb") as f:
                z.writestr(info, f.read())
    with tarfile.open(tpath, "w:gz", format=tarfile.PAX_FORMAT) as t:
        for src, name, mode in files:
            info = t.gettarinfo(src, name)
            info.mode, info.uid, info.gid, info.uname, info.gname = mode, 0, 0, "", ""
            with open(src, "rb") as f:
                t.addfile(info, f)
    return [zpath, tpath]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="release")
    root = Path(__file__).resolve().parents[1]
    for p in build(root, root / ap.parse_args().out):
        print(f"{p.relative_to(root)}: {p.stat().st_size / 1048576:.1f} MiB")


if __name__ == "__main__":
    main()
