"""python scripts/check_tag.py <ref>: exit 0 when the git tag (refs/tags/v1.0.0 or v1.0.0) is
"v" + the version in pyproject.toml, so a release never ships files with another version."""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def tag_matches(ref: str, version: str) -> bool:
    if ref.startswith("refs/") and not ref.startswith("refs/tags/"):
        return False
    return ref.removeprefix("refs/tags/") == f"v{version}"


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else ""
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if not tag_matches(ref, version):
        print(f"Tag {ref.removeprefix('refs/tags/')} does not match pyproject.toml version {version}", file=sys.stderr)
        return 1
    print(f"Tag v{version} matches pyproject.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
