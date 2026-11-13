"""`fanfare-studio allow | allowed | disallow` — websites allowed to use the server."""

from __future__ import annotations

from . import origins


def cmd_allow(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: fanfare-studio allow <origin>   e.g. https://<user>.github.io")
        return 2
    try:
        o = origins.save(argv[0])
    except origins.OriginError as e:
        print(f"Error: {e} (wildcards are not allowed)")
        return 1
    print(f"allowed {o} (saved to {origins.config_path()}; a running server picks it up after restart)")
    return 0


def cmd_allowed(argv: list[str]) -> int:
    print("built-in (local development): " + ", ".join(origins.DEFAULT_PATTERNS))
    saved = origins.load_saved()
    print(f"allowed websites ({origins.config_path()}):")
    for o in saved:
        print(f"  {o}")
    if not saved:
        print("  (none)")
    return 0


def cmd_disallow(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: fanfare-studio disallow <origin>")
        return 2
    try:
        removed = origins.remove(argv[0])
    except origins.OriginError as e:
        print(f"Error: {e}")
        return 1
    print(f"removed {argv[0]}" if removed else f"{argv[0]} was not allowed")
    return 0 if removed else 1
