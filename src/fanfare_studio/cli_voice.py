"""`fanfare-studio voice add | list | delete` — club reference voices."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import references as refs


def cmd_voice(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="fanfare-studio voice", description="Manage your club's own voices.")
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("add", help="add a recording as a new voice")
    a.add_argument("--name", required=True, help=f"label shown in Fanfare (max {refs.MAX_NAME_CHARS} chars)")
    a.add_argument("--lang", required=True, choices=refs.CHATTERBOX_LANGS)
    a.add_argument("--file", required=True, help="wav/webm/ogg/mp3/m4a/flac, 15-20 s of announcer-style speech")
    a.add_argument("--consent", action="store_true", help="the speaker agreed (required): " + refs.CONSENT_TEXT)
    sub.add_parser("list", help="list club voices")
    d = sub.add_parser("delete", help="delete a club voice and its recording")
    d.add_argument("id")
    args = p.parse_args(argv)

    if args.action == "add":
        if not args.consent:
            print("Refused: add --consent to confirm that the speaker agreed.\n"
                  "A voice is personal data. Only add it if its owner agreed that it is stored on this "
                  "computer and used to generate announcements, and delete it when they ask.")  # fmt: skip
            return 2
        try:
            ref, warnings = refs.add(args.name, args.lang, Path(args.file).read_bytes(), consent=True)
        except refs.ReferenceError as e:
            print(f"Error: {e}")
            return 1
        print(f"added {ref.id} ({ref.lang}, {ref.label}) -> {ref.path}")
        for w in warnings:
            print(f"warning: {w}")
        print("Restart the server to use it (voices added over HTTP are available immediately).")
        return 0
    if args.action == "list":
        items = refs.list_all()
        for r in items:
            print(f"{r.id}\t{r.lang}\t{r.created}\t{r.label}")
        if not items:
            print(f"no club voices in {refs.references_dir()}")
        return 0
    if refs.delete(args.id):
        print(f"deleted {args.id}")
        return 0
    print(f"unknown voice {args.id}")
    return 1
