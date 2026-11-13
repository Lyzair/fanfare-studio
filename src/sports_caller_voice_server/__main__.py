from __future__ import annotations

import argparse
import logging
import sys

from .backends import BACKENDS, create_backend
from .runtime import EP_CHOICES, TORCH_DEVICES
from .server import DEFAULT_PORT, HOST, VoiceServer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="sports-caller-voice-server",
        description="Local TTS server for Sports Caller (binds 127.0.0.1 only).",
    )
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}")
    p.add_argument("--backend", choices=BACKENDS, default="chatterbox")
    p.add_argument(
        "--device", choices=TORCH_DEVICES, default="auto", help="torch backends (chatterbox)"
    )
    p.add_argument(
        "--ep",
        choices=EP_CHOICES,
        default="auto",
        help="ONNX Runtime execution provider (piper, chatterbox-onnx); falls back to CPU",
    )
    p.add_argument("--voices-dir", default="models/piper", help="piper: directory with *.onnx voices")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("sports_caller_voice_server")
    log.info("loading backend %s ...", args.backend)
    backend = create_backend(
        args.backend, device=args.device, ep=args.ep, voices_dir=args.voices_dir
    )
    server = VoiceServer(backend, args.port)
    log.info(
        "backend %s on %s, %d voices; listening on http://%s:%d",
        backend.name,
        backend.runtime,
        len(backend.voices()),
        HOST,
        server.server_address[1],
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
