from __future__ import annotations

import argparse
import logging
import os
import sys

from . import __version__
from .backends import BACKENDS, DEFAULT_BACKEND, create_backend
from .runtime import EP_CHOICES, TORCH_DEVICES
from .server import DEFAULT_PORT, HOST, VoiceServer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fanfare-studio",
        description="Fanfare Studio: local text-to-speech server for the Fanfare app "
        "(binds 127.0.0.1 only, never logs request text). "
        "Dev tools: fanfare-studio eval|tune|prefer --help.",
    )
    p.add_argument("--version", action="version", version=f"Fanfare Studio {__version__}")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}")
    p.add_argument(
        "--backend",
        choices=BACKENDS,
        default=DEFAULT_BACKEND,
        help=f"default {DEFAULT_BACKEND} (Chatterbox, 4-bit ONNX, CPU)",
    )
    p.add_argument(
        "--device", choices=TORCH_DEVICES, default="auto", help="chatterbox (PyTorch, experimental) only"
    )
    p.add_argument(
        "--ep",
        choices=EP_CHOICES,
        default="auto",
        help="ONNX Runtime execution provider (piper, chatterbox-onnx); falls back to CPU",
    )
    p.add_argument(
        "--onnx-variant",
        choices=("fp32", "fp16", "q4"),
        default="q4",
        help="chatterbox-onnx: language-model precision (default q4: ~3 GB RAM, fastest on CPU)",
    )
    p.add_argument(
        "--voices-config",
        default="voices.toml",
        help="chatterbox backends: per-voice reference clip and settings (default voices.toml, if present)",
    )
    p.add_argument(
        "--torch-math-sdp",
        action="store_true",
        help="chatterbox (torch): disable fused attention kernels (diagnostic)",
    )
    p.add_argument("--voices-dir", default="models/piper", help="piper: directory with *.onnx voices")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from .cli_quality import dispatch

    rc = dispatch(argv)
    if rc is not None:
        return rc
    args = build_parser().parse_args(argv)

    # No usage telemetry from the Hugging Face libraries (model downloads still need the network
    # on first start; set HF_HUB_OFFLINE=1 afterwards to run fully offline).
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("fanfare_studio")
    log.info("Fanfare Studio %s - loading backend %s ...", __version__, args.backend)
    backend = create_backend(
        args.backend, device=args.device, ep=args.ep, voices_dir=args.voices_dir,
        onnx_variant=args.onnx_variant,
        voices_config=args.voices_config,
        math_sdp=args.torch_math_sdp,
    )
    server = VoiceServer(backend, args.port)
    log.info(
        "Fanfare Studio ready: backend %s on %s, %d voices; listening on http://%s:%d",
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
