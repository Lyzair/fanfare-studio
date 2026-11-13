from __future__ import annotations

import argparse
import logging
import os
import sys

from . import __version__
from .backends import BACKENDS, DEFAULT_BACKEND, create_backend
from .runtime import EP_CHOICES, TORCH_DEVICES
from .server import DEFAULT_PORT, HOST, VoiceServer
from .watermark import WatermarkError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fanfare-studio",
        description="Fanfare Studio: local text-to-speech server for the Fanfare app "
        "(binds 127.0.0.1 by default, never logs request text). "
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
    p.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="URL",
        help="also allow this website for this run (repeatable); persistent: fanfare-studio allow URL",
    )
    p.add_argument(
        "--host",
        default=os.environ.get("FANFARE_HOST", HOST),
        help=f"bind address (default {HOST}, env FANFARE_HOST). 0.0.0.0 is for containers only: "
        "publish the port on 127.0.0.1 of the host",
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
    try:
        backend = create_backend(
            args.backend, device=args.device, ep=args.ep, voices_dir=args.voices_dir,
            onnx_variant=args.onnx_variant,
            voices_config=args.voices_config,
            math_sdp=args.torch_math_sdp,
        )
    except WatermarkError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    from .origins import OriginError, OriginPolicy, from_env

    try:
        policy = OriginPolicy(extra=[*args.allow_origin, *from_env()])
    except OriginError as e:
        print(f"Error: {e} (check --allow-origin / FANFARE_ALLOWED_ORIGINS)", file=sys.stderr)
        return 1
    server = VoiceServer(
        backend, args.port, origins=policy, prompt_unknown_origins=sys.stdin.isatty(), host=args.host
    )
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("binding %s: only do this inside a container whose port is published on 127.0.0.1", args.host)
    log.info("allowed websites: %s", ", ".join(policy.listing()))
    log.info(
        "Fanfare Studio ready: backend %s on %s, %d voices; listening on http://%s:%d",
        backend.name,
        backend.runtime,
        len(backend.voices()),
        args.host,
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
