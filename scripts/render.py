"""Render clips in-process with explicit settings (for comparisons / parameter sweeps).

    python scripts/render.py --backend chatterbox-onnx \
        "chatterbox-onnx-q4-cpu-de:de-high:de:reference=voices/de-thorsten.wav,cfg_weight=0.5"

Each job is NAME:VOICE:SENTENCE[:key=value,...]; keys are fields of voice_config.Settings
(reference=none for the built-in speaker). Writes samples/NAME.wav and prints timings.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import logging
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fanfare_studio.backends import create_backend  # noqa: E402

SENTENCES = {
    "de": "Und jetzt, mit der Nummer 7: Lea! Viel Spaß und gib alles!",
    "en": "And now, wearing number 7: Lea! Have fun and give it everything!",
}


def parse_value(v: str):  # type: ignore[no-untyped-def]
    if v.lower() == "none":
        return None
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="chatterbox-onnx")
    p.add_argument("--device", default="auto")
    p.add_argument("--ep", default="auto")
    p.add_argument("--onnx-variant", default="q4")
    p.add_argument("--math-sdp", action="store_true")
    p.add_argument("--voices-config", default=None)
    p.add_argument("--no-normalize", action="store_true", help="reproduce the reference script")
    p.add_argument("jobs", nargs="+")
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    b = create_backend(a.backend, device=a.device, ep=a.ep, onnx_variant=a.onnx_variant,
                       voices_config=a.voices_config, math_sdp=a.math_sdp)
    if a.no_normalize:
        b.normalize = False
    print(f"# backend {b.name} on {b.runtime} normalize={not a.no_normalize}")
    Path("samples").mkdir(exist_ok=True)
    for job in a.jobs:
        parts = job.split(":", 3)
        name, voice, key = parts[:3]
        over = dict(kv.split("=", 1) for kv in parts[3].split(",")) if len(parts) > 3 and parts[3] else {}
        s = dataclasses.replace(b.settings(voice), **{k: parse_value(v) for k, v in over.items()})
        t0 = time.perf_counter()
        wav = b.synthesize(SENTENCES[key], voice, settings=s)
        dt = time.perf_counter() - t0
        with wave.open(io.BytesIO(wav)) as w:
            dur = w.getnframes() / w.getframerate()
        Path(f"samples/{name}.wav").write_bytes(wav)
        ref = Path(s.reference).stem if s.reference else "builtin"
        print(f"{name}.wav\t{dt:.1f}\t{dur:.2f}\tref={ref} cfg={s.cfg_weight} ex={s.exaggeration} "
              f"t={s.temperature} rp={s.repetition_penalty} min_p={s.min_p} seed={s.seed}", flush=True)


if __name__ == "__main__":
    main()
