"""`fanfare-studio eval | tune | prefer` subcommands."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from pathlib import Path


def _config(path: str) -> dict:
    p = Path(path)
    return tomllib.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _evaluator(cfg: dict):  # type: ignore[no-untyped-def]
    from .quality.evaluator import Evaluator
    from .quality.judge import from_config

    return Evaluator(whisper_model=cfg.get("whisper", {}).get("model", "small"), judge=from_config(cfg))


def cmd_eval(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="fanfare-studio eval", description="Score TTS clips automatically.")
    p.add_argument("wavs", nargs="+")
    p.add_argument("--text", required=True, help="the intended text")
    p.add_argument("--lang", required=True)
    p.add_argument("--reference", help="reference clip for speaker similarity")
    p.add_argument("--config", default="eval.toml")
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = p.parse_args(argv)
    from dataclasses import asdict

    from .quality.prefs import DEFAULT_WEIGHTS, combined, rejected

    cfg = _config(a.config)
    weights = {**DEFAULT_WEIGHTS, **cfg.get("weights", {})}
    max_cer = cfg.get("filters", {}).get("max_cer", 0.15)
    ev = _evaluator(cfg)
    out = []
    for w in a.wavs:
        d = asdict(ev.evaluate(w, a.text, a.lang, a.reference))
        d["rejected"] = rejected(d, max_cer)
        d["combined"] = round(combined(d, weights), 4)
        out.append(d)
    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    f = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
    print("| clip | score | CER | WER | lang | MOS | spk | lead/trail s | flags | transcript |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in out:
        flags = d["rejected"] or ("extra: " + " ".join(d["extra_words"]) if d["extra_words"] else "")
        print(f"| {Path(d['path']).name} | {d['combined']:.3f} | {f(d['cer'])} | {f(d['wer'])} | "
              f"{f(d['lang_prob'])} | {f(d['mos'])} | {f(d['spk_sim'])} | {f(d['lead_s'])}/{f(d['trail_s'])} | "
              f"{flags} | {d['transcript']} |")
    return 0


def cmd_tune(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="fanfare-studio tune", description="Search voice settings, score every clip.")
    p.add_argument("--lang", required=True)
    p.add_argument("--sentences", required=True, help="text file, one sentence per line")
    p.add_argument("--budget", type=int, default=30, help="max clips to generate")
    p.add_argument("--voice", help="voice id (default <lang>-high)")
    p.add_argument("--reference", action="append", default=None,
                   help="reference clip(s) to try; 'builtin' = model speaker; default: the voice's own")
    p.add_argument("--backend", default="chatterbox-onnx")
    p.add_argument("--ep", default="auto")
    p.add_argument("--onnx-variant", default="q4")
    p.add_argument("--device", default="auto")
    p.add_argument("--voices-config", default="voices.toml")
    p.add_argument("--config", default="eval.toml")
    p.add_argument("--out", default="tune-out")
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args(argv)
    from dataclasses import asdict

    from .backends import create_backend
    from .quality.tune import current_weights, run_tune

    cfg = _config(a.config)
    weights, n_pairs = current_weights(cfg.get("weights"))
    print(f"weights ({'learned from ' + str(n_pairs) + ' preference pairs' if n_pairs >= 5 else 'defaults: fewer than 5 preference pairs'}): {json.dumps(weights)}")
    backend = create_backend(a.backend, ep=a.ep, onnx_variant=a.onnx_variant, device=a.device,
                             voices_config=a.voices_config, voices_dir="models/piper")
    voice = a.voice or f"{a.lang}-high"
    own = backend.settings(voice).reference
    refs = [None if r == "builtin" else str(Path(r).resolve()) for r in a.reference] if a.reference else [own]
    sentences = [s.strip() for s in Path(a.sentences).read_text(encoding="utf-8").splitlines() if s.strip()]
    ev = _evaluator(cfg)
    run = run_tune(backend, lambda path, text, lang, ref: asdict(ev.evaluate(path, text, lang, ref)),
                   lang=a.lang, sentences=sentences, budget=a.budget, out_root=Path(a.out), voice_id=voice,
                   references=refs, weights=weights, max_cer=cfg.get("filters", {}).get("max_cer", 0.15),
                   seed=a.seed)  # fmt: skip
    print(f"leaderboard: {run / 'leaderboard.md'}\nlisten (blind): {run / 'listen'}/A.wav B.wav C.wav\n"
          f"then: fanfare-studio prefer {run} --pick <A|B|C>  (or --rank B,A,C)")
    return 0


def cmd_prefer(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="fanfare-studio prefer", description="Record which blind clip you liked.")
    p.add_argument("run", help="tune-out/<lang>/<timestamp> directory")
    p.add_argument("--pick")
    p.add_argument("--rank", help="comma-separated, best first, e.g. B,A,C")
    a = p.parse_args(argv)
    from .quality.prefs import preferences_path
    from .quality.tune import current_weights, prefer

    n = prefer(Path(a.run), pick=a.pick, rank=a.rank.split(",") if a.rank else None)
    weights, total = current_weights(_config("eval.toml").get("weights"))
    print(f"stored {n} preference pair(s) in {preferences_path()} ({total} total)")
    print(f"weights now: {json.dumps(weights)}" + ("" if total >= 5 else " (defaults until 5 pairs)"))
    return 0


def cmd_voice(argv: list[str]) -> int:
    from .cli_voice import cmd_voice as run

    return run(argv)


COMMANDS = {"eval": cmd_eval, "tune": cmd_tune, "prefer": cmd_prefer, "voice": cmd_voice}


def dispatch(argv: list[str]) -> int | None:
    if argv and argv[0] in COMMANDS:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "speechbrain", "faster_whisper"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        return COMMANDS[argv[0]](argv[1:])
    return None


if __name__ == "__main__":
    sys.exit(dispatch(sys.argv[1:]) or 0)
