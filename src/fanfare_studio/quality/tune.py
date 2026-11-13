"""Small parameter search scored by the evaluator, with blind A/B/C listening output."""

from __future__ import annotations

import csv
import dataclasses
import io
import itertools
import json
import logging
import time
import wave
from collections.abc import Callable
from pathlib import Path

from .prefs import DEFAULT_WEIGHTS, combined, features, fit_weights, load_pairs, rejected

log = logging.getLogger("fanfare_studio")

COARSE = {"cfg_weight": (0.3, 0.5), "exaggeration": (0.5, 0.8), "temperature": (0.6, 0.8)}
STEPS = {"cfg_weight": 0.15, "exaggeration": 0.15, "temperature": 0.1}
BOUNDS = {"cfg_weight": (0.0, 1.0), "exaggeration": (0.25, 1.5), "temperature": (0.3, 1.2)}
METRICS = ("cer", "wer", "lang_prob", "mos", "spk_sim", "lead_s", "trail_s")

Scorer = Callable[[str, str, str, "str | None"], dict]


def current_weights(config_weights: dict | None = None) -> tuple[dict, int]:
    prior = {**DEFAULT_WEIGHTS, **(config_weights or {})}
    pairs = load_pairs()
    return fit_weights(pairs, prior=prior), len(pairs)


def _key(c: dict) -> tuple:
    return tuple(sorted((k, round(v, 3) if isinstance(v, float) else v) for k, v in c.items()))


def _concat(paths: list[Path], out: Path, gap_s: float = 0.6) -> None:
    frames, params = [], None
    for p in paths:
        with wave.open(str(p)) as w:
            params = w.getparams()
            frames.append(w.readframes(w.getnframes()))
    assert params is not None
    silence = b"\x00\x00" * int(params.framerate * gap_s)
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        w.writeframes(silence.join(frames))


def run_tune(
    backend,  # type: ignore[no-untyped-def]
    score_clip: Scorer,
    *,
    lang: str,
    sentences: list[str],
    budget: int,
    out_root: Path,
    voice_id: str,
    references: list[str | None],
    weights: dict[str, float],
    max_cer: float = 0.15,
    seed: int = 1234,
) -> Path:
    run = out_root / lang / time.strftime("%Y%m%d-%H%M%S")
    (run / "clips").mkdir(parents=True)
    base = backend.settings(voice_id)
    per_cfg = len(sentences)
    if budget < per_cfg:
        raise ValueError(f"budget {budget} < {per_cfg} sentences")
    max_configs = budget // per_cfg
    grid = [dict(zip(COARSE, vals), reference=ref)
            for ref in references for vals in itertools.product(*COARSE.values())]  # fmt: skip
    n_coarse = min(len(grid), max(1, max_configs - max(1, max_configs // 4)))
    # spread the coarse budget over the grid (every k-th point) when it doesn't fit
    if n_coarse < len(grid):
        step = len(grid) / n_coarse
        grid = [grid[int(i * step)] for i in range(n_coarse)]
    results: list[dict] = []
    seen: set = set()
    rows: list[dict] = []

    def evaluate(cfg: dict) -> None:
        idx = len(results)
        s = dataclasses.replace(base, seed=seed, **cfg)
        clips, scores = [], []
        for j, text in enumerate(sentences):
            path = run / "clips" / f"c{idx:02d}-s{j}.wav"
            path.write_bytes(backend.synthesize(text, voice_id, settings=s))
            sc = score_clip(str(path), text, lang, cfg.get("reference"))
            sc["rejected"] = rejected(sc, max_cer)
            sc["combined"] = round(combined(sc, weights), 4)
            clips.append(path)
            scores.append(sc)
            rows.append({"config": idx, **{k: cfg.get(k) for k in (*COARSE, "reference")},
                         "clip": path.name, "sentence": j, **{m: sc.get(m) for m in METRICS},
                         "prefix_flag": sc.get("prefix_flag"), "transcript": sc.get("transcript", ""),
                         "combined": sc["combined"], "rejected": sc["rejected"]})  # fmt: skip
        mean = {m: (sum(x[m] for x in scores if x.get(m) is not None) /
                    max(1, sum(1 for x in scores if x.get(m) is not None))) for m in METRICS}  # fmt: skip
        reasons = [x["rejected"] for x in scores if x["rejected"]]
        results.append({"idx": idx, "config": cfg, "clips": clips, "mean": mean,
                        "features": features(mean),
                        "score": sum(x["combined"] for x in scores) / len(scores),
                        "rejected": "; ".join(reasons)})  # fmt: skip
        seen.add(_key(cfg))
        log.info("tune: config %d %s -> score %.3f %s", idx, cfg, results[-1]["score"], reasons or "")

    for cfg in grid:
        evaluate(cfg)
    while len(results) < max_configs:
        ok = [r for r in results if not r["rejected"]] or results
        best = max(ok, key=lambda r: r["score"])["config"]
        cand = []
        for k, step in STEPS.items():
            for sign in (1, -1):
                c = dict(best)
                lo, hi = BOUNDS[k]
                c[k] = round(min(hi, max(lo, c[k] + sign * step)), 3)
                if _key(c) not in seen:
                    cand.append(c)
        if not cand:
            break
        evaluate(cand[0])

    ranked = sorted(results, key=lambda r: (bool(r["rejected"]), -r["score"]))
    with (run / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    lines = [f"# Tune leaderboard — {lang}, {len(results)} configs × {per_cfg} sentences", "",
             f"Weights: {json.dumps(weights)}", "",
             "| rank | config | cfg | ex | t | reference | score | CER | lang | MOS | spk | lead/trail s | rejected |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for i, r in enumerate(ranked, 1):
        c, m = r["config"], r["mean"]
        ref = Path(c["reference"]).stem if c.get("reference") else "builtin"
        fmt = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(f"| {i} | c{r['idx']:02d} | {c['cfg_weight']} | {c['exaggeration']} | {c['temperature']} | "
                     f"{ref} | {r['score']:.3f} | {fmt(m['cer'])} | {fmt(m['lang_prob'])} | {fmt(m['mos'])} | "
                     f"{fmt(m['spk_sim'])} | {fmt(m['lead_s'])}/{fmt(m['trail_s'])} | {r['rejected'] or ''} |")
    (run / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    listen = run / "listen"
    listen.mkdir()
    mapping = {}
    for letter, r in zip("ABC", [r for r in ranked if not r["rejected"]][:3] or ranked[:3]):
        _concat(r["clips"], listen / f"{letter}.wav")
        mapping[letter] = {"config_index": r["idx"], "config": r["config"], "features": r["features"],
                           "score": r["score"]}  # fmt: skip
    # blind: mapping lives outside listen/
    (run / "mapping.json").write_text(json.dumps({"lang": lang, "voice": voice_id, "letters": mapping},
                                                 indent=2), encoding="utf-8")  # fmt: skip
    return run


def prefer(run: Path, pick: str | None = None, rank: list[str] | None = None) -> int:
    from .prefs import record_preferences

    data = json.loads((run / "mapping.json").read_text(encoding="utf-8"))
    letters = data["letters"]
    if rank:
        order = [x.strip().upper() for x in rank]
    elif pick:
        order = [pick.upper()] + [x for x in letters if x != pick.upper()]
    else:
        raise ValueError("give --pick or --rank")
    unknown = [x for x in order if x not in letters]
    if unknown:
        raise ValueError(f"unknown letter(s) {unknown}; this run has {sorted(letters)}")
    pairs = []
    if rank:
        pairs = [(letters[a]["features"], letters[b]["features"]) for a, b in itertools.combinations(order, 2)]
    else:
        pairs = [(letters[order[0]]["features"], letters[b]["features"]) for b in order[1:]]
    return record_preferences(pairs, {"lang": data["lang"], "run": str(run)})
