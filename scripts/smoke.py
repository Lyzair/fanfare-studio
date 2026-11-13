"""Call a running server like the browser app does and save the clips.

    python scripts/smoke.py --port 5123 --tag rocm de-high:de en-high:en

Each argument is VOICE_ID:SENTENCE_KEY. Prints duration and generation time per clip.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import wave
from pathlib import Path

SENTENCES = {
    "de": "Und jetzt, mit der Nummer 7: Lea! Viel Spaß und gib alles!",
    "en": "And now, wearing number 7: Lea! Have fun and give it everything!",
    "uk": "А тепер під номером 7: Лея! Успіхів і викладайся на повну!",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5123)
    p.add_argument("--tag", default="run")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("jobs", nargs="+")
    a = p.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    voices = json.load(urllib.request.urlopen(base + "/voices"))
    print(f"/voices -> {len(voices)} voices: {', '.join(v['id'] for v in voices)}")
    out = Path("samples")
    out.mkdir(exist_ok=True)
    for job in a.jobs:
        voice, key = job.split(":")
        for r in range(a.repeat):
            req = urllib.request.Request(
                base + "/synthesize",
                data=json.dumps({"text": SENTENCES[key], "voice": voice}).encode(),
                headers={"Content-Type": "application/json"},
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = resp.read()
                ctype = resp.headers["Content-Type"]
            dt = time.perf_counter() - t0
            with wave.open(io.BytesIO(body)) as w:
                dur = w.getnframes() / w.getframerate()
                rate = w.getframerate()
            path = out / f"{a.tag}-{voice}-{key}{'' if r == 0 else f'-{r}'}.wav"
            path.write_bytes(body)
            print(f"{voice:>22} {key}  {ctype}  {rate} Hz  audio {dur:5.2f}s  "
                  f"generated in {dt:6.2f}s  RTF {dt / dur:5.2f}  -> {path}")


if __name__ == "__main__":
    main()
