# Fanfare Studio (proof of concept)

A small **local** text-to-speech server for **Fanfare**, the static, privacy-first browser app
in which sports clubs announce their lineups. Fanfare calls Fanfare Studio on
`http://127.0.0.1:5123` to generate high-quality voice clips on **your own computer**.
You prepare the clips at home, Fanfare stores them and puts them into its export zip, and on
game day the phone plays them **without** Fanfare Studio.

Status: proof of concept. No installer, no GUI. Licence: MIT.

## Privacy

- Binds to `127.0.0.1` only — never reachable from your network. There is no option to change that.
- The access log contains only method, path, status and duration (`POST /synthesize 200 4845ms`).
  Request bodies (player names) and query strings are never logged.
- Nothing is uploaded. Model weights are downloaded once from Hugging Face on first start;
  Hugging Face telemetry is disabled; afterwards run with `HF_HUB_OFFLINE=1` to stay fully offline.

## Backends and hardware

| `--backend` | Model | Runtime | Selection flag | Languages |
|---|---|---|---|---|
| `chatterbox-onnx` (**default, recommended**) | Chatterbox Multilingual (MIT), community ONNX export (MIT), 4-bit LM | ONNX Runtime | `--ep auto\|cuda\|rocm\|dml\|openvino\|coreml\|qnn\|vitisai\|cpu`, `--onnx-variant q4\|fp16\|fp32` | de en fr es it nl pl tr ar |
| `chatterbox` (*experimental*) | same model, PyTorch package | PyTorch | `--device auto\|cuda\|rocm\|mps\|cpu` | de en fr es it nl pl tr ar |
| `piper` | Piper voices in `--voices-dir` (licence per voice) | ONNX Runtime (CPU) | `--ep ...` | any — fills the **uk** gap |
| `mock` | sine beep | none | – | all 10 (tests / client development) |

- **Recommendation for every machine, including AMD/Linux: the default** — Chatterbox ONNX
  q4 on the CPU. It sounded best in listening tests and needs ~5–6 s per sentence on a desktop
  CPU. The PyTorch backend works on GPUs but sounded worse (see `docs/model-choice.md`).
- `--ep auto` picks CUDA → DirectML → CoreML → OpenVINO → QNN → Vitis AI → CPU, whichever is
  installed. The **ROCm EP is never picked automatically** (it produced garbled audio here);
  `--ep rocm` forces it and logs a warning. Piper always uses the CPU unless `--ep` is explicit.
  An unavailable `--ep` falls back to CPU with a warning; the active provider is logged.
- `--device auto` (PyTorch) tries CUDA/ROCm → Apple MPS → CPU; an explicit unavailable device is
  an error.

**Verified** (Linux, Ryzen 7 5800X3D, RX 7900 XTX): `chatterbox-onnx` and `piper` on CPU,
`chatterbox` on ROCm and CPU. **Untested**: CUDA, MPS, DirectML, OpenVINO, CoreML, QNN,
Vitis AI. **Broken**: ONNX Runtime ROCm EP (`onnxruntime-rocm` 1.22 on ROCm 7.2).

### Voices and settings (`voices.toml`)

Chatterbox copies the voice of a short reference clip. Without one it uses its built-in
**English** speaker — German then sounds English-accented. `voices.toml` (server config, the
HTTP protocol is unchanged) sets per-voice `reference`, `cfg_weight`, `exaggeration`,
`temperature`, `top_p`, `min_p`, `repetition_penalty`, `seed` and `max_new_tokens`, with a
`[defaults]` table. Shipped defaults: CFG 0.5, exaggeration 0.5, temperature 0.6, repetition
penalty 1.2, seed 1234 (same text → same clip); `de-high` uses `voices/de-thorsten.wav`
(Thorsten-Voice, CC0). Raise `exaggeration` (0.8–1.1) for a livelier announcer. For other
languages add a 6–10 s clean single-speaker clip from a CC0/CC-BY dataset, or a consenting club
member's own recording (a voice is personal data under GDPR: consent required, it stays on
this computer) — see `voices/README.md`. `--voices-config` selects another file.

### Speed

German test sentence (~4 s of audio), desktop Ryzen 7 5800X3D (8 cores) / RX 7900 XTX,
round-2 settings:

| Backend / device | Per sentence | 18 clips (16-player roster) |
|---|---|---|
| `chatterbox-onnx` q4, CPU (default, ~3 GB RAM) | 5.5–6 s | ~2 min |
| `chatterbox-onnx` fp32, CPU | 9–10 s | ~3 min |
| `chatterbox` (experimental), ROCm (4.7 GB VRAM) | 3–5 s (first after start ~10 s) | ~1.5 min |
| `chatterbox` (experimental), CPU | 13–16 s | ~4.5 min |
| `piper` high, CPU | 0.3–0.5 s | seconds |

A typical notebook CPU is 2–3× slower — ~12–17 s per sentence with the default, far inside
Fanfare's 120 s per-request timeout.

## Install

Requires [uv](https://docs.astral.sh/uv/); uv fetches Python 3.12 for the project.

| Platform / hardware | Command |
|---|---|
| **Any (recommended default)** — Windows, macOS, Linux; CPU | `uv sync --extra onnx --extra cpu` |
| iGPU / NPU (untested) | `uv sync --extra onnx --extra cpu`, then swap `onnxruntime` for the EP build: `onnxruntime-directml` (Windows, any DX12 GPU incl. Intel/AMD iGPUs), `onnxruntime-openvino` (Intel iGPU/NPU), `onnxruntime-qnn` (Snapdragon X), Ryzen AI SDK (Vitis AI EP). CoreML EP ships in the macOS wheel. |
| Experimental PyTorch, AMD GPU (Linux ROCm) | `uv sync --extra chatterbox --extra rocm` |
| Experimental PyTorch, NVIDIA GPU | `uv sync --extra chatterbox --extra cuda` |
| Experimental PyTorch, macOS (MPS) / CPU | `uv sync --extra chatterbox --extra cpu` |
| Piper voices | `uv sync --extra piper`; put `*.onnx` + `*.onnx.json` from [piper-voices](https://huggingface.co/rhasspy/piper-voices) into `models/piper/` |

Notes:
- Always pick one torch flavour (`--extra cpu|cuda|rocm`): the required PerTh watermarker
  (`resemble-perth`) imports PyTorch, so torch (~200 MB CPU wheel) is part of every install, even
  for the ONNX backend. Without an extra uv would pick the huge PyPI CUDA build.
- ROCm (experimental path): the PyTorch ROCm wheel brings its own ROCm runtime; you need the
  amdgpu kernel driver and access to `/dev/kfd` and `/dev/dri/renderD*`. gfx1100 works.
- First start downloads ~1.6 GB (`chatterbox-onnx` q4) or ~3 GB (`chatterbox`) into the
  Hugging Face cache.

## Run

```sh
uv run fanfare-studio                                   # chatterbox-onnx q4, --ep auto (CPU here), port 5123
uv run fanfare-studio --ep openvino                     # try an execution provider (untested)
uv run fanfare-studio --backend chatterbox --device rocm   # experimental PyTorch path
uv run fanfare-studio --backend piper --voices-dir models/piper
uv run fanfare-studio --backend mock --port 5999
uv run fanfare-studio --allow-origin https://<user>.github.io   # allow a Fanfare site for this run
uv run fanfare-studio --version
```

Quick check against a running server: `uv run python scripts/smoke.py de-high:de en-high:en`.
Comparisons with explicit settings: `uv run python scripts/render.py --help`.

## Run with Docker

Default image: Chatterbox 4-bit ONNX on CPU (the recommended path, best quality in our listening
tests). Needs Docker with Compose; the first start downloads ~3 GB of models into the `models`
volume.

```sh
docker compose up -d --build            # http://127.0.0.1:5123
docker compose logs -f fanfare-studio   # "Fanfare Studio ready ..."
docker compose ps                       # healthy once /voices answers
```

- **Published on the host's `127.0.0.1:5123` only** (`compose.yaml`). Inside the container the
  server binds `0.0.0.0` (`FANFARE_HOST` / `--host`) so Docker can forward to it; never publish
  it as `5123:5123`, that would expose it to your network.
- **Allowed websites** (the origin rules are unchanged; localhost origins always work): there is
  no interactive prompt in a container. Either set `FANFARE_ALLOWED_ORIGINS` (comma-separated,
  exact scheme + host + port, no wildcards):
  `FANFARE_ALLOWED_ORIGINS=https://you.github.io docker compose up -d`, or persist one with
  `docker compose exec fanfare-studio fanfare-studio allow https://you.github.io` (stored in the
  `config` volume; restart the container to apply).
- **Volumes**: `models` (downloads, `/models`), `references` (club voices, `/data`; add them over
  `POST /references`, or `docker compose exec fanfare-studio fanfare-studio voice add ...` with the
  consent flag), `config` (`/config`, allowed origins). `docker compose down -v` deletes them,
  including club voices.
- **Hardening**: non-root user, `read_only: true` (+ tmpfs `/tmp`), `cap_drop: [ALL]`,
  `no-new-privileges`. `HEALTHCHECK` does `GET /voices` without an `Origin` header.
- **GPU variants (experimental)**: `docker compose -f compose.yaml -f compose.rocm.yaml up -d --build`
  (AMD, ROCm PyTorch, `/dev/kfd` + `/dev/dri`, torch Chatterbox backend; the image is several GB)
  and `compose.cuda.yaml` (NVIDIA, `gpus: all`; **untested**, no NVIDIA hardware available). ROCm
  status: see the Round 6 note in the PoC report. These are not recommended: CPU ONNX 4-bit sounded
  best.
- **With the Fanfare web app**: the web repo's `compose.studio.yaml` runs the app and this server
  together (the app's origin is passed to the server through `FANFARE_ALLOWED_ORIGINS`).
- Smoke test: `scripts/docker-smoke.sh` (build, start, health, origin 200/403, German clip, `down -v`).
  Podman works too (`DOCKER_HOST=unix:///run/user/$UID/podman/podman.sock`).

## Automatic quality checks (dev tooling)

`fanfare-studio eval` scores clips (Whisper round-trip CER/WER, Silero VAD leading/trailing
silence and garbage-prefix flag, Whisper language probability, UTMOSv2 MOS, ECAPA speaker
similarity, optional *local* LLM judge). `fanfare-studio tune` searches voice settings within a
clip budget and writes a leaderboard plus three blind clips `listen/A.wav B.wav C.wav`;
`fanfare-studio prefer <run> --pick B` records your choice and future `tune` runs learn metric
weights from your picks. Install with `uv sync --extra onnx --extra eval --extra cpu`
(~2 GB of evaluation models on first use). Details: `docs/eval.md`.

## Protocol

The contract Fanfare is built against:

- `GET /voices` → `200 application/json`
  `[{ "id": "de-high", "lang": "de", "label": "German (high quality)" }]`,
  `lang` ∈ `de en fr es it nl pl tr uk ar`.
- `POST /synthesize`, body `application/json` `{ "text": "...", "voice": "de-high" }` →
  `200 audio/wav` (mono, 16-bit PCM).
  Errors carry a `text/plain` message: empty/missing text, text > 1000 characters or malformed
  JSON → `400`; unknown voice → `404`; body > 64 KiB → `413`; model failure → `500`.
- **CORS / allowed websites** (see *Security*): a request with an `Origin` header is answered
  only if that origin is allowed; the response then carries `Access-Control-Allow-Origin:
  <that origin>` and `Vary: Origin` (never `*`). Unknown origins get `403 text/plain` without
  CORS headers. Requests without `Origin` (curl, scripts) are allowed. Preflight `OPTIONS` →
  `204` with `Access-Control-Allow-Methods: GET, POST, DELETE`,
  `Access-Control-Allow-Headers: Content-Type, X-Fanfare`,
  `Access-Control-Allow-Private-Network: true` (Private Network Access).
- Fanfare sends `X-Fanfare: 1` on every request; it is **required** on `POST /references` and
  `DELETE /references/{id}` (missing → `400`).
- Synthesis is serialized (one at a time): the GPU and the model are shared.

**Addition — club voices** (with the `chatterbox-onnx`, `chatterbox` and `mock` backends):

- `POST /references`, `multipart/form-data` with fields `audio` (wav/webm/ogg/mp3/m4a/flac,
  ≤ 10 MB, ≤ 60 s), `name` (≤ 40 characters), `lang` (`de en fr es it nl pl tr ar`) and
  `consent` (must be `true`) → `201 application/json`
  `{ "id": "club-1a2b3c4d", "lang": "de", "label": "…", "warnings": [] }`.
  Errors (`text/plain`): missing consent, bad field, undecodable audio or < 5 s of speech →
  `400`; upload > 10 MB → `413`. Warnings (still 201): very quiet, clipped.
- `DELETE /references/{id}` → `204` (recording and metadata deleted), unknown → `404`.
- Club voices appear in `GET /voices` with ids `club-…` and are used like any other voice.
- Audio is never returned; names and audio are never logged
  (ids are random, so they reveal nothing).

## Security

- **Local only**: the server listens on `127.0.0.1`; nothing on your network can reach it.
- **But every website open in your browser can send requests to `127.0.0.1`.** Without
  protection any page could list voices, burn your CPU/GPU with synthesis, or add/delete club
  voices. So browser requests (they always carry an `Origin` header) must come from an
  **allowed website**:
  - built in: `http://localhost:<any port>` and `http://127.0.0.1:<any port>` (local Fanfare
    development and `vite preview`);
  - your list, e.g. the Fanfare site you use:
    `uv run fanfare-studio allow https://<user>.github.io` (exact scheme + host + port; no
    wildcards), `fanfare-studio allowed` to list, `fanfare-studio disallow <origin>` to remove.
    Stored in the user config folder (Linux `~/.config/fanfare-studio/allowed-origins.toml`);
  - `--allow-origin URL` (repeatable) for one run.
- **First-use approval**: when an unknown website calls the server and the server runs in a
  terminal, it asks once: `Website https://… wants to use Fanfare Studio. Allow? [y/N]`. That
  first request is refused (403); after `y` the website is saved and works from then on.
  Nothing is ever allowed automatically.
- Unknown websites get `403` **without** CORS headers, so the browser does not let the page read
  the answer, and their preflights fail.
- **Write requests need `X-Fanfare: 1`.** A `multipart/form-data` upload is a CORS "simple
  request": a browser sends it *without* asking the server first (no preflight), so a foreign
  page could fire an upload even though it could not read the reply. Requiring a custom header
  makes the browser send a preflight first, which unknown websites fail — the upload is never
  sent. (`POST /synthesize` uses `application/json`, which already forces a preflight.)
- Requests without `Origin` (curl, your own scripts) are allowed: they don't come from a web
  page, and anything that can run them on your computer could also read your files.
- Logs contain method, path, status and duration only — never texts, names or audio.

## Your club's own voice

Fanfare can announce with the voice of your own stadium announcer (or coach, or club
volunteer). Chatterbox clones a voice from a short recording.

**Recording tips:** a quiet room, phone or headset microphone 20–30 cm from the mouth,
**15–20 seconds** of speech in the announcer style you want ("Herzlich willkommen zum
Heimspiel! Mit der Nummer 7 …"), in the language of the announcements, no music or crowd
noise. Less than 5 s of speech is rejected; very quiet or clipped recordings get a warning.

**Add it** — from Fanfare (upload over `POST /references`) or on the command line:

```sh
uv run fanfare-studio voice add --name "Stadionsprecher Max" --lang de --file max.wav --consent
uv run fanfare-studio voice list
uv run fanfare-studio voice delete club-1a2b3c4d
```

Processing: decoded with ffmpeg if installed (wav, webm, ogg, mp3, m4a, flac; otherwise
soundfile: wav/flac/ogg only), mono, 24 kHz, leading/trailing silence trimmed,
loudness-normalised, capped at 30 s. Stored per user, never in the repository:
`~/.local/share/fanfare-studio/references/<id>/` (Linux; the platform's user-data folder on
Windows/macOS) with `reference.wav` and `meta.json` (id, label, language, date, consent).
Club voices use the tuned settings of the built-in voice of the same language (CFG 0.5,
exaggeration 0.5, temperature 0.6).

**Consent and privacy (GDPR):** a voice is personal data. Only add a recording if the person
agreed that it is stored on this computer and used to generate announcements
(`--consent` / `consent=true` is required and recorded). The recording **stays on this
computer**; only the generated announcements leave it, through your Fanfare export. Delete
the voice whenever the person asks (`voice delete` or `DELETE /references/{id}` removes the
files). **Never imitate famous people or anyone who has not agreed.** Every clip carries
Resemble's inaudible PerTh watermark that marks it as AI-generated (see below).

## Workflow: prepare at home → export → phone

1. At home, start Fanfare Studio (`uv run fanfare-studio`).
2. In Fanfare, step 4 → "Local voice server (advanced)" → Connect (`http://127.0.0.1:5123`).
3. Pick a voice from "On this computer" (`local:de-high`, …).
4. **Prepare voices** — Fanfare requests one clip per player/phrase and caches it.
   Listen to them: generative TTS occasionally adds a stray syllable; regenerate those.
5. **Export** — the zip contains the clips.
6. Import the zip on the phone. On game day the phone plays the cached clips; Fanfare Studio is
   not needed.

## Why the standard-library HTTP server

Two routes, JSON in, bytes out. `http.server.ThreadingHTTPServer` covers that with zero
dependencies, keeps the core installable without the model stack (tests need only pytest),
and leaves no framework defaults to get wrong (`0.0.0.0` binds, verbose access logs). No
HTTP/2 or async — irrelevant for one local user with serialized GPU work.

## Watermark (always on)

`resemble-perth` is a **required** dependency. Every clip from every backend is watermarked with
PerTh: `chatterbox` (torch) via the library itself, `chatterbox-onnx`, `piper` (Ukrainian etc.) and
`mock` through a wrapper applied to the finished WAV (PerTh works on arbitrary speech audio; we
verified detection on a Piper clip). If the watermarker cannot be loaded the server **refuses to
start** with an error; there is no unmarked mode. Watermarking adds a fraction of a second per
clip. Removing it (or using it to hide the AI origin) is against the spirit of the model licence.

## Tests

```sh
uv run pytest
```

Uses the `mock` backend: no GPU, no model download. Covers every protocol rule (status codes,
content types, CORS + PNA preflight, 127.0.0.1 binding, request text absent from logs,
serialized synthesis) and the device / execution-provider selection logic.

## Licences

- Fanfare Studio: **MIT** (`LICENSE`).
- Chatterbox Multilingual code and weights: **MIT** (Resemble AI); ONNX export
  (`onnx-community/chatterbox-multilingual-ONNX`): **MIT**. Generated audio may be used and
  shared. Clips carry Resemble's inaudible **PerTh watermark** marking them as AI-generated.
- Piper engine `piper-tts` is **GPL-3.0** (optional extra, not bundled). Piper **voices have
  individual licences** — check each `MODEL_CARD` before sharing clips:
  `de_DE-thorsten-high` CC0, `en_US-ljspeech-high` public domain, `uk_UA-tetiana-high`
  Apache-2.0; but `en_US-ryan-high` is CC BY-NC-SA and `en_US-lessac-*` research-only.
- Never clone someone's voice without their consent.

Model comparison and sources: `docs/model-choice.md`.
