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
| **Any (recommended default)** — Windows, macOS, Linux; CPU | `uv sync --extra onnx` |
| iGPU / NPU (untested) | `uv sync --extra onnx`, then swap `onnxruntime` for the EP build: `onnxruntime-directml` (Windows, any DX12 GPU incl. Intel/AMD iGPUs), `onnxruntime-openvino` (Intel iGPU/NPU), `onnxruntime-qnn` (Snapdragon X), Ryzen AI SDK (Vitis AI EP). CoreML EP ships in the macOS wheel. |
| Experimental PyTorch, AMD GPU (Linux ROCm) | `uv sync --extra chatterbox --extra rocm` |
| Experimental PyTorch, NVIDIA GPU | `uv sync --extra chatterbox --extra cuda` |
| Experimental PyTorch, macOS (MPS) / CPU | `uv sync --extra chatterbox --extra cpu` |
| Piper voices | `uv sync --extra piper`; put `*.onnx` + `*.onnx.json` from [piper-voices](https://huggingface.co/rhasspy/piper-voices) into `models/piper/` |

Notes:
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
uv run fanfare-studio --version
```

Quick check against a running server: `uv run python scripts/smoke.py de-high:de en-high:en`.
Comparisons with explicit settings: `uv run python scripts/render.py --help`.

## Protocol

The contract Fanfare is built against:

- `GET /voices` → `200 application/json`
  `[{ "id": "de-high", "lang": "de", "label": "German (high quality)" }]`,
  `lang` ∈ `de en fr es it nl pl tr uk ar`.
- `POST /synthesize`, body `application/json` `{ "text": "...", "voice": "de-high" }` →
  `200 audio/wav` (mono, 16-bit PCM).
  Errors carry a `text/plain` message: empty/missing text, text > 1000 characters or malformed
  JSON → `400`; unknown voice → `404`; body > 64 KiB → `413`; model failure → `500`.
- Every response has `Access-Control-Allow-Origin: *`. Preflight `OPTIONS` → `204` with
  `Access-Control-Allow-Methods: GET, POST`, `Access-Control-Allow-Headers: Content-Type`,
  `Access-Control-Allow-Private-Network: true` (Private Network Access: an HTTPS page calling
  `127.0.0.1`).
- Synthesis is serialized (one at a time): the GPU and the model are shared.

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
  shared. Clips carry Resemble's inaudible **PerTh watermark** marking them as AI-generated
  (always with `chatterbox`; with `chatterbox-onnx` only if `resemble-perth` is installed).
- Piper engine `piper-tts` is **GPL-3.0** (optional extra, not bundled). Piper **voices have
  individual licences** — check each `MODEL_CARD` before sharing clips:
  `de_DE-thorsten-high` CC0, `en_US-ljspeech-high` public domain, `uk_UA-tetiana-high`
  Apache-2.0; but `en_US-ryan-high` is CC BY-NC-SA and `en_US-lessac-*` research-only.
- Never clone someone's voice without their consent.

Model comparison and sources: `docs/model-choice.md`.
