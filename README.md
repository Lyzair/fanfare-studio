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
| `chatterbox` (default) | Chatterbox Multilingual (MIT) | PyTorch | `--device auto\|cuda\|rocm\|mps\|cpu` | de en fr es it nl pl tr ar |
| `chatterbox-onnx` | same model, community ONNX export (MIT) | ONNX Runtime | `--ep auto\|cuda\|rocm\|dml\|openvino\|coreml\|qnn\|vitisai\|cpu`, `--onnx-variant q4\|fp16\|fp32` | de en fr es it nl pl tr ar |
| `piper` | Piper voices in `--voices-dir` (licence per voice) | ONNX Runtime | `--ep ...` | any — fills the **uk** gap |
| `mock` | sine beep | none | – | all 10 (tests / client development) |

- `--device auto` tries CUDA/ROCm (both appear as `torch.cuda`) → Apple MPS → CPU. An explicit
  `--device` that is not available is an error, never a silent fallback.
- `--ep` tries the requested ONNX Runtime execution provider and falls back to CPU with a
  warning; the active provider is logged at start-up.

**Verified** in the PoC (Linux, Ryzen 7 5800X3D, Radeon RX 7900 XTX):
`chatterbox` on ROCm and on CPU; `chatterbox-onnx` and `piper` on the CPU execution provider.
**Untested** (wired up and documented only): CUDA, MPS, DirectML, OpenVINO, CoreML, QNN, Vitis AI.
**Known broken**: ONNX Runtime's ROCm EP (`onnxruntime-rocm` 1.22 with a ROCm 7.2 system) —
sessions load but produce garbage audio. Use `chatterbox --device rocm` on AMD GPUs.

### Speed

Sentence "Und jetzt, mit der Nummer 7: Lea! Viel Spaß und gib alles!" (~4–5 s of audio),
desktop Ryzen 7 5800X3D (8 cores) / RX 7900 XTX:

| Backend / device | First request | Following | 18 clips (16-player roster) |
|---|---|---|---|
| `chatterbox`, ROCm (4.7 GB VRAM) | 20 s (GPU warm-up) | 4.7–6.2 s | ~1.8 min |
| `chatterbox`, CPU | 13 s | 12–14 s | ~4 min |
| `chatterbox-onnx` q4, CPU (~3 GB RAM) | 5.3 s | 5.3–5.7 s | ~1.7 min |
| `chatterbox-onnx` fp32, CPU | 13 s | 12–13 s | ~4 min |
| `piper` high, CPU | 0.5 s | 0.3–0.4 s | ~6 s |

Expect a typical notebook CPU to be 2–3× slower — still well inside Fanfare's 120 s
per-request timeout. "Prepare at home" is not time-critical, so CPU works on every notebook.

## Install

Requires [uv](https://docs.astral.sh/uv/). uv fetches Python 3.12 for the project
(PyTorch 2.6, pinned by Chatterbox, has no wheels for Python 3.14).

| Platform / hardware | Command |
|---|---|
| Linux + AMD GPU (ROCm) | `uv sync --extra chatterbox --extra rocm` |
| Linux / Windows + NVIDIA GPU | `uv sync --extra chatterbox --extra cuda` |
| macOS Apple Silicon (MPS) | `uv sync --extra chatterbox --extra cpu` (PyPI torch includes MPS) |
| Any, CPU only | `uv sync --extra chatterbox --extra cpu` **or** `uv sync --extra onnx` (lighter, faster on CPU) |
| iGPU / NPU (portable ONNX path, untested) | `uv sync --extra onnx`, then replace `onnxruntime` with the EP build: `onnxruntime-directml` (Windows, any DX12 GPU incl. Intel/AMD iGPUs), `onnxruntime-openvino` (Intel iGPU/NPU), `onnxruntime-qnn` (Snapdragon X), Ryzen AI SDK (Vitis AI EP). CoreML EP is in the default macOS wheel. |
| Piper voices | `uv sync --extra piper`; put `*.onnx` + `*.onnx.json` from [piper-voices](https://huggingface.co/rhasspy/piper-voices) into `models/piper/` |

Notes:
- ROCm: the PyTorch ROCm wheel brings its own ROCm runtime; you need the amdgpu kernel driver
  and access to `/dev/kfd` and `/dev/dri/renderD*`. RDNA3 (gfx1100) works out of the box.
- Windows + AMD GPU: no PyTorch ROCm path here → `chatterbox-onnx --ep dml` (untested) or CPU.
- First start downloads ~3 GB (`chatterbox`) or ~1.6 GB (`chatterbox-onnx` q4) into the
  Hugging Face cache.

## Run

```sh
uv run fanfare-studio                                   # chatterbox, --device auto, port 5123
uv run fanfare-studio --device cpu                      # force CPU
uv run fanfare-studio --backend chatterbox-onnx         # ONNX q4, --ep auto
uv run fanfare-studio --backend chatterbox-onnx --ep openvino
uv run fanfare-studio --backend piper --voices-dir models/piper
uv run fanfare-studio --backend mock --port 5999
uv run fanfare-studio --version
```

Quick check against a running server: `uv run python scripts/smoke.py de-high:de en-high:en`
(saves clips to `samples/`, prints audio length and generation time).

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
