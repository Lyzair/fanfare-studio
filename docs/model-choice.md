# Model choice

Researched 2026-11-13 for the Sports Caller Voice Server proof of concept.

## Requirements

- The licence of the code **and** the weights permits use and redistribution of the
  generated audio (clips go into an export zip and are shared between club devices).
  Prefer MIT / Apache-2.0 / CC0; exclude non-commercial licences.
- Languages of the browser app: de, en, fr, es, it, nl, pl, tr, uk, ar. **German is mandatory.**
- Main target hardware is a **notebook**: AMD or NVIDIA dGPU, or only an iGPU (Intel Arc/Xe,
  Radeon 780M/890M), maybe an NPU (Intel Core Ultra, AMD Ryzen AI/XDNA, Snapdragon X
  Hexagon, Apple Neural Engine). Generation is "prepare at home", so it is not
  time-critical: a slow path that still gives high quality is acceptable.

## Comparison

| Model | Code licence | Weights licence | Our 10 langs covered | Size / VRAM | Speed (where we know) | Runtime portability |
|---|---|---|---|---|---|---|
| **Chatterbox Multilingual** (Resemble AI) | MIT | MIT | **9/10**: de en fr es it nl pl tr ar (no uk) — 23 languages total | 0.5B Llama backbone + S3Gen decoder; ~3–4 GB VRAM in fp32 (measured, see report) | Measured here, see README / report | PyTorch (CUDA, ROCm, MPS, CPU). **Community ONNX export exists** (`onnx-community/chatterbox-multilingual-ONNX`, MIT, fp32/fp16/q4 LM variants) → ONNX Runtime EPs (CPU, DirectML, OpenVINO, CoreML, QNN, Vitis AI, CUDA). |
| Qwen3-TTS 12Hz 1.7B (Alibaba) | Apache-2.0 | Apache-2.0 | 5/10: de en fr es it (+ zh ja ko ru pt) | ~2B params, bf16, FlashAttention recommended → realistically CUDA dGPU | not measured | PyTorch only; no official ONNX |
| Kokoro-82M (hexgrad) | Apache-2.0 | Apache-2.0 | 4/10: en fr es it (+ ja zh hi pt) — **no German** | 82M, tiny | very fast, real-time on CPU | ONNX export available (kokoro-onnx) — very portable, but no German → excluded |
| Piper voices (OHF-Voice) | engine `piper-tts` (piper1-gpl): **GPL-3.0** (old rhasspy/piper: MIT, archived) | per voice; e.g. `de_DE-thorsten-high` **CC0**, `uk_UA-tetiana-high` **Apache-2.0** | 10/10 (each language has at least one voice; quality varies, check each MODEL_CARD) | 20–110 MB per voice | faster than real time on any CPU | **ONNX native** → every ONNX Runtime EP; runs everywhere |
| XTTS-v2 (Coqui) | MPL-2.0 | **CPML (non-commercial)** | 9/10 (no uk) | ~2 GB | — | excluded: licence |
| F5-TTS | MIT | **CC-BY-NC** (pretrained weights) | en/zh base | — | — | excluded: licence |
| Fish-Speech / OpenAudio | Apache-2.0 code | **CC-BY-NC-SA** weights | many | — | — | excluded: licence |

Notes:

- Chatterbox outputs carry Resemble AI's **PerTh** imperceptible watermark (the PyTorch
  package applies it always). That does not restrict use; it marks the audio as AI-generated,
  which we consider a feature and should mention in the app.
- The Chatterbox default speaker is a single built-in English reference voice. In other
  languages it speaks with that timbre; quality in German was checked by ear in the PoC run
  (see report). A real project can ship one CC0/consented reference clip per language
  (Chatterbox is zero-shot voice cloning) — **never let users clone third-party voices
  without consent**.
- Piper's current engine package is GPL-3.0 (because it bundles espeak-ng). Running it as an
  optional, separately installed backend is fine; bundling it into a single distributed
  binary with our MIT code would make that binary GPL. Generated audio is not affected by
  the engine licence, only by the voice licence.

## Decision

**Tiered strategy**

1. **Quality tier — Chatterbox Multilingual (MIT/MIT) on PyTorch**, `--backend chatterbox
   --device auto|cuda|rocm|mps|cpu`. Best quality, 9/10 languages, licence clean.
   Verified here on **AMD ROCm (RX 7900 XTX)** and on **CPU**.
2. **Portable quality tier — the same model through ONNX Runtime**, `--backend
   chatterbox-onnx --ep auto|cuda|rocm|dml|openvino|coreml|qnn|vitisai|cpu`. Same weights,
   same voices, for notebooks whose GPU/NPU has no PyTorch path (Intel iGPU/NPU,
   Windows DirectML iGPUs, Snapdragon). Verified here on **CPU EP only**; all other EPs are
   configuration + documentation, **untested** (the 0.5B autoregressive LM with KV cache and
   dynamic shapes may not map well onto NPUs — expect fallbacks to CPU for some ops).
3. **Works-everywhere tier / Ukrainian — Piper** (`--backend piper --ep ...`), CPU
   faster-than-real-time, ONNX native. Fills the Ukrainian gap (`uk_UA-tetiana-high`,
   Apache-2.0) and is the fallback for very weak notebooks.

Because generation happens at home and the result is cached and exported, CPU is an
acceptable fallback for every notebook as long as one sentence stays well inside the
browser client's 120 s per-request timeout (measured numbers: see README and report).

## Sources

- Chatterbox model card (MIT, 23 languages, PerTh watermark): https://huggingface.co/ResembleAI/chatterbox
- Chatterbox code (MIT) and pinned deps: https://github.com/resemble-ai/chatterbox , https://raw.githubusercontent.com/resemble-ai/chatterbox/master/pyproject.toml
- Chatterbox PyPI (0.1.7, 2026-03-26): https://pypi.org/project/chatterbox-tts/
- Chatterbox ONNX export (MIT, file list fp32/fp16/q4/q4f16): https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX
- ONNX export scripts: https://github.com/VladOS95-cyber/onnx_conversion_scripts/tree/main/chatterbox
- Qwen3-TTS (Apache-2.0, 10 languages): https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- Kokoro-82M (Apache-2.0): https://huggingface.co/hexgrad/Kokoro-82M (language list in VOICES.md; no German)
- Piper engine (GPL-3.0): https://github.com/OHF-Voice/piper1-gpl ; voices: https://huggingface.co/rhasspy/piper-voices
- Thorsten high (CC0): https://huggingface.co/rhasspy/piper-voices/blob/main/de/de_DE/thorsten/high/MODEL_CARD
- Tetiana high (Apache-2.0): https://huggingface.co/rhasspy/piper-voices/blob/main/uk/uk_UA/tetiana/high/MODEL_CARD
- XTTS-v2 CPML: https://huggingface.co/coqui/XTTS-v2 ; F5-TTS weights CC-BY-NC: https://huggingface.co/SWivid/F5-TTS ; Fish-Speech weights CC-BY-NC-SA: https://huggingface.co/fishaudio
- ONNX Runtime execution providers: https://onnxruntime.ai/docs/execution-providers/
