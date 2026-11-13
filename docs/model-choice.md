# Model choice

Researched 2026-11-13 for the Fanfare Studio proof of concept (local TTS companion of the
Fanfare browser app).

## Requirements

- The licence of the code **and** the weights permits use and redistribution of the
  generated audio (clips go into an export zip and are shared between club devices).
  Prefer MIT / Apache-2.0 / CC0; exclude non-commercial licences.
- Languages of Fanfare: de, en, fr, es, it, nl, pl, tr, uk, ar. **German is mandatory.**
- Main target hardware is a **notebook**: AMD or NVIDIA dGPU, or only an iGPU (Intel Arc/Xe,
  Radeon 780M/890M), maybe an NPU (Intel Core Ultra, AMD Ryzen AI/XDNA, Snapdragon X
  Hexagon, Apple Neural Engine). Generation is "prepare at home", so it is not
  time-critical: a slow path that still gives high quality is acceptable.

## Comparison

| Model | Code licence | Weights licence | Our 10 langs covered | Size / VRAM | Speed (where we know) | Runtime portability |
|---|---|---|---|---|---|---|
| **Chatterbox Multilingual** (Resemble AI) | MIT | MIT | **9/10**: de en fr es it nl pl tr ar (no uk) — 23 languages total | 0.5B Llama backbone + S3Gen decoder; **4.7 GB VRAM** measured (ROCm, fp32); ONNX q4 ~3 GB RAM | ROCm 7900 XTX ~5 s/sentence; CPU (5800X3D) torch ~13 s, ONNX q4 ~5.5 s | PyTorch (CUDA, ROCm, MPS, CPU). **Community ONNX export exists** (`onnx-community/chatterbox-multilingual-ONNX`, MIT, fp32/fp16/q4 LM variants) → ONNX Runtime EPs (CPU, DirectML, OpenVINO, CoreML, QNN, Vitis AI, CUDA). |
| Qwen3-TTS 12Hz 1.7B (Alibaba) | Apache-2.0 | Apache-2.0 | 5/10: de en fr es it (+ zh ja ko ru pt) | ~2B params, bf16, FlashAttention recommended → realistically CUDA dGPU | not measured | PyTorch only; no official ONNX |
| Kokoro-82M (hexgrad) | Apache-2.0 | Apache-2.0 | 4/10: en fr es it (+ ja zh hi pt) — **no German** | 82M, tiny | very fast, real-time on CPU | ONNX export available (kokoro-onnx) — very portable, but no German → excluded |
| Piper voices (OHF-Voice) | engine `piper-tts` (piper1-gpl): **GPL-3.0** (old rhasspy/piper: MIT, archived) | per voice; e.g. `de_DE-thorsten-high` **CC0**, `uk_UA-tetiana-high` **Apache-2.0** | 10/10 (each language has at least one voice; quality varies, check each MODEL_CARD) | 20–110 MB per voice | faster than real time on any CPU | **ONNX native** → every ONNX Runtime EP; runs everywhere |
| XTTS-v2 (Coqui) | MPL-2.0 | **CPML (non-commercial)** | 9/10 (no uk) | ~2 GB | — | excluded: licence |
| F5-TTS | MIT | **CC-BY-NC** (pretrained weights) | en/zh base | — | — | excluded: licence |
| Fish-Speech / OpenAudio | Apache-2.0 code | **CC-BY-NC-SA** weights | many | — | — | excluded: licence |

Notes:

- Chatterbox outputs carry Resemble AI's **PerTh** imperceptible watermark (applied to every clip
  of every backend, Piper included; `resemble-perth` is a required dependency). That does not restrict use; it marks the audio as AI-generated,
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

## Decision (revised after listening tests, round 2)

**One recommended backend: Chatterbox Multilingual through ONNX Runtime with the 4-bit
language model on the CPU** (`--backend chatterbox-onnx`, default; `--onnx-variant q4`,
default; `--ep auto` → CPU here). In listening tests it sounded best, it is the fastest CPU
path (~5–6 s per sentence on a desktop 8-core CPU, ~3 GB RAM), and it runs on every notebook.
That includes **AMD/Linux**: ONNX q4 on the CPU is the recommendation there too — ~5.5 s per
sentence is fine for prepare-at-home.

1. **Default — `chatterbox-onnx` q4, CPU EP.** Verified. Other EPs (DirectML, OpenVINO,
   CoreML, QNN, Vitis AI, CUDA) are configuration + documentation, **untested**; the 0.5B
   autoregressive LM with KV cache and dynamic shapes may not map well onto NPUs, and q4
   (MatMulNBits) is CPU/CUDA/DML-oriented, so NPUs likely need an fp16/QDQ export. The
   **ROCm EP is denylisted** for `--ep auto` (garbled audio, see below); an explicit `--ep rocm`
   still works but logs a warning.
2. **Experimental — `chatterbox` (PyTorch)**, `--device auto|cuda|rocm|mps|cpu`. Works on ROCm
   (~3–5 s/sentence) and CPU (~13–16 s), but sounds worse than ONNX q4 (user listening test, CPU
   and ROCm alike). Kept for comparison and for future GPU work.
3. **`piper`** — Ukrainian (`uk_UA-tetiana-high`, Apache-2.0) and a near-instant fallback.
   Always CPU (`--ep auto` = CPU for Piper).

CPU is an acceptable path for every notebook: a notebook CPU 2–3× slower than the test machine
needs ~12–17 s per sentence with ONNX q4 — far below 60 s and the client's 120 s timeout. A
16-player roster (~18 clips) takes ~2 min on the test CPU, ~4–6 min on a slow notebook.

## Round 2 investigation: why ONNX and PyTorch sounded different

Measured on the same machine; listening verdicts from the user, intelligibility checked with
Whisper-small (scratch tool, not shipped).

| Aspect | PyTorch package (`chatterbox-tts` 0.1.7) | ONNX reference script (model card) | Effect / fix |
|---|---|---|---|
| Text preparation | `punc_norm` (":"→",", capitalise, final ".") then **lower-case + NFKD** | raw text, **upper-case kept** | Upper-case graphemes (`U`, `N`, `L`, `V`, `S`) are tokens the model never saw in training. **This caused the fp32 garbage prefix** ("Mein Name ist Die …", deterministic). Fixed: ONNX backend now normalises like the package; the fp32 prefix is gone. |
| Decoding | sampling: temperature 0.8, min_p 0.05, top_p 1.0, **CFG 0.5** (2-row batch, text embeddings zeroed in the unconditional row) | **greedy** argmax, no CFG | ONNX backend now supports both (+ seed). Greedy is fragile: with a cloned voice it never emitted stop (32 s babble) → default is sampling t 0.6 + CFG 0.5, and output length is capped at 3 tokens/character + 50. |
| Repetition penalty | **2.0** (mtl default; T3 default is 1.2) | 1.2 | 2.0 penalises every speech token already used — speech naturally repeats tokens, so this is a likely reason the torch path sounds strained. But the package also runs an `AlignmentStreamAnalyzer` that **forces end-of-speech after repeated tokens**: with 1.2 the torch clips were cut after 0.8–1.4 s. So torch must stay at ≥ 2.0 (enforced in the backend); ONNX uses 1.2. |
| Speaker | built-in `conds.pt` (English speaker) | `default_voice.wav` from the ONNX repo — a **different English speaker** (speaker-embedding cosine 0.65; Whisper: an English gaming sentence) | Both made German sound English-accented → German now uses a native CC0 reference (`voices/de-thorsten.wav`). |
| Max tokens | 1000 | 256 (script) / 800 (round 1) | length cap above |
| Checkpoint | `t3_mtl23ls_v2.safetensors` | export of the multilingual T3 (version not stated) | unknown; may differ |
| Watermark | always PerTh | optional | none audible |

**ROCm-specific numerics: not the cause.** Torch 2.6.0+rocm6.2.4 on gfx1100 runs in fp32
throughout (no autocast; T3 uses `sdpa`, S3Gen fp32); the only warning is "hipBLASLt
unsupported → hipblas". With near-greedy decoding CPU and ROCm produce the same length and
transcript ("Und jetzt mit der Nummer 7!" — truncated by the analyzer at t≈0, on both devices).
Disabling fused attention (`--torch-math-sdp`: flash + mem-efficient off, math on) gives a
clip of identical length with only tiny sample differences. Same-seed waveforms still differ
between CPU and GPU because the RNG streams differ and S3Gen's flow-matching decoder samples
noise. Note: Chatterbox's perceiver re-enables all SDP kernels in its own context manager, so
the diagnostic flag does not cover that block. Conclusion: the torch path sounds worse because
of its decoding heuristics (rp 2.0 + forced EOS), not because of ROCm.

**ONNX Runtime ROCm EP** (`onnxruntime-rocm` 1.22.2, system ROCm 7.2): sessions load, output
is garbled for both Chatterbox and Piper → denylisted for `--ep auto`.

### Accent

Root cause: Chatterbox clones the reference voice, and without one it uses its built-in English
speaker, so German comes out with an English accent. The language token itself is correct
(`[de]` is the first text token; logged at DEBUG). Fix: a native German reference,
`voices/de-thorsten.wav` — 6.6 s from Thorsten-Voice **TV-2022.10-Neutral** (CC0-1.0),
downloaded from the official dataset https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full
(also Zenodo DOI 10.5281/zenodo.7265581); details in `voices/README.md`. Parameter sweep
(fixed seed; files `samples/accent-*.wav`): greedy runs away with the Thorsten reference;
sampling with CFG 0.5 is clean; t 0.6 chosen as default (clean transcript incl. "Lea");
exaggeration 0.8/1.1 available for a livelier announcer style. Final choice needs ears.

## Sources

- Chatterbox model card (MIT, 23 languages, PerTh watermark): https://huggingface.co/ResembleAI/chatterbox
- Chatterbox code (MIT) and pinned deps: https://github.com/resemble-ai/chatterbox , https://raw.githubusercontent.com/resemble-ai/chatterbox/master/pyproject.toml
- Chatterbox PyPI (0.1.7, 2026-03-26): https://pypi.org/project/chatterbox-tts/
- Chatterbox ONNX export (MIT, file list fp32/fp16/q4/q4f16): https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX
- ONNX export scripts: https://github.com/VladOS95-cyber/onnx_conversion_scripts/tree/main/chatterbox
- Qwen3-TTS (Apache-2.0, 10 languages): https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- Kokoro-82M (Apache-2.0): https://huggingface.co/hexgrad/Kokoro-82M (language list in VOICES.md; no German)
- Piper engine (GPL-3.0): https://github.com/OHF-Voice/piper1-gpl ; voices: https://huggingface.co/rhasspy/piper-voices
- Thorsten-Voice dataset TV-44kHz-Full (CC0): https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full
- Thorsten high (CC0): https://huggingface.co/rhasspy/piper-voices/blob/main/de/de_DE/thorsten/high/MODEL_CARD
- Tetiana high (Apache-2.0): https://huggingface.co/rhasspy/piper-voices/blob/main/uk/uk_UA/tetiana/high/MODEL_CARD
- XTTS-v2 CPML: https://huggingface.co/coqui/XTTS-v2 ; F5-TTS weights CC-BY-NC: https://huggingface.co/SWivid/F5-TTS ; Fish-Speech weights CC-BY-NC-SA: https://huggingface.co/fishaudio
- ONNX Runtime execution providers: https://onnxruntime.ai/docs/execution-providers/
