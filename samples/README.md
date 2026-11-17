# Samples (round 2)

Audio files are git-ignored; regenerate with `scripts/render.py` (options: `uv run python scripts/render.py --help`).
Sentences: de "Und jetzt, mit der Nummer 7: Lea! Viel Spaß und gib alles!",
en "And now, wearing number 7: Lea! Have fun and give it everything!".
Machine: Ryzen 7 5800X3D (8 cores), RX 7900 XTX (ROCm 6.2 torch wheel). Generation seconds are
in-process (no HTTP), after model load; GPU numbers after one warm-up clip. All clips use
seed 1234, rp = repetition penalty, t = temperature (0 = greedy), cfg = CFG weight,
ex = exaggeration. "thorsten" = `voices/de-thorsten.wav` (CC0), "builtin" = the model's
built-in **English** reference speaker.

## Side by side (listen to these first)

| File | Backend | Settings | Gen. s | Audio s |
|---|---|---|---|---|
| `chatterbox-onnx-q4-cpu-de.wav` | **new default**: Chatterbox ONNX, 4-bit LM, CPU EP | thorsten, cfg 0.5, ex 0.5, t 0.6, rp 1.2 | 6.0 | 3.9 |
| `chatterbox-onnx-q4-cpu-en.wav` | same | builtin, cfg 0.5, ex 0.5, t 0.6, rp 1.2 | 6.1 | 3.8 |
| `chatterbox-onnx-q4-cpu-builtin-greedy-de.wav` | ONNX q4 CPU = round-1 "best so far" settings, now with text normalisation | builtin, cfg 0, t 0 (greedy) | 5.5 | 3.9 |
| `chatterbox-onnx-q4-cpu-builtin-greedy-en.wav` | same | builtin, cfg 0, t 0 | 5.8 | 4.4 |
| `chatterbox-onnx-fp32-cpu-de.wav` | ONNX full-precision LM, CPU (prefix bug fixed) | thorsten, cfg 0.5, ex 0.5, t 0.6 | 9.2 | 3.6 |
| `chatterbox-onnx-fp32-cpu-en.wav` | same | builtin, same | 9.6 | 3.5 |
| `chatterbox-torch-fp32-cpu-de.wav` | PyTorch (experimental), CPU | thorsten, cfg 0.5, ex 0.5, t 0.6, rp **2.0** (forced, see docs) | 15.1 | 4.6 |
| `chatterbox-torch-fp32-cpu-en.wav` | same | builtin, same | 15.8 | 4.7 |
| `chatterbox-torch-fp32-cpu-pkgdefaults-de.wav` | PyTorch CPU, package defaults (= round 1) | builtin, cfg 0.5, ex 0.5, t 0.8, rp 2.0 | 13.3 | 4.3 |
| `chatterbox-torch-fp32-rocm-de.wav` | PyTorch, ROCm (fixed-seed pair with the cpu file) | thorsten, cfg 0.5, ex 0.5, t 0.6, rp 2.0 | 3.0 | 4.9 |
| `chatterbox-torch-fp32-rocm-en.wav` | same | builtin, same | 5.1 | 4.5 |
| `chatterbox-torch-fp32-rocm-pkgdefaults-de.wav` | PyTorch ROCm, package defaults (= round 1) | builtin, t 0.8, rp 2.0 | 3.3 | 3.9 |
| `chatterbox-torch-fp32-rocm-mathsdp-de.wav` | PyTorch ROCm, fused attention kernels disabled (`--torch-math-sdp`) | as `…-rocm-de` | 3.0 | 4.9 |
| `piper-high-cpu-de.wav` | Piper thorsten-high (CC0), CPU | – | 0.3–0.5 | 3.9 |
| `piper-high-cpu-en.wav` | Piper ljspeech-high (public domain), CPU | – | 0.3 | 4.0 |

Same seed on CPU vs ROCm does **not** give the same clip: the torch CPU and GPU random
generators differ (and the S3Gen decoder samples noise), so the pair is "same settings", not
"same tokens". With near-greedy decoding both devices produced the same length and text
(see docs/model-choice.md), and the math-SDP clip has the same length as the fused-kernel clip
with only tiny sample differences — ROCm numerics are not the problem.

## Accent experiments (German, ONNX q4, CPU, seed 1234, rp 1.2)

| File | Description | Gen. s | Audio s |
|---|---|---|---|
| `accent-builtin-cfg0.0-ex0.5-t0-de.wav` | built-in English speaker, greedy, no CFG (round-1 setting, normalised text) | 5.6 | 3.9 |
| `accent-builtin-cfg0.5-ex0.5-t0.8-de.wav` | built-in speaker, sampling + CFG (torch-like) | 5.3 | 3.0 |
| `accent-thorsten-cfg0.0-ex0.5-t0-de.wav` | Thorsten reference, greedy: **never stops** (32 s babble) — why greedy is not the default; now capped by text length | 70.9 | 32.0 |
| `accent-thorsten-cfg0.0-ex0.8-t0-de.wav` | same with ex 0.8: also runs away | 68.8 | 32.0 |
| `accent-thorsten-cfg0.0-ex0.5-t0.8-de.wav` | Thorsten, sampling, no CFG | 5.3 | 4.2 |
| `accent-thorsten-cfg0.5-ex0.5-t0.8-de.wav` | Thorsten, sampling + CFG 0.5 | 5.9 | 3.2 |
| `accent-thorsten-cfg0.5-ex0.5-t0.6-de.wav` | Thorsten, CFG 0.5, lower temperature (**= new default**) | 6.9 | 3.9 |
| `accent-thorsten-cfg0.5-ex0.8-t0.8-de.wav` | Thorsten, more expressive (ex 0.8) | 6.7 | 3.8 |
| `accent-thorsten-cfg0.5-ex1.1-t0.8-de.wav` | Thorsten, very expressive (ex 1.1) | 6.0 | 3.6 |

English: the built-in speaker of both paths is English (the ONNX `default_voice.wav` is a
7.4 s English clip; torch's built-in `conds.pt` speaker embedding is a different English
voice, cosine 0.65 to it), so English needs no extra reference.

Older files (`rocm-*`, `cpu-*`, `onnxcpu-*`, `onnxq4-*`, `piper-*`, `final-rocm-*`) are round 1.

## Club voice (round 3)

Stand-in "club voice": 20 s of three other CC0 Thorsten-Voice sentences (TV-2022.10-Neutral),
uploaded as WebM/Opus over `POST /references` (decoded by ffmpeg, 17.9 s of speech kept), once
as `de` and once as `en`; default backend (ONNX q4, CPU), settings cfg 0.5, ex 0.5, t 0.6.

| File | Description | Gen. s | Audio s | Eval |
|---|---|---|---|---|
| `club-de.wav` | German test sentence in the club voice | 6.6 | 3.4 | CER 0, MOS 3.06, speaker sim 0.81 |
| `club-en.wav` | English test sentence in the (German) club voice | 6.8 | 3.5 | CER 0.02, MOS 3.30, speaker sim 0.49 |
