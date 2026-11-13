# Automatic quality evaluation (`fanfare-studio eval | tune | prefer`)

Dev / power-user tooling so a person only listens to the best few candidates. Everything runs
locally; models are downloaded once into the user's caches.

```sh
uv sync --extra onnx --extra eval --extra cpu        # or --extra rocm / --extra cuda instead of cpu
uv run fanfare-studio eval samples/*.wav --text "Und jetzt, mit der Nummer 7: Lea!" --lang de \
    --reference voices/de-thorsten.wav [--json]
uv run fanfare-studio tune --lang de --sentences tests/data/sentences-de.txt --budget 30
uv run fanfare-studio prefer tune-out/de/<timestamp> --pick B      # or --rank B,A,C
```

Configuration: `eval.toml` (weights, filters, Whisper size, optional judge).

## Metrics

| Metric | How | Catches | Limits |
|---|---|---|---|
| **CER / WER**, extra & missing words | faster-whisper (`small`, CPU int8) transcript vs. the intended text, both lower-cased, punctuation stripped; edit distance; `difflib` word diff | wrong / extra / missing words, garbage prefixes and tails ("Mein Name Die …", "… And, Jonik."), truncation | Whisper fixes up mumbled speech (language-model prior) and normalises names ("Lea" → "Leah", "leer"); digits vs words ("7" vs "sieben") count as errors |
| **Leading / trailing non-speech**, **prefix flag** | Silero VAD (bundled in faster-whisper) speech segments; flag if > 0.4 s before the first speech, if extra words precede the first matched word, or a speech segment ends before the first transcript word | silence or babble before the announcement | VAD can count breaths or noise as speech; music beds would confuse it |
| **Language probability** (accent proxy) | Whisper language detection, P(target language) | clips in the wrong language, strong foreign accent | **weak**: an English-accented German clip still gets P(de) ≈ 1.0 in our tests. A real accent classifier (e.g. SpeechBrain VoxLingua107, Apache-2.0) would be the next step |
| **MOS** (naturalness) | UTMOSv2 (`fusion_stage3`, fold 0) reference-free MOS prediction, 1–5 | robotic / noisy / artefact-laden audio | trained on English-heavy MOS data; differences of < 0.2 are noise; scores TTS around 3 |
| **Speaker similarity** | SpeechBrain ECAPA (`spkrec-ecapa-voxceleb`) cosine to `--reference` | whether a cloned voice actually sounds like the reference (built-in speaker vs Thorsten: 0.27 vs 0.75) | says nothing about accent or quality |
| **LLM judge** (optional) | local **Unsloth** instance (OpenAI-compatible); fixed rubric: native / exact / announcer / quality 1-5, JSON; see "LLM judge with Unsloth" below | holistic "does this sound like a native announcer" | disabled by default; only `localhost`/`127.0.0.1`/`[::1]` URLs ending in `/v1`, so audio never leaves the machine; quality depends on the model |

**Combined score** = Σ weight × feature, with features in [0, 1] and larger = better:
`asr` = 1 − CER, `lang` = P(lang), `mos` = (MOS − 1) / 4, `spk` = cosine, `edges` = 1 − penalty for
lead > 0.2 s / trail > 0.4 s. Default weights (eval.toml): asr 0.35, lang 0.15, mos 0.35, spk 0.10,
edges 0.05. **Hard filters**: CER > 0.15 or a prefix flag → rejected (never reaches the listening set).

## Tuning loop

`tune` renders every config for every sentence (fixed seed), scores each clip, and averages per
config. Coarse grid: cfg_weight {0.3, 0.5} × exaggeration {0.5, 0.8} × temperature {0.6, 0.8}
× references (spread over the budget when it doesn't fit), then single-axis refinement around
the current best (cfg ±0.15, ex ±0.15, t ±0.1) until the clip budget is used. Output in
`tune-out/<lang>/<timestamp>/`: `clips/`, `scores.csv` (one row per clip), `leaderboard.md`,
`listen/A.wav B.wav C.wav` (top three configs, all sentences concatenated, blind order by
score) and `mapping.json` (letters → configs + features; kept outside `listen/`).

## Preference learning

`prefer` turns a pick (winner beats the other two) or a ranking (all ordered pairs) into
pairwise records in `<user data dir>/fanfare-studio/preferences.jsonl` (Linux:
`~/.local/share/fanfare-studio/`). On the next `tune` the weights are refitted with a
Bradley-Terry / pairwise logistic regression on feature differences, L2-regularised toward the
configured weights, clipped at 0 and rescaled to the same total; with fewer than 5 pairs the
configured weights are used. `tune` prints which weights it used.

## Model sizes and licences (downloaded on first use)

| Model | Licence | Size on disk |
|---|---|---|
| faster-whisper `small` (Systran CTranslate2 conversion of OpenAI Whisper small) | MIT | 464 MB |
| Silero VAD v6 (bundled in the faster-whisper wheel) | MIT | < 3 MB |
| UTMOSv2 `fusion_stage3` fold 0 (sarulab-speech) | MIT (code and weights) | 781 MB |
| facebook/wav2vec2-base (SSL backbone fetched by UTMOSv2) | Apache-2.0 | 726 MB |
| SpeechBrain `spkrec-ecapa-voxceleb` | Apache-2.0 | 85 MB |

Excluded: torchaudio SQUIM subjective MOS — weights are CC-BY-NC-4.0.

## LLM judge with Unsloth

An optional judge asks a local [Unsloth](https://unsloth.ai) instance to rate each clip.

1. Start it with an audio-capable model: `unsloth run --model <audio-capable GGUF>`. It prints the
   port (default `http://localhost:8888`) and an API key `sk-unsloth-...`.
2. Put the key in the environment (**only there**; it is never read from a config file and never
   logged): `export UNSLOTH_STUDIO_AUTH_TOKEN=sk-unsloth-...`
3. Configure `eval.toml`:
   ```toml
   [judge]
   enabled = true
   url = "http://localhost:8888/v1"   # localhost only, must end with /v1
   model = "default"                  # or an id from GET /v1/models
   ```
4. `uv run fanfare-studio eval ...` / `tune ...` now include a `judge` object per clip.

Sending audio to a chat completion (`input_audio`, OpenAI format) is **not documented** by
Unsloth, so the judge probes on first use (result cached for the run): `GET /v1/models` (a 401 gives
"check UNSLOTH_STUDIO_AUTH_TOKEN"; an unknown `judge.model` lists the served ids), `GET /openapi.json`,
then one chat completion with a base64 wav `input_audio` part on a 1 s test clip.

- accepted -> mode `audio`: the full rubric including accent;
- rejected -> mode `transcribe`: `POST /v1/audio/transcriptions` plus a text-only rubric
  ("exact" only); native / announcer / quality are `null` and the comment says
  "accent not assessable in fallback mode".
