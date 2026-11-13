# Reference voices

Chatterbox clones the voice of a short reference clip. Without one it uses its built-in
speaker, an **English** voice — German then sounds like an English speaker with an accent.

| File | Speaker / source | Licence |
|---|---|---|
| `de-thorsten.wav` | Thorsten Müller (Thorsten-Voice), recording `27f4ef76bcf8b7d537e9322b97c89d68` ("Im November vor zwei Jahren habe ich einen Beitrag im Mozilla Forum veröffentlicht und meine Stimmspende angekündigt."), subset TV-2022.10-Neutral, Rode Podcaster. Downloaded from the official dataset https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full (file `TV-2022.10-Neutral/train-00000-of-00008.parquet`; also on Zenodo, DOI 10.5281/zenodo.7265581). Converted: 44.1 kHz → 24 kHz mono 16-bit, silence trimmed (40 dB), peak-normalised to 0.9; 6.6 s, 315 KB. | **CC0-1.0** (public domain dedication; Thorsten donated his voice for TTS) |

Adding a voice for another language: use a 6–10 s clean, single-speaker clip from a CC0 or
CC-BY dataset (record the source and licence here), **or a consenting club member's own
recording**. A voice is personal data (GDPR): get explicit consent, keep the clip on this
computer, delete it when asked. Never use recordings of famous or non-consenting people.
