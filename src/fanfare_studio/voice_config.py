"""Per-voice synthesis settings for the Chatterbox backends (server config, not protocol).

Loaded from ``voices.toml`` (``--voices-config``). Example::

    [defaults]                 # applied to every voice
    cfg_weight = 0.5
    exaggeration = 0.5
    temperature = 0.8
    seed = 1234

    [voices.de-high]
    lang = "de"
    label = "German (high quality)"
    reference = "voices/de-thorsten.wav"   # relative to the config file
    exaggeration = 0.8

Voices not listed fall back to the built-in list (all 9 Chatterbox languages, built-in
English reference speaker).
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .backends.base import Voice

_LANG_NAMES = {
    "de": "German", "en": "English", "fr": "French", "es": "Spanish", "it": "Italian",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "ar": "Arabic",
}  # fmt: skip
CHATTERBOX_LANGS = tuple(_LANG_NAMES)


@dataclass(frozen=True)
class Settings:
    reference: str | None = None  # path to a 24 kHz+ mono WAV; None = model's built-in speaker
    cfg_weight: float = 0.5
    exaggeration: float = 0.5
    temperature: float = 0.8  # 0 = greedy
    top_p: float = 1.0
    min_p: float = 0.05
    repetition_penalty: float = 1.2
    seed: int | None = 1234  # fixed seed = reproducible clips; None = random
    max_new_tokens: int = 800


@dataclass(frozen=True)
class VoiceEntry:
    voice: Voice
    settings: Settings = field(default_factory=Settings)


_SETTING_KEYS = {f.name for f in dataclasses.fields(Settings)}


def _settings(base: Settings, table: dict, root: Path) -> Settings:
    unknown = set(table) - _SETTING_KEYS - {"lang", "label"}
    if unknown:
        raise ValueError(f"unknown voice setting(s): {', '.join(sorted(unknown))}")
    values = {k: v for k, v in table.items() if k in _SETTING_KEYS}
    if values.get("reference"):
        ref = Path(values["reference"])
        values["reference"] = str(ref if ref.is_absolute() else (root / ref))
    return dataclasses.replace(base, **values)


def load_voices(path: str | None) -> list[VoiceEntry]:
    """Built-in voices merged with ``path`` (if given and present)."""
    data: dict = {}
    root = Path.cwd()
    if path and Path(path).is_file():
        root = Path(path).resolve().parent
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    base = _settings(Settings(), data.get("defaults", {}), root)
    entries: dict[str, VoiceEntry] = {
        f"{lang}-high": VoiceEntry(Voice(f"{lang}-high", lang, f"{name} (high quality)"), base)
        for lang, name in _LANG_NAMES.items()
    }
    for vid, table in data.get("voices", {}).items():
        prev = entries.get(vid)
        lang = table.get("lang", prev.voice.lang if prev else None)
        if lang not in CHATTERBOX_LANGS:
            raise ValueError(f"voice {vid}: lang {lang!r} not supported by Chatterbox")
        label = table.get("label", prev.voice.label if prev else vid)
        entries[vid] = VoiceEntry(Voice(vid, lang, label), _settings(base, table, root))
    for e in entries.values():
        if e.settings.reference and not Path(e.settings.reference).is_file():
            raise FileNotFoundError(f"voice {e.voice.id}: reference {e.settings.reference} not found")
    return list(entries.values())
