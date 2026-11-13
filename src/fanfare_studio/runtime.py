"""Device / execution-provider selection for the model backends.

Two families:

* torch backends take ``--device auto|cuda|rocm|mps|cpu``.
  ROCm builds of PyTorch expose AMD GPUs through the ``torch.cuda`` API, so
  ``rocm`` and ``cuda`` both map to the torch device ``"cuda"``; we only check
  that the installed torch build matches what was asked for.
* ONNX Runtime backends take ``--ep auto|cuda|rocm|dml|openvino|coreml|qnn|vitisai|cpu``.
  The requested provider is tried first, then CPU. ``auto`` picks the first
  available provider in ``EP_AUTO_ORDER``.

The functions take the "what is available" facts as arguments so they can be
unit-tested without torch or onnxruntime installed.
"""

from __future__ import annotations

import logging

log = logging.getLogger("fanfare_studio")

TORCH_DEVICES = ("auto", "cuda", "rocm", "mps", "cpu")

# Friendly name -> ONNX Runtime provider names (first available wins).
EP_PROVIDERS: dict[str, tuple[str, ...]] = {
    "cuda": ("CUDAExecutionProvider",),
    # ROCMExecutionProvider was removed from recent ORT releases in favour of MIGraphX.
    "rocm": ("MIGraphXExecutionProvider", "ROCMExecutionProvider"),
    "dml": ("DmlExecutionProvider",),  # Windows, any DirectX 12 GPU incl. iGPUs
    "openvino": ("OpenVINOExecutionProvider",),  # Intel CPU / iGPU / NPU (untested)
    "coreml": ("CoreMLExecutionProvider",),  # Apple GPU / Neural Engine (untested)
    "qnn": ("QNNExecutionProvider",),  # Qualcomm Snapdragon NPU (untested)
    "vitisai": ("VitisAIExecutionProvider",),  # AMD Ryzen AI NPU (untested)
    "cpu": ("CPUExecutionProvider",),
}
EP_CHOICES = ("auto", *EP_PROVIDERS)
# "rocm" is deliberately NOT in the auto order: onnxruntime-rocm 1.22 on a ROCm 7.2 system
# produced garbled audio for both Chatterbox ONNX and Piper (see docs/model-choice.md).
EP_AUTO_ORDER = ("cuda", "dml", "coreml", "openvino", "qnn", "vitisai", "cpu")
EP_KNOWN_BAD = {"rocm": "known to produce garbled audio (onnxruntime-rocm 1.22, ROCm 7.2)"}
UNTESTED_EPS = frozenset({"openvino", "coreml", "qnn", "vitisai", "dml", "cuda"})


class RuntimeSelectionError(RuntimeError):
    pass


def pick_torch_device(
    requested: str, *, cuda_available: bool, hip_version: str | None, mps_available: bool
) -> str:
    """Return the torch device string for ``requested``.

    Raises RuntimeSelectionError when an explicit device is not usable (explicit
    requests never silently fall back, so the user sees why their GPU is unused).
    """
    if requested not in TORCH_DEVICES:
        raise RuntimeSelectionError(f"unknown device {requested!r}")
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not mps_available:
            raise RuntimeSelectionError("MPS requested but torch.backends.mps is not available")
        return "mps"
    if not cuda_available:
        raise RuntimeSelectionError(f"{requested} requested but torch.cuda.is_available() is False")
    if requested == "rocm" and not hip_version:
        raise RuntimeSelectionError("rocm requested but this torch build is not a ROCm build")
    if requested == "cuda" and hip_version:
        raise RuntimeSelectionError("cuda requested but this torch build is a ROCm build (use rocm)")
    return "cuda"


def torch_device(requested: str) -> tuple[str, str]:
    """Detect with the real torch. Returns (torch device, human description)."""
    import torch

    hip = getattr(torch.version, "hip", None)
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    dev = pick_torch_device(
        requested, cuda_available=torch.cuda.is_available(), hip_version=hip, mps_available=mps
    )
    if dev == "cuda":
        kind = f"ROCm {hip}" if hip else f"CUDA {torch.version.cuda}"
        desc = f"{kind}: {torch.cuda.get_device_name(0)}"
    elif dev == "mps":
        desc = "Apple MPS"
    else:
        desc = f"CPU ({torch.get_num_threads()} threads)"
    return dev, desc


def pick_providers(requested: str, available: list[str]) -> list[str]:
    """Return the ORT provider list to pass to InferenceSession (always ends with CPU)."""
    if requested not in EP_CHOICES:
        raise RuntimeSelectionError(f"unknown execution provider {requested!r}")
    if requested in EP_KNOWN_BAD:
        log.warning("execution provider %r requested explicitly: %s", requested, EP_KNOWN_BAD[requested])
    order = EP_AUTO_ORDER if requested == "auto" else (requested,)
    chosen: list[str] = []
    for name in order:
        hits = [p for p in EP_PROVIDERS[name] if p in available]
        if hits:
            chosen.append(hits[0])
            break
    if requested != "auto" and not chosen:
        log.warning(
            "execution provider %r not available (have: %s); falling back to CPU",
            requested,
            ", ".join(available) or "none",
        )
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen
