from __future__ import annotations

import logging

import pytest

from fanfare_studio.runtime import (
    RuntimeSelectionError,
    pick_providers,
    pick_torch_device,
)


def dev(req, cuda=False, hip=None, mps=False):
    return pick_torch_device(req, cuda_available=cuda, hip_version=hip, mps_available=mps)


def test_torch_auto_order():
    assert dev("auto", cuda=True, hip="6.2", mps=True) == "cuda"
    assert dev("auto", mps=True) == "mps"
    assert dev("auto") == "cpu"


def test_torch_explicit():
    assert dev("rocm", cuda=True, hip="6.2") == "cuda"
    assert dev("cuda", cuda=True) == "cuda"
    assert dev("cpu", cuda=True) == "cpu"
    assert dev("mps", mps=True) == "mps"


@pytest.mark.parametrize(
    "req,kw",
    [("rocm", {"cuda": True}), ("cuda", {"cuda": True, "hip": "6.2"}), ("cuda", {}), ("mps", {}), ("tpu", {})],
)
def test_torch_explicit_unavailable_raises(req, kw):
    with pytest.raises(RuntimeSelectionError):
        dev(req, **kw)


def test_ep_auto_prefers_gpu_then_cpu():
    assert pick_providers("auto", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert pick_providers("auto", ["DmlExecutionProvider", "CPUExecutionProvider"]) == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert pick_providers("auto", ["MIGraphXExecutionProvider", "CPUExecutionProvider"])[0] == (
        "MIGraphXExecutionProvider"
    )


@pytest.mark.parametrize(
    "ep,provider",
    [
        ("cuda", "CUDAExecutionProvider"),
        ("rocm", "ROCMExecutionProvider"),
        ("dml", "DmlExecutionProvider"),
        ("openvino", "OpenVINOExecutionProvider"),
        ("coreml", "CoreMLExecutionProvider"),
        ("qnn", "QNNExecutionProvider"),
        ("vitisai", "VitisAIExecutionProvider"),
    ],
)
def test_ep_explicit(ep, provider):
    assert pick_providers(ep, [provider, "CPUExecutionProvider"]) == [provider, "CPUExecutionProvider"]


def test_ep_missing_falls_back_to_cpu_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert pick_providers("openvino", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert "falling back to CPU" in caplog.text


def test_ep_unknown_raises():
    with pytest.raises(RuntimeSelectionError):
        pick_providers("tpu", [])
