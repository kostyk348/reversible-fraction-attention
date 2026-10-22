"""Tests for :class:`rfa.streaming.QuantizedStreamDelta`."""

from __future__ import annotations

import pytest
import torch

from rfa import QuantizedStreamDelta


def test_compress_returns_int8_and_scale() -> None:
    x = torch.randn(4, 8, 8) * 3.0
    q, scale = QuantizedStreamDelta.compress(x)
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert scale.shape == ()
    assert q.shape == x.shape


def test_decompress_roundtrip_is_close() -> None:
    x = torch.randn(16, 16) * 2.0
    q, scale = QuantizedStreamDelta.compress(x)
    x_rec = QuantizedStreamDelta.decompress(q, scale)
    # INT8 quantisation: max abs error bounded by ~ scale, relative error tiny.
    rel = (torch.norm(x - x_rec) / torch.norm(x)).item()
    assert rel < 1e-2, f"Relative error too high: {rel}"


def test_decompress_rejects_non_int8() -> None:
    x = torch.randn(8, 8)
    _, scale = QuantizedStreamDelta.compress(x)
    with pytest.raises(TypeError):
        QuantizedStreamDelta.decompress(torch.zeros(8, 8), scale)


def test_quantisation_handles_zero_tensor() -> None:
    """A tensor of all zeros should quantise cleanly without NaNs."""

    x = torch.zeros(4, 4)
    q, scale = QuantizedStreamDelta.compress(x)
    assert torch.isfinite(scale)
    assert torch.all(q == 0)
    x_rec = QuantizedStreamDelta.decompress(q, scale)
    assert torch.all(x_rec == 0)


def test_dynamic_scale_saturates_large_values() -> None:
    """Outliers are clipped to ±127 instead of overflowing the int8 range."""

    x = torch.linspace(-100.0, 100.0, steps=200)
    q, _ = QuantizedStreamDelta.compress(x)
    assert q.max().item() <= 127
    assert q.min().item() >= -127


def test_compression_factor_is_4x() -> None:
    """FP32 -> INT8 should reduce bytes by a factor of 4."""

    x = torch.randn(64, 64)
    fp32_bytes = x.element_size() * x.numel()
    q, _ = QuantizedStreamDelta.compress(x)
    int8_bytes = q.element_size() * q.numel()
    assert fp32_bytes / int8_bytes == 4.0
