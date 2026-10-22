"""Streaming helpers — compress / decompress deltas for rewind.

The forward step of :class:`rfa.attention.ReversibleFractionalAttention`
returns a delta ``(delta_N, delta_Z)`` that, together with the current
state, is sufficient to recover the previous state. Storing the full
FP32 delta at every step would defeat the purpose of having an O(1)
state in the first place, so we provide two lossy-but-compact encodings:

* :class:`QuantizedStreamDelta` — symmetric INT8 quantisation with a
  per-tensor dynamic scale. Compresses by ~4x for FP32 inputs and is
  the simplest building block for an external stream buffer.
* :class:`Rank1Stream` — instead of streaming the full ``d x d``
  matrix ``delta_N``, stream only the underlying ``k`` and ``v``
  vectors (which factorise ``delta_N = k ⊗ v``). This shrinks the
  per-step footprint from ``d^2 * 4`` bytes to ``2d * 1`` byte for
  INT8, a 128x compression for ``d=64``.
"""

from __future__ import annotations

import torch

Quant = tuple[torch.Tensor, torch.Tensor]  # (int8 tensor, fp32 scale)


class QuantizedStreamDelta:
    """Symmetric INT8 quantisation of a single delta tensor.

    The quantiser maps the original tensor to the range ``[-127, 127]``
    using a per-tensor ``scale = max(|x|) / 127``. Decompression is a
    single multiply. This is the simplest possible streaming format and
    is meant to emulate what a chip would push to / pull from an
    external stream buffer.
    """

    @staticmethod
    def compress(tensor: torch.Tensor) -> Quant:
        """Compress ``tensor`` to INT8 with a dynamic per-tensor scale.

        Parameters
        ----------
        tensor:
            Any floating point tensor.

        Returns
        -------
        (q_tensor, scale):
            ``q_tensor`` is ``int8`` of the same shape; ``scale`` is a
            0-d float32 tensor such that
            ``q_tensor.float() * scale ≈ tensor``.
        """

        max_val = torch.max(torch.abs(tensor)).clamp(min=1e-8)
        scale = (max_val / 127.0).to(torch.float32)
        q_tensor = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
        return q_tensor, scale

    @staticmethod
    def decompress(q_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Reverse :meth:`compress`."""

        if q_tensor.dtype != torch.int8:
            raise TypeError(f"Expected int8 quantised tensor, got {q_tensor.dtype}")
        return q_tensor.to(torch.float32) * scale


class Rank1Stream:
    """Stream ``(k, v)`` instead of the full ``d x d`` delta matrix.

    Because the numerator delta factorises as
    ``delta_N = k ⊗ v`` (an outer product of two ``d``-vectors), we
    never need to store the matrix. We stream the two vectors instead
    — each quantised to INT8 — and reconstruct the outer product on the
    fly when rewinding.

    For ``d_model = 64`` this reduces the per-step footprint from
    ``16384`` bytes (FP32 ``d x d`` matrix) to ``128`` bytes (two INT8
    vectors of length ``d``), a 128x compression with negligible error.
    """

    @staticmethod
    def pack(k: torch.Tensor, v: torch.Tensor) -> tuple[Quant, Quant]:
        """Quantise ``k`` and ``v`` independently and return both.

        Parameters
        ----------
        k, v:
            Two tensors of the same shape, typically ``[batch, d_model]``.

        Returns
        -------
        ((q_k, s_k), (q_v, s_v)):
            Two quantised pairs, ready to push onto the stream.
        """

        return QuantizedStreamDelta.compress(k), QuantizedStreamDelta.compress(v)

    @staticmethod
    def unpack_and_reconstruct(
        packed_k: Quant,
        packed_v: Quant,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuild ``(delta_N, delta_Z)`` from packed ``(k, v)``.

        Returns the reconstructed numerator delta (an outer product)
        and the denominator delta (the reconstructed ``k`` vector),
        matching the convention used by
        :meth:`rfa.attention.ReversibleFractionalAttention.step_backward_reconstruct`.
        """

        (q_k, s_k) = packed_k
        (q_v, s_v) = packed_v
        k_rec = QuantizedStreamDelta.decompress(q_k, s_k)
        v_rec = QuantizedStreamDelta.decompress(q_v, s_v)
        delta_n = torch.einsum("bd,bm->bdm", k_rec, v_rec)
        delta_z = k_rec
        return delta_n, delta_z
