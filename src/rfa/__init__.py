"""Reversible Fractional Attention (RFA).

A PyTorch implementation of attention as a linear dynamic system with
constant O(1) state and a mathematically reversible time step.

Three interchangeable building blocks are exposed:

* :class:`ReversibleFractionalAttention` — step-wise inference with O(1)
  state and a reversible backward pass via streamed deltas.
* :class:`ParallelReversibleAttention` — vectorised parallel form for
  training, mathematically equivalent to the step-wise form.
* :class:`ComplexReversibleAttention` — complex-valued selective variant
  with a learnable per-channel phase rotation.

Streaming helpers (INT8 quantisation, Rank-1 compression) live in
:mod:`rfa.streaming`.
"""

from rfa.attention import (
    ComplexReversibleAttention,
    ParallelReversibleAttention,
    ReversibleFractionalAttention,
)
from rfa.streaming import QuantizedStreamDelta, Rank1Stream

__all__ = [
    "ComplexReversibleAttention",
    "ParallelReversibleAttention",
    "ReversibleFractionalAttention",
    "QuantizedStreamDelta",
    "Rank1Stream",
    "__version__",
]

__version__ = "0.1.0"
