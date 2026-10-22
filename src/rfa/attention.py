"""Attention layers.

This module contains three attention variants that share the same
linear-dynamic-system formulation but differ in how the recurrence is
executed:

* :class:`ReversibleFractionalAttention` — token-by-token inference with
  constant-size state ``(N, Z)`` and an *exactly invertible* forward
  step. The inverse step uses only the streamed delta, so the previous
  state can be reconstructed without re-running the forward pass.
* :class:`ParallelReversibleAttention` — a vectorised, fully parallel
  form intended for training. The decay matrix is materialised once and
  all positions are computed in a single batched matmul.
* :class:`ComplexReversibleAttention` — a complex-valued selective
  variant where each channel has a learnable phase ``exp(i * theta)``.
  The state rotates in the complex plane, which makes the recurrence
  sensitive to *what* is being stored (rotational selection) rather than
  only to *when* it was stored (exponential decay).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

State = tuple[torch.Tensor, torch.Tensor]
Delta = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class StepOutput:
    """Container returned by :meth:`ReversibleFractionalAttention.step_forward`.

    Attributes
    ----------
    out:
        Output of the attention layer for the current token, shape
        ``[batch, d_model]``.
    state:
        Updated ``(N, Z)`` state, with ``N`` of shape
        ``[batch, d_model, d_model]`` and ``Z`` of shape ``[batch, d_model]``.
    delta:
        The ``(delta_N, delta_Z)`` pair that was added to the previous
        state to obtain the new one. Storing it (or a quantised version
        of it) is sufficient to invert the step later.
    """

    out: torch.Tensor
    state: State
    delta: Delta


def _positive_kernel(x: torch.Tensor) -> torch.Tensor:
    """ELU + 1: strictly positive feature map.

    Using a strictly positive kernel guarantees that the denominator of
    the fractional attention stays away from zero without an explicit
    ``+ eps`` and removes the singularity that vanilla ``exp`` would
    introduce for very negative pre-activations.
    """

    return functional.elu(x) + 1.0


class ReversibleFractionalAttention(nn.Module):
    """Step-wise reversible attention with constant O(1) state.

    The recurrence is

    .. math::

        N_t &= \\gamma \\, N_{t-1} + k_t \\otimes v_t \\\\
        Z_t &= \\gamma \\, Z_{t-1} + k_t \\\\
        o_t &= W_o \\,\\frac{q_t^\\top N_t}{q_t^\\top Z_t + \\varepsilon}

    Both ``N`` (a ``d x d`` numerator matrix) and ``Z`` (a ``d``-vector
    denominator accumulator) have *constant* shape that does not depend
    on the sequence length — hence "O(1) state".

    Because the update is a linear combination of the previous state
    and a freshly computed delta, it is exactly invertible: given
    ``(N_t, Z_t)`` and the delta ``(delta_N, delta_Z)`` we can recover
    ``(N_{t-1}, Z_{t-1}) = ((N_t - delta_N) / gamma, (Z_t - delta_Z) / gamma)``.
    This is exploited by :meth:`step_backward_reconstruct` to "rewind"
    the state without re-running the forward pass.

    Parameters
    ----------
    d_model:
        Model dimensionality (a.k.a. ``d``).
    decay_rate:
        Exponential decay coefficient ``gamma``. Must be in ``(0, 1)``.
    eps:
        Numerical stabiliser added to the denominator.
    """

    def __init__(self, d_model: int, decay_rate: float = 0.95, eps: float = 1e-5) -> None:
        super().__init__()
        if not 0.0 < decay_rate < 1.0:
            raise ValueError(f"decay_rate must lie in (0, 1), got {decay_rate}")
        self.d_model = d_model
        self.gamma = decay_rate
        self.eps = eps
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

    # Public alias used by the parallel form / external code.
    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Strictly positive feature map (ELU + 1)."""

        return _positive_kernel(x)

    def initial_state(self, batch_size: int, device=None, dtype=None) -> State:
        """Return a zero-initialised ``(N, Z)`` state."""

        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        n = torch.zeros(batch_size, self.d_model, self.d_model, device=device, dtype=dtype)
        z = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
        return n, z

    def step_forward(self, x_t: torch.Tensor, state: State) -> StepOutput:
        """Advance one step.

        Parameters
        ----------
        x_t:
            Input token, shape ``[batch, d_model]``.
        state:
            Current ``(N, Z)`` state.

        Returns
        -------
        :class:`StepOutput`
            A dataclass holding the output, the new state and the delta
            that was applied. The delta can be streamed to disk /
            quantised / pushed to a remote buffer and used later to
            reconstruct the previous state via
            :meth:`step_backward_reconstruct`.
        """

        n_prev, z_prev = state
        q = self.kernel(self.w_q(x_t))  # [B, D]
        k = self.kernel(self.w_k(x_t))  # [B, D]
        v = self.w_v(x_t)  # [B, D]

        delta_n = torch.einsum("bd,bm->bdm", k, v)  # [B, D, D]
        delta_z = k  # [B, D]

        n_curr = self.gamma * n_prev + delta_n
        z_curr = self.gamma * z_prev + delta_z

        num = torch.einsum("bd,bdm->bm", q, n_curr)  # [B, D]
        den = torch.einsum("bd,bd->b", q, z_curr).unsqueeze(-1) + self.eps  # [B, 1]
        attn_out = num / den
        out = self.w_out(attn_out)

        return StepOutput(out=out, state=(n_curr, z_curr), delta=(delta_n, delta_z))

    def step_backward_reconstruct(
        self,
        n_curr: torch.Tensor,
        z_curr: torch.Tensor,
        streamed_delta: Delta,
    ) -> State:
        """Invert one forward step.

        Given ``(N_t, Z_t)`` and the delta that was applied at step
        ``t``, recover ``(N_{t-1}, Z_{t-1})`` *exactly* (up to floating
        point error) without re-running the projections or recomputing
        the kernel.

        This is the core "reversibility" property of the layer. With a
        quantised delta (see :class:`rfa.streaming.QuantizedStreamDelta`)
        the reconstruction is approximate but remains stable for many
        steps — see ``examples/03_int8_streaming.py`` and
        ``examples/04_rank1_compression.py`` for stress tests.
        """

        delta_n, delta_z = streamed_delta
        n_prev = (n_curr - delta_n) / self.gamma
        z_prev = (z_curr - delta_z) / self.gamma
        return n_prev, z_prev

    def forward_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, State, list[Delta]]:
        """Run the step-wise recurrence over a full sequence.

        Convenience wrapper around :meth:`step_forward` for training /
        evaluation on a ``[batch, seq_len, d_model]`` tensor.

        Returns
        -------
        outputs:
            Tensor of shape ``[batch, seq_len, d_model]``.
        final_state:
            The ``(N, Z)`` state after the last token.
        deltas:
            The list of per-step deltas, useful for replay / rollback.
        """

        if x.dim() != 3:
            raise ValueError(f"Expected x of shape [B, L, D], got {tuple(x.shape)}")
        b, length, _ = x.shape
        state = self.initial_state(b, device=x.device, dtype=x.dtype)
        outs: list[torch.Tensor] = []
        deltas: list[Delta] = []
        for t in range(length):
            step = self.step_forward(x[:, t, :], state)
            outs.append(step.out)
            state = step.state
            deltas.append(step.delta)
        return torch.stack(outs, dim=1), state, deltas


class ParallelReversibleAttention(nn.Module):
    """Vectorised parallel form of :class:`ReversibleFractionalAttention`.

    Used at training time: the causal decay matrix
    ``Decay[t, i] = gamma^{t - i}`` for ``t >= i`` is materialised once
    and all positions are computed with batched matmuls. This is
    mathematically equivalent to running the step-wise form (see
    ``tests/test_parallel_equivalence.py``), but runs in parallel across
    the time axis at the cost of O(L^2) intermediate memory.

    The decay rate is parameterised as ``gamma = sigmoid(decay_param)``
    so the model can learn it end-to-end while keeping it in ``(0, 1)``.

    Parameters
    ----------
    d_model:
        Model dimensionality.
    max_len:
        Maximum sequence length for which a causal decay buffer is
        pre-allocated.
    init_decay_param:
        Initial value of the raw decay parameter; the corresponding
        ``gamma`` is ``sigmoid(init_decay_param * 10 + 2)``.
    eps:
        Numerical stabiliser added to the denominator.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 2048,
        init_decay_param: float = -0.05,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.decay_param = nn.Parameter(torch.tensor(float(init_decay_param)))
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

        # Lower-triangular distance matrix: distance[t, i] = t - i.
        # The unsqueeze order matters: unsqueeze(1) gives a column vector
        # (the row index ``t``), unsqueeze(0) gives a row vector (the
        # column index ``i``). Subtracting yields a matrix whose entries
        # above the diagonal are negative — i.e. the causal mask
        # ``distance >= 0`` selects the lower triangle, exactly what we
        # need for causal attention.
        indices = torch.arange(max_len)
        distance = indices.unsqueeze(1) - indices.unsqueeze(0)  # [L, L]
        self.register_buffer("distance", distance)

    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        return _positive_kernel(x)

    def current_gamma(self) -> torch.Tensor:
        """Return the current ``gamma`` as a 0-d tensor in ``(0, 1)``."""

        return torch.sigmoid(self.decay_param * 10.0 + 2.0)

    def decay_matrix(self, length: int, gamma: torch.Tensor) -> torch.Tensor:
        """Build the lower-triangular decay matrix for ``length`` steps."""

        dist = self.distance[:length, :length]
        causal = dist >= 0
        return torch.where(
            causal,
            gamma ** dist.float(),
            torch.zeros_like(dist, dtype=torch.float),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel forward pass.

        Parameters
        ----------
        x:
            Input of shape ``[batch, seq_len, d_model]``.

        Returns
        -------
        torch.Tensor
            Output of shape ``[batch, seq_len, d_model]``.
        """

        if x.dim() != 3:
            raise ValueError(f"Expected x of shape [B, L, D], got {tuple(x.shape)}")
        _, length, _ = x.shape
        gamma = self.current_gamma()
        q = self.kernel(self.w_q(x))  # [B, L, D]
        k = self.kernel(self.w_k(x))  # [B, L, D]
        v = self.w_v(x)  # [B, L, D]

        decay = self.decay_matrix(length, gamma).to(dtype=x.dtype, device=x.device)  # [L, L]
        qk = torch.bmm(q, k.transpose(1, 2))  # [B, L, L]
        attn = qk * decay.unsqueeze(0)  # [B, L, L]
        numerator = torch.bmm(attn, v)  # [B, L, D]

        k_decayed = torch.matmul(decay, k)  # [B, L, D]
        denominator = (q * k_decayed).sum(dim=-1, keepdim=True) + self.eps  # [B, L, 1]

        return self.w_out(numerator / denominator)


class ComplexReversibleAttention(nn.Module):
    """Complex-valued reversible attention with per-channel phase rotation.

    The recurrence becomes

    .. math::

        N_t = \\gamma \\, e^{i\\theta} \\odot N_{t-1} + k_t \\otimes v_t

    where ``theta`` is a learnable per-channel phase. The denominator
    accumulator ``Z`` is similarly rotated. This adds a *selective*
    component on top of pure exponential decay: two tokens with the same
    ``k`` but injected at different phases will interfere differently
    in the complex plane, so the model can learn to "rotate away" stale
    associations.

    The implementation works entirely with real tensors by stacking the
    real and imaginary parts along a trailing axis of size 2. This keeps
    it compatible with vanilla ``nn.Linear`` and CUDA kernels.

    Parameters
    ----------
    d_model:
        Model dimensionality.
    decay_rate:
        Decay coefficient ``gamma`` in ``(0, 1)``.
    eps:
        Numerical stabiliser.
    """

    def __init__(
        self,
        d_model: int,
        decay_rate: float = 0.95,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if not 0.0 < decay_rate < 1.0:
            raise ValueError(f"decay_rate must lie in (0, 1), got {decay_rate}")
        self.d_model = d_model
        self.gamma = decay_rate
        self.eps = eps
        self.theta = nn.Parameter(torch.randn(d_model) * 0.1)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        return _positive_kernel(x)

    def phase_rotation(self) -> torch.Tensor:
        """Return the per-channel complex phase ``exp(i * theta)``.

        Returned tensor has shape ``[d_model]`` and dtype ``complex64``.
        """

        return torch.complex(
            torch.cos(self.theta),
            torch.sin(self.theta),
        )

    def initial_state(
        self, batch_size: int, device=None, dtype=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-initialise the complex ``(N, Z)`` state.

        ``N`` has shape ``[batch, d_model, d_model]`` and ``Z`` has shape
        ``[batch, d_model]``, both with ``complex64`` dtype.
        """

        device = device if device is not None else next(self.parameters()).device
        n = torch.zeros(
            batch_size, self.d_model, self.d_model, device=device, dtype=torch.complex64
        )
        z = torch.zeros(batch_size, self.d_model, device=device, dtype=torch.complex64)
        return n, z

    def step_forward(
        self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> StepOutput:
        """Advance one complex-valued step.

        ``x_t`` is a real tensor of shape ``[batch, d_model]``. The
        projections ``q``, ``k`` are made complex by multiplying the
        real kernel output with the per-channel phase ``exp(i*theta)``.
        ``v`` is kept real and the recurrence proceeds in complex
        arithmetic. The output is the real part of ``W_o`` applied to
        the complex attention output.
        """

        n_prev, z_prev = state
        phase = self.phase_rotation()  # [D] complex

        q = self.kernel(self.w_q(x_t)).to(torch.complex64) * phase  # [B, D] complex
        k = self.kernel(self.w_k(x_t)).to(torch.complex64) * phase  # [B, D] complex
        v = self.w_v(x_t).to(torch.complex64)  # [B, D] complex

        delta_n = torch.einsum("bd,bm->bdm", k, v)  # [B, D, D] complex
        delta_z = k  # [B, D] complex

        rot = self.gamma * phase  # [D] complex — combined decay+rotation
        n_curr = rot.view(1, -1, 1) * n_prev + delta_n
        z_curr = rot.view(1, -1) * z_prev + delta_z

        num = torch.einsum("bd,bdm->bm", q.conj(), n_curr)  # [B, D] complex
        den = torch.einsum("bd,bd->b", q.conj(), z_curr).unsqueeze(-1) + self.eps
        attn_out = num / den
        out = self.w_out(attn_out.real)

        return StepOutput(out=out, state=(n_curr, z_curr), delta=(delta_n, delta_z))

    def step_backward_reconstruct(
        self,
        n_curr: torch.Tensor,
        z_curr: torch.Tensor,
        streamed_delta: Delta,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert one complex-valued step.

        The inverse operator is ``(1 / (gamma * exp(i*theta)))`` applied
        per-channel. As with the real variant, only the streamed delta
        is required.
        """

        delta_n, delta_z = streamed_delta
        phase = self.phase_rotation()
        rot = self.gamma * phase  # [D] complex
        inv_rot = (1.0 / rot).conj()  # for complex division by gamma*exp(iθ)
        n_prev = inv_rot.view(1, -1, 1) * (n_curr - delta_n)
        z_prev = inv_rot.view(1, -1) * (z_curr - delta_z)
        return n_prev, z_prev
