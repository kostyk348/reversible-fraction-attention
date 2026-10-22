"""Equivalence between the step-wise and parallel forward passes.

This is the most important property of the library: the step-wise form
(used at inference time, with O(1) state) and the parallel form (used
at training time, with O(L^2) intermediate memory) should produce the
same outputs up to floating-point error when they share the same
``gamma`` and projections.
"""

from __future__ import annotations

import torch

from rfa import ParallelReversibleAttention, ReversibleFractionalAttention


def _share_weights(
    parallel: ParallelReversibleAttention, step: ReversibleFractionalAttention
) -> None:
    """Copy weights from ``step`` into ``parallel`` (same shapes)."""

    parallel.w_q.weight.data.copy_(step.w_q.weight.data)
    parallel.w_k.weight.data.copy_(step.w_k.weight.data)
    parallel.w_v.weight.data.copy_(step.w_v.weight.data)
    parallel.w_out.weight.data.copy_(step.w_out.weight.data)


def test_parallel_matches_step_wise_up_to_tolerance() -> None:
    d_model, batch, length = 16, 1, 32
    gamma_value = 0.95

    step_model = ReversibleFractionalAttention(d_model=d_model, decay_rate=gamma_value)
    parallel_model = ParallelReversibleAttention(
        d_model=d_model, max_len=length, init_decay_param=0.0
    )
    _share_weights(parallel_model, step_model)
    # Force ``parallel_model.current_gamma()`` to return ``gamma_value``.
    # gamma = sigmoid(decay_param * 10 + 2); solve for decay_param.
    # logit(0.95) ≈ 2.944; decay_param = (logit(gamma) - 2) / 10
    import math

    logit = math.log(gamma_value / (1.0 - gamma_value))
    parallel_model.decay_param.data.fill_((logit - 2.0) / 10.0)
    # Sanity-check that we hit the right gamma.
    assert abs(parallel_model.current_gamma().item() - gamma_value) < 1e-5

    x = torch.randn(batch, length, d_model)
    step_out, _, _ = step_model.forward_sequence(x)
    parallel_out = parallel_model(x)

    rel = (torch.norm(step_out - parallel_out) / torch.norm(step_out)).item()
    assert rel < 1e-4, f"step vs parallel mismatch: {rel}"


def test_parallel_output_shape() -> None:
    d_model, batch, length = 8, 3, 16
    model = ParallelReversibleAttention(d_model=d_model, max_len=length)
    x = torch.randn(batch, length, d_model)
    out = model(x)
    assert out.shape == (batch, length, d_model)


def test_parallel_is_differentiable() -> None:
    d_model, batch, length = 8, 2, 16
    model = ParallelReversibleAttention(d_model=d_model, max_len=length)
    x = torch.randn(batch, length, d_model, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # gamma should also have a gradient.
    assert model.decay_param.grad is not None


def test_gamma_stays_in_unit_interval() -> None:
    """``gamma = sigmoid(decay_param * 10 + 2)`` is always in (0, 1).

    We avoid the most extreme values (e.g. ``decay_param = ±100``) where
    sigmoid saturates to exactly 0 or 1 due to float32 underflow — those
    are out of distribution for the parameterisation. With the
    ``decay_param * 10 + 2`` parameterisation, sigmoid saturates for
    ``|decay_param| > ~1.3``, so we use ``±1.0`` which gives
    ``sigmoid(±8) ∈ (3e-4, 0.999)`` — safely inside the open interval.
    """

    model = ParallelReversibleAttention(d_model=8, init_decay_param=-1.0)
    assert 0.0 < model.current_gamma().item() < 1.0
    model.decay_param.data.fill_(1.0)
    assert 0.0 < model.current_gamma().item() < 1.0


def test_associative_recall_learns() -> None:
    """Sanity-check that the parallel form can solve a tiny recall task.

    The model must learn to predict a linear projection of the first
    token from the last output. A healthy implementation drops the loss
    well below its initial value within 200 steps.
    """

    torch.manual_seed(0)
    d_model, length, batch, epochs = 16, 32, 8, 200
    model = ParallelReversibleAttention(d_model=d_model, max_len=length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    target_proj = torch.nn.Linear(d_model, d_model, bias=False)
    target_proj.weight.data.normal_(0, 0.5)

    # Use a single fixed batch — recompute y_target inside the loop so
    # each backward pass has a fresh graph.
    x = torch.randn(batch, length, d_model)

    initial_loss = None
    final_loss = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_target = target_proj(x[:, 0, :])
        out = model(x)
        pred_last = out[:, -1, :]
        loss = torch.nn.functional.mse_loss(pred_last, y_target)
        loss.backward()
        optimizer.step()
        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss * 0.9, (
        f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
    )
