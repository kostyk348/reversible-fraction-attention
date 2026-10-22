"""Tests for :class:`rfa.attention.ReversibleFractionalAttention`."""

from __future__ import annotations

import pytest
import torch

from rfa import ReversibleFractionalAttention


def test_init_rejects_invalid_decay_rate() -> None:
    with pytest.raises(ValueError):
        ReversibleFractionalAttention(d_model=8, decay_rate=0.0)
    with pytest.raises(ValueError):
        ReversibleFractionalAttention(d_model=8, decay_rate=1.0)
    with pytest.raises(ValueError):
        ReversibleFractionalAttention(d_model=8, decay_rate=-0.1)


def test_initial_state_shape() -> None:
    model = ReversibleFractionalAttention(d_model=16)
    n, z = model.initial_state(batch_size=4)
    assert n.shape == (4, 16, 16)
    assert z.shape == (4, 16)
    assert torch.all(n == 0)
    assert torch.all(z == 0)


def test_step_forward_output_shapes() -> None:
    d_model, batch = 16, 3
    model = ReversibleFractionalAttention(d_model=d_model, decay_rate=0.9)
    state = model.initial_state(batch_size=batch)
    x_t = torch.randn(batch, d_model)
    step = model.step_forward(x_t, state)

    assert step.out.shape == (batch, d_model)
    n_curr, z_curr = step.state
    assert n_curr.shape == (batch, d_model, d_model)
    assert z_curr.shape == (batch, d_model)
    delta_n, delta_z = step.delta
    assert delta_n.shape == (batch, d_model, d_model)
    assert delta_z.shape == (batch, d_model)


def test_state_does_not_grow_with_sequence_length() -> None:
    """The whole point of the layer: state shape is O(1) in seq_len."""

    model = ReversibleFractionalAttention(d_model=32, decay_rate=0.98)
    state = model.initial_state(batch_size=1)
    initial_n_bytes = state[0].element_size() * state[0].nelement()

    for _ in range(1000):
        x_t = torch.randn(1, 32)
        step = model.step_forward(x_t, state)
        state = step.state

    n_final, _ = state
    final_n_bytes = n_final.element_size() * n_final.nelement()
    assert final_n_bytes == initial_n_bytes
    assert n_final.shape == (1, 32, 32)


def test_forward_sequence_returns_full_output() -> None:
    d_model, batch, length = 16, 2, 20
    model = ReversibleFractionalAttention(d_model=d_model)
    x = torch.randn(batch, length, d_model)
    outputs, final_state, deltas = model.forward_sequence(x)

    assert outputs.shape == (batch, length, d_model)
    assert len(deltas) == length
    assert final_state[0].shape == (batch, d_model, d_model)


def test_forward_sequence_rejects_bad_shape() -> None:
    model = ReversibleFractionalAttention(d_model=8)
    with pytest.raises(ValueError):
        model.forward_sequence(torch.randn(8))  # 1-D, not 3-D


def test_kernel_is_strictly_positive() -> None:
    model = ReversibleFractionalAttention(d_model=8)
    x = torch.randn(4, 8) * 5.0  # large magnitude to probe ELU+1
    k = model.kernel(x)
    assert torch.all(k > 0)


def test_step_forward_is_differentiable() -> None:
    """Smoke test that gradients flow through the step."""

    d_model = 8
    model = ReversibleFractionalAttention(d_model=d_model)
    state = model.initial_state(batch_size=1)
    x = torch.randn(1, d_model, requires_grad=True)
    step = model.step_forward(x, state)
    step.out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
