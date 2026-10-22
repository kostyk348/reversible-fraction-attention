"""Tests for the strict reversibility of the forward step.

These tests check that :meth:`ReversibleFractionalAttention.step_backward_reconstruct`
recovers the previous state up to floating-point precision. We also
verify the same property for the complex-valued variant.
"""

from __future__ import annotations

import torch

from rfa import ComplexReversibleAttention, ReversibleFractionalAttention


def test_single_step_reversibility_real() -> None:
    d_model, batch = 16, 2
    model = ReversibleFractionalAttention(d_model=d_model, decay_rate=0.97)
    state = model.initial_state(batch_size=batch)

    x_t = torch.randn(batch, d_model)
    step = model.step_forward(x_t, state)

    n_curr, z_curr = step.state
    n_prev, z_prev = model.step_backward_reconstruct(n_curr, z_curr, step.delta)

    err_n = torch.max(torch.abs(n_prev - state[0])).item()
    err_z = torch.max(torch.abs(z_prev - state[1])).item()
    assert err_n < 1e-5, f"N reconstruction error too high: {err_n}"
    assert err_z < 1e-5, f"Z reconstruction error too high: {err_z}"


def test_multi_step_reversibility_real() -> None:
    """Walk the state forward ``T`` steps then backward ``T`` steps."""

    d_model, batch, steps = 8, 1, 50
    model = ReversibleFractionalAttention(d_model=d_model, decay_rate=0.99)
    state = model.initial_state(batch_size=batch)
    initial_state = (state[0].clone(), state[1].clone())

    deltas: list = []
    xs = torch.randn(steps, batch, d_model)
    for t in range(steps):
        step = model.step_forward(xs[t], state)
        state = step.state
        deltas.append(step.delta)

    # Now walk backward using only the deltas.
    n_curr, z_curr = state
    for delta in reversed(deltas):
        n_curr, z_curr = model.step_backward_reconstruct(n_curr, z_curr, delta)

    err_n = torch.max(torch.abs(n_curr - initial_state[0])).item()
    err_z = torch.max(torch.abs(z_curr - initial_state[1])).item()
    assert err_n < 1e-4, f"N reconstruction error after {steps} steps: {err_n}"
    assert err_z < 1e-4, f"Z reconstruction error after {steps} steps: {err_z}"


def test_reversibility_with_zero_delta_recovers_initial_state() -> None:
    """If the delta is zero, the inverse step gives back the same state scaled by 1/gamma."""

    d_model = 8
    model = ReversibleFractionalAttention(d_model=d_model, decay_rate=0.5)
    n_curr = torch.randn(2, d_model, d_model)
    z_curr = torch.randn(2, d_model)
    zero_delta = (
        torch.zeros_like(n_curr),
        torch.zeros_like(z_curr),
    )
    n_prev, z_prev = model.step_backward_reconstruct(n_curr, z_curr, zero_delta)
    assert torch.allclose(n_prev, n_curr / 0.5)
    assert torch.allclose(z_prev, z_curr / 0.5)


def test_single_step_reversibility_complex() -> None:
    d_model, batch = 8, 1
    model = ComplexReversibleAttention(d_model=d_model, decay_rate=0.95)
    state = model.initial_state(batch_size=batch)

    x_t = torch.randn(batch, d_model)
    step = model.step_forward(x_t, state)

    n_curr, z_curr = step.state
    n_prev, z_prev = model.step_backward_reconstruct(n_curr, z_curr, step.delta)

    err_n = torch.max(torch.abs(n_prev - state[0])).item()
    err_z = torch.max(torch.abs(z_prev - state[1])).item()
    # Complex dtor raises the floor a bit; the property should still hold.
    assert err_n < 1e-4, f"complex N reconstruction error: {err_n}"
    assert err_z < 1e-4, f"complex Z reconstruction error: {err_z}"


def test_complex_state_dtype() -> None:
    model = ComplexReversibleAttention(d_model=8)
    n, z = model.initial_state(batch_size=2)
    assert n.dtype == torch.complex64
    assert z.dtype == torch.complex64


def test_phase_rotation_is_unit_magnitude() -> None:
    """``exp(i*theta)`` should always lie on the unit circle."""

    model = ComplexReversibleAttention(d_model=32)
    phase = model.phase_rotation()
    mags = phase.abs()
    assert torch.allclose(mags, torch.ones_like(mags), atol=1e-6)
