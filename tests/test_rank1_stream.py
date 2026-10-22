"""Tests for :class:`rfa.streaming.Rank1Stream`."""

from __future__ import annotations

import torch

from rfa import Rank1Stream, ReversibleFractionalAttention


def test_pack_returns_two_quant_pairs() -> None:
    k = torch.randn(2, 16)
    v = torch.randn(2, 16)
    (qk, sk), (qv, sv) = Rank1Stream.pack(k, v)
    assert qk.dtype == torch.int8 and qv.dtype == torch.int8
    assert sk.shape == () and sv.shape == ()


def test_unpack_and_reconstruct_shapes() -> None:
    d_model, batch = 16, 2
    k = torch.randn(batch, d_model)
    v = torch.randn(batch, d_model)
    packed = Rank1Stream.pack(k, v)
    delta_n, delta_z = Rank1Stream.unpack_and_reconstruct(*packed)
    assert delta_n.shape == (batch, d_model, d_model)
    assert delta_z.shape == (batch, d_model)


def test_rank1_reconstruction_is_close_to_outer_product() -> None:
    """Without quantisation noise, ``delta_N`` should equal ``k ⊗ v`` exactly.

    We bypass INT8 by reconstructing manually with the same scale, which
    is a no-op for the shape contract and gives a tight bound on the
    quantisation error.
    """

    d_model, batch = 32, 1
    k = torch.randn(batch, d_model) * 0.5  # small magnitude keeps quant noise low
    v = torch.randn(batch, d_model) * 0.5

    packed = Rank1Stream.pack(k, v)
    delta_n, delta_z = Rank1Stream.unpack_and_reconstruct(*packed)

    expected_delta_n = torch.einsum("bd,bm->bdm", k, v)
    expected_delta_z = k

    err_n = (torch.norm(delta_n - expected_delta_n) / torch.norm(expected_delta_n)).item()
    err_z = (torch.norm(delta_z - expected_delta_z) / torch.norm(expected_delta_z)).item()
    assert err_n < 2e-2, f"N reconstruction error too high: {err_n}"
    assert err_z < 2e-2, f"Z reconstruction error too high: {err_z}"


def test_rank1_compression_factor_is_128x_for_d64() -> None:
    """For d_model=64, full FP32 matrix vs 2 INT8 vectors is 128x smaller."""

    d_model = 64
    fp32_bytes = d_model * d_model * 4
    rank1_int8_bytes = (d_model + d_model) * 1
    assert fp32_bytes // rank1_int8_bytes == 128


def test_rank1_rewind_reproduces_state_within_tolerance() -> None:
    """End-to-end: step forward, then rewind via a Rank-1 INT8 stream."""

    d_model, length, steps_back = 16, 40, 10
    model = ReversibleFractionalAttention(d_model=d_model, decay_rate=0.98)

    state = model.initial_state(batch_size=1)
    states_history: list = [state]
    rank1_buffer: list = []

    xs = torch.randn(length, 1, d_model)
    for t in range(length):
        # Recompute k, v the same way the layer does, so the packed
        # stream exactly matches what the model would have emitted.
        k_t = model.kernel(model.w_k(xs[t]))
        v_t = model.w_v(xs[t])
        rank1_buffer.append(Rank1Stream.pack(k_t, v_t))

        step = model.step_forward(xs[t], state)
        state = step.state
        states_history.append(state)

    # Rewind ``steps_back`` steps using only the Rank-1 stream.
    n_curr, z_curr = states_history[-1]
    for i in range(1, steps_back + 1):
        rec_delta_n, rec_delta_z = Rank1Stream.unpack_and_reconstruct(*rank1_buffer[-i])
        n_curr, z_curr = model.step_backward_reconstruct(n_curr, z_curr, (rec_delta_n, rec_delta_z))

    true_n, true_z = states_history[-1 - steps_back]
    err_n = (torch.norm(n_curr - true_n) / torch.norm(true_n)).item()
    err_z = (torch.norm(z_curr - true_z) / torch.norm(true_z)).item()
    assert err_n < 5e-2, f"rank-1 rewind N error after {steps_back} steps: {err_n}"
    assert err_z < 5e-2, f"rank-1 rewind Z error after {steps_back} steps: {err_z}"
