"""04 — Rank-1 streaming: 128x compression.

Instead of streaming the full ``d x d`` FP32 delta matrix, stream only
the underlying ``k`` and ``v`` vectors (each quantised to INT8). The
outer product is reconstructed on the fly when rewinding.
"""

from __future__ import annotations

import torch

from rfa import Rank1Stream, ReversibleFractionalAttention


def main() -> None:
    torch.manual_seed(42)
    d_model, batch_size, seq_len = 64, 1, 100
    model = ReversibleFractionalAttention(d_model=d_model, decay_rate=0.98)

    state = model.initial_state(batch_size=batch_size)
    states_history = [state]
    inputs = torch.randn(seq_len, batch_size, d_model)
    rank1_buffer: list = []

    for x_t in inputs:
        k_t = model.kernel(model.w_k(x_t))
        v_t = model.w_v(x_t)
        rank1_buffer.append(Rank1Stream.pack(k_t, v_t))

        step = model.step_forward(x_t, state)
        state = step.state
        states_history.append(state)

    fp32_bytes = d_model * d_model * 4
    rank1_bytes = (d_model + d_model) * 1
    print("--- 5. Rank-1 streaming ---")
    print(f"per-step FP32 matrix: {fp32_bytes} bytes")
    print(f"per-step Rank-1 INT8: {rank1_bytes} bytes  ({fp32_bytes // rank1_bytes}x compression)")

    steps_back = 30
    n_curr, z_curr = states_history[-1]
    errors = []
    for i in range(1, steps_back + 1):
        rec_n, rec_z = Rank1Stream.unpack_and_reconstruct(*rank1_buffer[-i])
        n_curr, z_curr = model.step_backward_reconstruct(n_curr, z_curr, (rec_n, rec_z))
        true_n, _ = states_history[-1 - i]
        rel = (torch.norm(n_curr - true_n) / torch.norm(true_n)).item()
        errors.append(rel)
    print(f"rewind error  1 step (Rank-1 INT8): {errors[0] * 100:.2f}%")
    print(f"rewind error 10 steps (Rank-1 INT8): {errors[9] * 100:.2f}%")
    print(f"rewind error 30 steps (Rank-1 INT8): {errors[-1] * 100:.2f}%")


if __name__ == "__main__":
    main()
