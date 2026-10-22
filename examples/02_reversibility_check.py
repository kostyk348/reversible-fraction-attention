"""02 — Strict reversibility check.

Walks the state forward ``seq_len`` steps and then inverts the last
step using :meth:`step_backward_reconstruct`. The reconstruction error
should be at the level of float32 rounding (``~1e-6``).
"""

from __future__ import annotations

import torch

from rfa import ReversibleFractionalAttention


def main() -> None:
    torch.manual_seed(42)
    d_model, batch_size, seq_len = 64, 1, 100
    model = ReversibleFractionalAttention(d_model=d_model, decay_rate=0.98)

    state = model.initial_state(batch_size=batch_size)
    states_history = [state]
    deltas: list = []

    inputs = torch.randn(seq_len, batch_size, d_model)
    for x_t in inputs:
        step = model.step_forward(x_t, state)
        state = step.state
        states_history.append(state)
        deltas.append(step.delta)

    print("--- 2. Strict reversibility (reverse step) ---")
    n_end, z_end = states_history[-1]
    last_delta = deltas[-1]
    n_rec, z_rec = model.step_backward_reconstruct(n_end, z_end, last_delta)

    n_true, z_true = states_history[-2]
    err_n = torch.max(torch.abs(n_rec - n_true)).item()
    err_z = torch.max(torch.abs(z_rec - z_true)).item()
    print(f"max N reconstruction error: {err_n:.2e}")
    print(f"max Z reconstruction error: {err_z:.2e}")
    if err_n < 1e-5 and err_z < 1e-5:
        print("State is exactly reversible in time without recomputation!")


if __name__ == "__main__":
    main()
