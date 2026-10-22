"""01 — Forward pass with O(1) state.

Runs the step-wise recurrence for ``seq_len`` tokens and prints the
final accumulator size. The point: the state shape does not depend on
the sequence length.
"""

from __future__ import annotations

import torch

from rfa import ReversibleFractionalAttention


def main() -> None:
    torch.manual_seed(42)
    d_model, batch_size, seq_len = 64, 1, 100
    model = ReversibleFractionalAttention(d_model=d_model, decay_rate=0.98)

    state = model.initial_state(batch_size=batch_size)
    inputs = torch.randn(seq_len, batch_size, d_model)

    print(f"--- 1. Forward pass over {seq_len} tokens ---")
    for x_t in inputs:
        step = model.step_forward(x_t, state)
        state = step.state

    n_final = state[0]
    print(
        f"Final numerator accumulator shape: {tuple(n_final.shape)} "
        f"({n_final.element_size() * n_final.nelement()} bytes)"
    )
    print("Shape is independent of seq_len — that is the O(1) state guarantee.")


if __name__ == "__main__":
    main()
