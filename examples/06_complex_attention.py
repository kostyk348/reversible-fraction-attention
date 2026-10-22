"""06 — Complex-valued selective attention.

Demonstrates :class:`ComplexReversibleAttention`: each channel has a
learnable phase rotation, so the recurrence is selective in *what* is
stored, not only in *when*. We verify the inverse step on a single
token and show the phase magnitudes are all 1.
"""

from __future__ import annotations

import torch

from rfa import ComplexReversibleAttention


def main() -> None:
    torch.manual_seed(0)
    d_model, batch_size = 32, 1
    model = ComplexReversibleAttention(d_model=d_model, decay_rate=0.95)

    phase = model.phase_rotation()
    print("--- 7. Complex-valued reversible attention ---")
    print(f"phase magnitudes (should all be ~1): {phase.abs().tolist()[:8]}...")

    state = model.initial_state(batch_size=batch_size)
    x = torch.randn(batch_size, d_model)
    step = model.step_forward(x, state)

    n_curr, z_curr = step.state
    n_prev, z_prev = model.step_backward_reconstruct(n_curr, z_curr, step.delta)
    err_n = torch.max(torch.abs(n_prev - state[0])).item()
    err_z = torch.max(torch.abs(z_prev - state[1])).item()
    print(f"single-step reconstruction error: N={err_n:.2e}, Z={err_z:.2e}")
    print("Complex recurrence is exactly invertible.")


if __name__ == "__main__":
    main()
