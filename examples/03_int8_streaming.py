"""03 — INT8 streaming and stress test.

Quantises every per-step delta to INT8 and rewinds ``steps_back`` steps
through the quantised stream. Reports per-step and accumulated error.
"""

from __future__ import annotations

import torch

from rfa import QuantizedStreamDelta, ReversibleFractionalAttention


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

    # Single-step INT8 error.
    print("--- 3. Rewind via INT8 stream ---")
    delta_n, delta_z = deltas[-1]
    q_n, s_n = QuantizedStreamDelta.compress(delta_n)
    q_z, s_z = QuantizedStreamDelta.compress(delta_z)
    print(f"FP32 delta_N size: {delta_n.nelement() * 4} bytes")
    print(f"INT8 delta_N size: {q_n.nelement() * 1} bytes  (4x compression)")
    rec_n = QuantizedStreamDelta.decompress(q_n, s_n)
    rec_z = QuantizedStreamDelta.decompress(q_z, s_z)
    n_curr, z_curr = states_history[-1]
    n_rec, z_rec = model.step_backward_reconstruct(n_curr, z_curr, (rec_n, rec_z))
    n_true, z_true = states_history[-2]
    rel = (torch.norm(n_rec - n_true) / torch.norm(n_true)).item()
    print(f"relative state error (1 step, INT8): {rel * 100:.2f}%")

    # Stress test: rewind 30 steps.
    print("\n--- 4. Stress test: rewind 30 steps via INT8 stream ---")
    steps_back = 30
    n_curr, z_curr = states_history[-1]
    errors = []
    for i in range(1, steps_back + 1):
        raw_dn, raw_dz = deltas[-i]
        q_dn, s_dn = QuantizedStreamDelta.compress(raw_dn)
        q_dz, s_dz = QuantizedStreamDelta.compress(raw_dz)
        dec_dn = QuantizedStreamDelta.decompress(q_dn, s_dn)
        dec_dz = QuantizedStreamDelta.decompress(q_dz, s_dz)
        n_curr, z_curr = model.step_backward_reconstruct(n_curr, z_curr, (dec_dn, dec_dz))
        true_n, _ = states_history[-1 - i]
        rel = (torch.norm(n_curr - true_n) / torch.norm(true_n)).item()
        errors.append(rel)
    print(f"error after  1 step: {errors[0] * 100:.2f}%")
    print(f"error after 10 steps: {errors[9] * 100:.2f}%")
    print(f"error after 30 steps: {errors[-1] * 100:.2f}%")


if __name__ == "__main__":
    main()
