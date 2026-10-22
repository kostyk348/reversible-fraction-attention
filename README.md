# Reversible Fractional Attention (RFA)

> Attention as a **linear dynamic system** with **O(1) state** and an **exactly
> reversible** time step. A research-grade PyTorch implementation extracted
> from the original `TRSS.ipynb` proof-of-concept.

[![CI](https://img.shields.io/badge/CI-pytest%20%2B%20ruff-blue)](.github/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Built with uv](https://img.shields.io/badge/built%20with-uv-purple)](https://docs.astral.sh/uv/)

Documentation in other languages: [**Русский**](README.ru.md)

---

## TL;DR

Standard softmax attention is **O(L²)** in sequence length. Linear attention
removes the softmax but still needs to remember every key/value pair. **RFA**
goes one step further: the recurrence

```
N_t = γ · N_{t-1} + k_t ⊗ v_t       (numerator, d×d matrix)
Z_t = γ · Z_{t-1} + k_t             (denominator, d-vector)
o_t = W_o · (q_t^T N_t) / (q_t^T Z_t + ε)
```

keeps a **constant-size** state `(N, Z)` that does not grow with the sequence
length. Because the update is a linear combination of the previous state and a
freshly computed **delta**, it is **exactly invertible** — given `(N_t, Z_t)`
and the delta `(delta_N, delta_Z)` we can recover `(N_{t-1}, Z_{t-1})` without
re-running the forward pass:

```
N_{t-1} = (N_t - delta_N) / γ
Z_{t-1} = (Z_t - delta_Z) / γ
```

This unlocks cheap **time-reversal** for streaming inference: only the deltas
need to be retained, and they can be heavily quantised (INT8, even Rank-1)
without breaking the inverse.

## Highlights

| Property | Implementation | Test |
|---|---|---|
| O(1) state (independent of `seq_len`) | `ReversibleFractionalAttention.forward_sequence` | `test_state_does_not_grow_with_sequence_length` |
| Strict reversibility (FP32) | `step_backward_reconstruct` | `test_multi_step_reversibility_real` |
| INT8 streaming (~4× compression) | `QuantizedStreamDelta` | `test_decompress_roundtrip_is_close` |
| **Rank-1 streaming** (up to **128×** for `d=64`) | `Rank1Stream` | `test_rank1_rewind_reproduces_state_within_tolerance` |
| Parallel training form (vectorised, O(L²) memory) | `ParallelReversibleAttention` | `test_parallel_matches_step_wise_up_to_tolerance` |
| Complex-valued selective variant | `ComplexReversibleAttention` | `test_single_step_reversibility_complex` |

## Installation

```bash
git clone https://github.com/your-org/reversible-fractional-attention.git
cd reversible-fractional-attention
uv sync --extra dev         # or: pip install -e ".[dev]"
pytest                      # ~5 seconds on CPU
```

## Quick start

```python
import torch
from rfa import ReversibleFractionalAttention

model = ReversibleFractionalAttention(d_model=64, decay_rate=0.98)
state = model.initial_state(batch_size=1)

for x_t in stream_of_tokens:
    step = model.step_forward(x_t, state)
    state = step.state
    # `step.delta` can be quantised + streamed to disk / network
    # and used later to invert the step exactly.
```

Rewind one step from a quantised delta:

```python
from rfa import QuantizedStreamDelta

q_n, s_n = QuantizedStreamDelta.compress(step.delta[0])
q_z, s_z = QuantizedStreamDelta.compress(step.delta[1])
# ... persist (q_n, s_n, q_z, s_z) somewhere ...
rec_n = QuantizedStreamDelta.decompress(q_n, s_n)
rec_z = QuantizedStreamDelta.decompress(q_z, s_z)
n_prev, z_prev = model.step_backward_reconstruct(
    n_curr, z_curr, (rec_n, rec_z)
)
```

## Examples

Run any of them with `python examples/<file>.py`:

| File | What it shows |
|---|---|
| `01_step_forward.py` | Constant O(1) state across 100 tokens |
| `02_reversibility_check.py` | Exact reconstruction error ≈ `1e-6` |
| `03_int8_streaming.py` | INT8 stream + 30-step stress test |
| `04_rank1_compression.py` | 128× compression with Rank-1 streaming |
| `05_parallel_training.py` | Learn an associative-recall task end-to-end |
| `06_complex_attention.py` | Complex-valued selective variant |

## How it works

### Step form (inference, O(1) memory)

At each step the model receives one token `x_t`, computes query/key/value
projections through `Linear` layers and updates two accumulators:

```
q_t = ELU(W_q x_t) + 1     # strictly positive feature map
k_t = ELU(W_k x_t) + 1
v_t = W_v x_t

N_t = γ N_{t-1} + k_t ⊗ v_t      # [d, d] outer product, decayed
Z_t = γ Z_{t-1} + k_t             # [d] vector, decayed

o_t = W_o · (q_t^T N_t) / (q_t^T Z_t + ε)
```

`ELU + 1` is used as a strictly-positive feature map: it keeps the denominator
away from zero without an explicit `+ eps` and is numerically stable for very
negative pre-activations (unlike `exp`).

### Parallel form (training, O(L²) memory)

For training we materialise the causal decay matrix
`Decay[t, i] = γ^{t-i}` for `t ≥ i` and compute all positions in a single
batched matmul. This is **mathematically equivalent** to the step form (the
test `test_parallel_matches_step_wise_up_to_tolerance` verifies relative error
< `1e-4`). The decay rate `γ` is parameterised as
`sigmoid(decay_param * 10 + 2)` so it stays in `(0, 1)` and can be learned
end-to-end.

### Rank-1 streaming

The numerator delta `delta_N = k ⊗ v` is a **rank-1** matrix. Instead of
streaming the full `d × d` FP32 matrix, we stream only the two `d`-vectors
`k` and `v` (each quantised to INT8). The outer product is reconstructed on
the fly when rewinding:

| `d_model` | FP32 matrix | Rank-1 INT8 | Compression |
|---:|---:|---:|---:|
| 32 | 4 KB | 64 B | 64× |
| 64 | 16 KB | 128 B | **128×** |
| 128 | 64 KB | 256 B | 256× |

Rewind error stays below 1.5% after 30 steps (see
`examples/04_rank1_compression.py`).

### Complex-valued variant

The complex variant `ComplexReversibleAttention` adds a **learnable
per-channel phase rotation** `exp(iθ)`:

```
N_t = γ · e^{iθ} ⊙ N_{t-1} + k_t ⊗ v_t
```

This makes the recurrence **selective in *what* is stored**, not only in
*when* — two tokens with the same key but injected at different phases will
interfere differently in the complex plane, so the model can learn to
"rotate away" stale associations.

## Repository layout

```
reversible-fractional-attention/
├── src/rfa/
│   ├── attention.py          # 3 attention layers
│   ├── streaming.py          # INT8 + Rank-1 streaming
│   └── data.py               # TinyShakespeare loader
├── tests/                    # pytest suite (6 files)
├── examples/                 # 6 runnable demos
├── notebooks/                # Cleaned .ipynb
├── .github/workflows/ci.yml  # Python 3.10 / 3.11 / 3.12
├── pyproject.toml            # uv + hatchling + ruff
└── README.md / README.ru.md
```

## Status & limitations

This is a **research preview** extracted from a Colab proof-of-concept.
It is not a drop-in replacement for `nn.MultiheadAttention`:

- No multi-head variant yet (the head dimension is folded into `d_model`).
- The denominator `q_t^T Z_t + ε` is not strictly normalised; for very long
  sequences with low `γ` the output magnitude can drift.
- INT8 quantisation uses a per-tensor scale; per-channel would be more
  accurate for outliers.
- The complex variant has not been benchmarked on a real LM task.

Contributions welcome — see [CONTRIBUTING](#contributing) below.

## Contributing

1. `uv sync --extra dev`
2. `pre-commit install`
3. Add a test under `tests/` that fails without your change.
4. `pytest` and `ruff check .` must be green.
5. Open a PR — CI runs on Python 3.10/3.11/3.12.

## Citation

```bibtex
@misc{rfa2025,
  title  = {Reversible Fractional Attention},
  author = {RFA Contributors},
  year   = {2025},
  note   = {Research preview, Apache-2.0}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
