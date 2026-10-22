# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2025-01-15

### Added
- Initial extraction from the `TRSS.ipynb` proof-of-concept.
- `ReversibleFractionalAttention` — step-wise inference with O(1) state.
- `ParallelReversibleAttention` — vectorised training form, mathematically
  equivalent to the step form.
- `ComplexReversibleAttention` — complex-valued selective variant with a
  learnable per-channel phase rotation.
- `QuantizedStreamDelta` — symmetric INT8 quantisation for streaming deltas.
- `Rank1Stream` — 128× compression by streaming only `(k, v)` vectors.
- Tiny Shakespeare character-level dataset loader.
- 6 example scripts under `examples/`.
- 6 test modules under `tests/`.
- GitHub Actions CI matrix on Python 3.10 / 3.11 / 3.12.
- Bilingual README (English + Russian).
- Apache 2.0 license.

[0.1.0]: https://github.com/your-org/reversible-fractional-attention/releases/tag/v0.1.0
