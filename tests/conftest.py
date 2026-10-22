"""Pytest fixtures shared across the RFA test-suite."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Make every test deterministic by default."""

    torch.manual_seed(42)


@pytest.fixture()
def tiny_config() -> dict:
    """Tiny dimensions so the suite runs in well under a second on CPU."""

    return {"d_model": 16, "batch_size": 2, "seq_len": 32}
