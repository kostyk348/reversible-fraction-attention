"""Tests for the Tiny Shakespeare data utilities."""

from __future__ import annotations

import pytest
import torch

from rfa.data import (
    CharTokenizer,
    build_tokenizer,
    encode_to_tensor,
    random_batch,
    split_dataset,
)


def test_tokenizer_round_trip() -> None:
    text = "hello world"
    tok = build_tokenizer(text)
    assert tok.vocab_size == len(set(text))
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_tokenizer_ignores_unknown_chars() -> None:
    tok = CharTokenizer(chars=list("abc"))
    assert tok.encode("abd") == [0, 1]  # 'd' silently dropped
    assert tok.decode([0, 2]) == "ac"


def test_split_dataset_lengths() -> None:
    train, val = split_dataset("x" * 100, train_fraction=0.9)
    assert len(train) == 90
    assert len(val) == 10


def test_encode_to_tensor_dtype() -> None:
    tok = CharTokenizer(chars=list("abc"))
    t = encode_to_tensor(tok, "abcabc")
    assert t.dtype == torch.long
    assert t.tolist() == [0, 1, 2, 0, 1, 2]


def test_random_batch_shapes() -> None:
    data = torch.arange(100, dtype=torch.long)
    x, y = random_batch(data, block_size=8, batch_size=4)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y should be x shifted by one.
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_random_batch_rejects_oversized_block() -> None:
    data = torch.arange(5, dtype=torch.long)
    with pytest.raises(ValueError):
        random_batch(data, block_size=10, batch_size=1)
