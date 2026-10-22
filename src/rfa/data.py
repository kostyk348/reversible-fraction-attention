"""Tiny Shakespeare character-level dataset.

A minimal, dependency-free loader for the canonical
`tinyshakespeare <https://github.com/karpathy/char-rnn>`__ corpus used
to sanity-check the attention modules on a real (if toy) language
modelling task. The data is fetched once and cached on disk.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

_DEFAULT_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
_DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "tinyshakespeare.txt"


@dataclass
class CharTokenizer:
    """Bidirectional char ↔ id mapping."""

    chars: list[str]

    def __post_init__(self) -> None:
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)


def load_tinyshakespeare(
    url: str = _DEFAULT_URL,
    cache: Path | None = None,
    force_refresh: bool = False,
) -> str:
    """Download (and cache) the tinyshakespeare corpus as a single string.

    The corpus is cached at ``<repo>/.cache/tinyshakespeare.txt`` so
    repeated runs in development do not re-download it.
    """

    cache = cache or _DEFAULT_CACHE
    if force_refresh or not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        text = urllib.request.urlopen(url).read().decode("utf-8")
        cache.write_text(text, encoding="utf-8")
    return cache.read_text(encoding="utf-8")


def build_tokenizer(text: str) -> CharTokenizer:
    """Build a character-level tokenizer from ``text``."""

    return CharTokenizer(chars=sorted(set(text)))


def split_dataset(
    text: str,
    train_fraction: float = 0.9,
) -> tuple[str, str]:
    """Split ``text`` into ``(train, val)`` by character count."""

    n_train = int(train_fraction * len(text))
    return text[:n_train], text[n_train:]


def encode_to_tensor(tokenizer: CharTokenizer, text: str) -> torch.Tensor:
    """Encode ``text`` with ``tokenizer`` into a 1-D ``long`` tensor."""

    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def random_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random ``(x, y)`` batch for next-token prediction.

    ``x`` and ``y`` are both of shape ``[batch_size, block_size]`` and
    ``y`` is ``x`` shifted by one position.
    """

    if data.numel() <= block_size:
        raise ValueError(f"data has {data.numel()} tokens, need > block_size={block_size}")
    high = data.numel() - block_size - 1
    ix = torch.randint(0, high, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
