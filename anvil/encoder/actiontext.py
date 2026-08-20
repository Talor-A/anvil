"""Learned, open-vocabulary embeddings for legal-action text.

Priority actions used to be represented by a row in a pinned list of every
spell-ability string seen while building the training corpus. That made the
policy checkpoint depend on the list and mapped every unseen action to one OOV
row. This module instead composes each action from hashed word and character
features. New actions therefore have a representation without changing an
action vocabulary or model shape.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import lru_cache

import torch
from torch import nn

ACTION_TEXT_VERSION = 1
ACTION_TEXT_BUCKETS = 32768
ACTION_TEXT_FEATURES = 96
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _bucket(feature: str) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % ACTION_TEXT_BUCKETS + 1  # 0 is padding


def action_text_tokens(text: str) -> list[int]:
    """Return a fixed-width sequence of hashed lexical features."""
    return list(_action_text_tokens_cached(text))


@lru_cache(maxsize=16384)
def _action_text_tokens_cached(text: str) -> tuple[int, ...]:
    """Cached immutable core shared by training workers and the server.

    Word unigrams and bigrams retain broad semantics; character trigrams let
    unseen names and rules words share fragments with training text. Hash
    buckets are an implementation detail, not IDs for complete actions.
    """
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    if not text:
        return (0,) * ACTION_TEXT_FEATURES

    words = _TOKEN.findall(text)
    # Put whole-word semantics first so long names near the front cannot crowd
    # the rest of the action out with character fragments. N-grams then fill
    # the remaining budget for open-vocabulary sharing.
    features = [f"w:{word}" for word in words]
    features.extend(f"b:{a}\0{b}" for a, b in zip(words, words[1:]))
    for word in words:
        if len(word) >= 3:
            marked = f"^{word}$"
            features.extend(f"c:{marked[j : j + 3]}" for j in range(len(marked) - 2))

    ids = [_bucket(feature) for feature in features[:ACTION_TEXT_FEATURES]]
    return tuple(ids + [0] * (ACTION_TEXT_FEATURES - len(ids)))


class ActionTextEncoder(nn.Module):
    """Encode ``(..., ACTION_TEXT_FEATURES)`` token IDs to ``(..., d_out)``.

    ``EmbeddingBag`` fuses lookup and reduction: memory scales with the number
    of candidates rather than candidates × text length × embedding width.
    """

    def __init__(self, d_out: int, d_token: int = 128):
        super().__init__()
        self.features = nn.EmbeddingBag(
            ACTION_TEXT_BUCKETS + 1, d_token, mode="mean", padding_idx=0
        )
        self.proj = nn.Sequential(
            nn.Linear(d_token, d_out), nn.GELU(), nn.Linear(d_out, d_out)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        shape = tokens.shape[:-1]
        flat = tokens.reshape(-1, tokens.shape[-1])
        return self.proj(self.features(flat)).reshape(*shape, -1)
