"""CardEncoder — turns card-name indices into dense card vectors for the transformer.

The downstream `StateAssembler` (see `anvil.state.tokens`) expects one
256-d vector per card slot. This encoder produces them by fusing three
complementary channels:

```
  card_row_idx
       │
  ┌────┼────┐
  ▼    ▼    ▼
 text  feats  id_emb        ← three channels
  │     │     │
  └─────┼─────┘
        ▼
   concat + MLP  →  d_card=256
```

**Text embedding** (`self.text`, frozen `(n, d_text)` buffer):
One pre-computed vector per card name, loaded from a safetensors cache
(see `cardtext.py` for how the raw name/type/oracle text is rendered).
The embedding model was trained externally and never ships at inference.
This channel gives the transformer a rich semantic signal — abilities,
types, keywords, flavor — all captured as a dense vector.

**Structured features** (`self.feats`, frozen `(n, d_feat)` buffer):
Pool stats computed offline — CMC, pip counts, type flags, power/toughness,
loyalty, number of faces. Same row order as the text table, so they align
one-to-one. Text captures *what the card does in words*; features capture
*what the card does in numbers* — the mechanical quantities the rules engine
actually computes.

**ID embedding** (`self.id_emb`, learned `(n+1) × d_id`):
A per-card memorization slot. Why does it exist? Two cards with very
different rules text might produce near-identical text embeddings (e.g.,
two different "destroy target creature" instant-speed spells). The model
needs to know they are *different cards*. The ID embedding provides a
disambiguation channel — a learned lookup that says "card 42 ≠ card 17"
even when their semantic vectors are close.

Initialised at `~1/sqrt(d_text)` scale so the ID channel starts at roughly
the same volume as the text embedding, rather than the default `N(0,1)`
which would give it a ~60× head start over the generalisation channels.

**The –1 / OOV path:**
When a row index is –1 (hidden identity, token, emblem — any entity whose
card text must not leak to the transformer), every channel substitutes a
null sentinel:

- `self.null_text` — learned zero-initialised vector for the text channel
- `self.null_feats` — learned zero-initialised vector for the feature channel
- `self.id_emb` index `n` (the dedicated no-card slot) — learned ID vector

Result: a deterministic learned null vector with *zero information leakage*
from the hidden entity.

**Fusion MLP:**

```
torch.cat([text, feats, ids], dim=-1)   # d_in = d_text + d_feat + d_id
  → Linear(d_in, 2*d_card)
    → GELU
      → Linear(2*d_card, d_card)       # output: (..., 256)
```

**Why frozen?**
The text embeddings and features come from a separate pipeline — the
embedding model, the cardsfolder parser, the feature extractor — that
exist independently of the policy network. Freezing them at the card-encoder
boundary enforces that separation. The designed escape hatch is *distillation*:
train the fusion MLP + ID embedding to produce identical output without the
text-embedding dependency, then drop the frozen tables at inference.

The reader has already seen `transform.py` (entity features from raw state)
and `tensors.py` (Example / Batch shapes). This module sits between them:
`transform.py` assigns card indices to entities, this encoder turns those
indices into vectors, and the vectors go into the `StateAssembler` token
sequence defined by `tensors.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class CardEncoder(nn.Module):
    def __init__(
        self, embedding_stem: str | Path, features: torch.Tensor, d_card: int = 256, d_id: int = 48
    ):
        super().__init__()
        from safetensors.torch import load_file

        meta = json.loads(Path(f"{embedding_stem}.json").read_text())
        text = load_file(f"{embedding_stem}.safetensors")["embeddings"].float()
        n, d_text = text.shape
        assert features.shape[0] == n, "feature table must align with the embedding cache"
        self.meta = meta
        self.register_buffer("text", text)  # frozen (n, d_text)
        self.register_buffer("feats", features.float())  # frozen (n, d_feat)
        self.null_text = nn.Parameter(torch.zeros(d_text))
        self.null_feats = nn.Parameter(torch.zeros(features.shape[1]))
        self.id_emb = nn.Embedding(n + 1, d_id)  # +1 = the no-card id
        # start the ID (memorization) channel at text-embedding volume
        # (~1/sqrt(d_text)); default N(0,1) init hands it a ~60x head start
        # over the generalization channel
        nn.init.normal_(self.id_emb.weight, std=0.02)
        d_in = d_text + features.shape[1] + d_id
        self.fuse = nn.Sequential(
            nn.Linear(d_in, 2 * d_card), nn.GELU(), nn.Linear(2 * d_card, d_card)
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """rows (..., ) int64, -1 = no card -> (..., d_card)."""
        safe = rows.clamp(min=0)
        known = (rows >= 0).unsqueeze(-1)
        text = torch.where(known, self.text[safe], self.null_text)
        feats = torch.where(known, self.feats[safe], self.null_feats)
        ids = self.id_emb(
            torch.where(rows >= 0, rows, torch.full_like(rows, self.id_emb.num_embeddings - 1))
        )
        return self.fuse(torch.cat([text, feats, ids], dim=-1))
