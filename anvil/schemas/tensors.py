"""Data contracts for the model: Example (one window) and Batch (padded collation).

Every module that feeds data into the model imports these two TypedDicts.  They are the
narrowest seam in the pipeline: transform.py decides what fields exist, tensors.py says
"these are the fields, period."

The pipeline has two distinct shapes for the same logical record:

    transform.py          tensors.py
    (feature extraction)   (type contracts)
           |                     |
    ┌──────┴──────┐     ┌───────┴───────┐
    │ one window  │ ──> │   Example     │  per-window, natural shapes
    │ of decision │     └───────┬───────┘
    └──────┬──────┘             │
           │              collate()  ←── padding to batch maxes
           │                   │
           │           ┌───────┴───────┐
           └──────────>│    Batch      │  padded to (B, Nmax, ...), plus masks
                       └───────────────┘

---

Example (TypedDict, total=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One decision window *before* batching.  Every field named in transform.py shows up here:
entities, globals, players, history, cand_rows, cand_sa, cand_kind, label, label_row,
x_val, task, bool_label, num_label, num_lo, num_hi, ctx_row, forced, has_outcome, won,
cmb_rows, cmb_count, cmb_count_label, blk_atk_rows, atk_label, atk_tgt_kind,
atk_tgt_idx, blk_label, plus the optional tgt_kind and tgt_idx.

All shapes are whatever one window produces: (n_entities, F), (n_candidates,), etc.
No padding, no masks.  This is the dataset stream type.

---

Batch (TypedDict, total=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The collated version — same conceptual fields but padded to batch-max width so the
GPU can run one contiguous kernel pass.  Each variable-length axis gets a corresponding
boolean mask:

    ent_mask   for the entity axis  (B, N_max_entities)
    cand_mask  for candidate axis   (B, C_max_candidates)
    cmb_mask   for combat rows      (B, A_max_combat_rows)
    blk_atk_mask for attacker rows  (B, M_max_attacker_rows)

Some fields also get fused: tgt_labels replaces tgt_kind/tgt_idx, atk_tgt_labels
replaces atk_tgt_kind/atk_tgt_idx.  This is the GPU type.

The two-type separation exists because:
  1. The dataset iterator yields per-window Examples (no padding waste).
  2. The GPU kernel expects one uniform tensor with masks (no conditionals per sample).
  3. The collation step (in dataset.py) is the single bridge between the two.

---

Slot (TypedDict, total=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Internal server type used by the GPU batcher.  Wraps one Example together with its
priority (pd), cached nonzero indices (nz), an event (ev), output dict (out), and
error (err).  Ignore this unless you are debugging the batcher internals.
"""

# pyright: basic
from __future__ import annotations

from typing import Any, NotRequired, TypedDict

import torch


class Example(TypedDict, total=True):
    """One model example before batching (serve or training)."""

    entities: torch.Tensor
    ent_emb: torch.Tensor
    globals: torch.Tensor
    players: torch.Tensor
    history: torch.Tensor
    cand_rows: torch.Tensor
    cand_sa: torch.Tensor
    cand_kind: torch.Tensor
    label: torch.Tensor
    label_row: torch.Tensor
    tgt_kind: NotRequired[torch.Tensor]
    tgt_idx: NotRequired[torch.Tensor]
    x_val: torch.Tensor
    task: torch.Tensor
    bool_label: torch.Tensor
    num_label: torch.Tensor
    num_lo: torch.Tensor
    num_hi: torch.Tensor
    ctx_row: torch.Tensor
    forced: torch.Tensor
    has_outcome: torch.Tensor
    won: torch.Tensor
    cmb_rows: torch.Tensor
    cmb_count: torch.Tensor
    cmb_count_label: torch.Tensor
    blk_atk_rows: torch.Tensor
    atk_label: torch.Tensor
    atk_tgt_kind: torch.Tensor
    atk_tgt_idx: torch.Tensor
    blk_label: torch.Tensor


class Batch(TypedDict, total=True):
    """Padded batch produced by dataset.collate()."""

    entities: torch.Tensor
    ent_emb: torch.Tensor
    ent_mask: torch.Tensor
    cand_rows: torch.Tensor
    cand_sa: torch.Tensor
    cand_kind: torch.Tensor
    cand_mask: torch.Tensor
    globals: torch.Tensor
    players: torch.Tensor
    history: torch.Tensor
    label: torch.Tensor
    label_row: torch.Tensor
    tgt_labels: torch.Tensor
    x_val: torch.Tensor
    task: torch.Tensor
    bool_label: torch.Tensor
    num_label: torch.Tensor
    num_lo: torch.Tensor
    num_hi: torch.Tensor
    ctx_row: torch.Tensor
    forced: torch.Tensor
    has_outcome: torch.Tensor
    won: torch.Tensor
    cmb_rows: torch.Tensor
    cmb_mask: torch.Tensor
    cmb_count: torch.Tensor
    cmb_count_label: torch.Tensor
    blk_atk_rows: torch.Tensor
    blk_atk_mask: torch.Tensor
    atk_label: torch.Tensor
    atk_tgt_labels: torch.Tensor
    blk_label: torch.Tensor


class Slot(TypedDict, total=True):
    ex: Example
    pd: float
    nz: dict[str, torch.Tensor] | None
    ev: Any
    out: dict[str, torch.Tensor]
    err: Exception
