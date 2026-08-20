"""StateAssembler — build the transformer input sequence from all the pieces.

Recall that the reader has already seen:
  - CardEncoder: turns each card (text + static features + ID) into a
    fixed-size vector d_card. It intentionally leaves out dynamic fields
    like tapped, counters, or count — those change every board state.
  - transform.py: defines the per-entity feature schema (zone, tapped,
    counters, count, ...), the global features (turn, phase, priority
    player, ...), and the player features (life, hand size, ...).
  - schemas/tensors.py: Batch type — the padded container that carries all
    of these as named tensors.

StateAssembler is the glue that fuses those separate feature streams into
one flat token sequence the trunk transformer can ingest:

    [STATE]  [PLAN]  ent_0  ent_1  ...  ent_N  hist_0  hist_1  ...  hist_K

Each slot is a d_model vector. The trunk sees a (B, 1+1+N+K, d_model)
sequence and doesn't need to know about zones, methods, or where the
pieces came from.

WHY fuse entity features with card vectors HERE instead of in CardEncoder?
CardEncoder's job is to map a card *identity* (text + oracle text + static
attributes) to a fixed embedding. That embedding is cached and reused across
games — it never changes during a match. Dynamic fields (tapped, counters,
count, damage, …) update every board state, so they need a separate path.
StateAssembler concatenates the static card vector with the dynamic entity
features before projecting to d_model, keeping the two cleanly separated
until the last possible moment. This means CardEncoder outputs can be
precomputed once per deck and the dynamic features flow straight from the
environment.

Token-by-token breakdown:

  [STATE]  — pooled global + player features → single read-out token.
             The value head and the pointer query both read from this
             position. Think of it as the "board summary" the trunk uses
             to make decisions.

  [PLAN]   — reserved latent token for a future turn-plan module.
             Currently initialized as a learned parameter and left unused
             (no gradient flows through it until the plan head is attached).
             It's here from the start to keep the sequence layout stable.

  ent_i    — one token per unique entity on the board.
             Constructed as:
               card_vec + entity_features  —[Linear(d_card + n_feat → d_model)]→ ent_i
             where card_vec comes from CardEncoder and entity_features are
             the dynamic per-entity fields (zone location, tapped flag,
             counters, etc.). The ent_mask from the batch tells the trunk
             which slots are padding.

  hist_k   — one token per past action in the history window.
             Each history row stores:
               [method_id, actor_is_self_flag, host_row_or_-1]
             The method-id gets an embedding lookup; the self/opponent
             flag gets a separate 2-element embedding; the two are
             concatenated, projected back to d_model, then summed with
             a learned position encoding (so the trunk knows the order
             of past actions). The third column (host_row) is reserved
             for future target-pointer work and is currently unused.

Data flow summary (pseudocode):

    def forward(card_vecs, batch):
        # card_vecs:   (B, N, d_card)   — from CardEncoder (static)
        # batch is a Batch dict:
        #   entities: (B, N, n_entity_features)   — dynamic per-entity
        #   globals:  (B, n_global)                — turn, phase, ...
        #   players:  (B, n_players, n_player_features) — life, hand, ...
        #   history:  (B, K, 3)                   — (method, self, host)
        #   ent_mask: (B, N)                      — True = real entity

        ent = ent_proj(card_vecs ‖ entities)             # (B, N, d_model)
        state = state_proj(globals ‖ players.flatten(1)) # (B, 1, d_model)

        hist embeddings:
            method = method_emb(history[:,:,0])          # (B, K, d_model/2)
            self   = self_emb(history[:,:,1])            # (B, K, d_model/2)
            htok   = hist_proj(method ‖ self) + hist_pos # (B, K, d_model)

        plan = plan_token expanded to batch               # (B, 1, d_model)

        tokens = [state, plan, ent, htok]                 # (B, 1+1+N+K, d_model)
        return tokens, padding_mask

The return is a pair: (tokens, key_padding_mask) where True means "ignore
this position" (padding). The [STATE] and [PLAN] slots are always real.
"""

from __future__ import annotations

import torch
from torch import nn


class StateAssembler(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_card: int,
        n_entity_features: int,
        n_global: int,
        n_players: int,
        n_player_features: int,
        n_methods: int,
        history_k: int,
    ):
        super().__init__()
        self.ent_proj = nn.Linear(d_card + n_entity_features, d_model)
        self.state_proj = nn.Linear(n_global + n_players * n_player_features, d_model)
        self.plan_tok = nn.Parameter(torch.zeros(1, d_model))  # [PLAN] latent (§3)
        self.method_emb = nn.Embedding(n_methods + 2, d_model // 2)  # +OOV +pad(-1)
        self.self_emb = nn.Embedding(2, d_model // 2)
        self.hist_proj = nn.Linear(d_model, d_model)
        self.hist_pos = nn.Parameter(torch.zeros(history_k, d_model))

    def forward(self, card_vecs: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (tokens (B, 1+N+K, d), key_padding_mask (B, 1+N+K) True=PAD)."""
        b = card_vecs.shape[0]
        ent = self.ent_proj(torch.cat([card_vecs, batch["entities"]], dim=-1))
        state = self.state_proj(
            torch.cat([batch["globals"], batch["players"].flatten(1)], dim=-1)
        ).unsqueeze(1)

        hist = batch["history"]  # (B, K, 3): method, self, host-row(-1 ok, unused v0)
        method = self.method_emb(
            hist[..., 0].clamp(min=0) + (hist[..., 0] < 0).long() * 0
        )  # pad -> id 0, masked below
        selfsame = self.self_emb(hist[..., 1].clamp(min=0))
        htok = self.hist_proj(torch.cat([method, selfsame], dim=-1)) + self.hist_pos

        plan = self.plan_tok.expand(b, 1, -1)
        tokens = torch.cat([state, plan, ent, htok], dim=1)
        pad = torch.cat(
            [
                torch.zeros(b, 2, dtype=torch.bool, device=ent.device),  # [STATE],[PLAN]
                ~batch["ent_mask"],
                hist[..., 0] < 0,  # unused history slots
            ],
            dim=1,
        )
        return tokens, pad
