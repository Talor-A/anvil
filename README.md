# Anvil

An MTG AI trained via behavior cloning on expert recordings, then iteratively
improved through V-trace self-play. The model is a pre-LN transformer
(d=512, 8 heads, 10 layers) with a pointer decoder and several task-specific
output heads.

## Module Reading Order

This is the intended order for a **new contributor** who wants to grow their
first checkpoint. Each module assumes familiarity with the ones before it.

### 1. Featurization — `anvil/encoder/transform.py`

The input representation. This is the **one place** where raw game observations
get turned into dense float32 arrays. Information-set enforcement (what the
model is allowed to know) lives here and only here. The three constant lists
(`ENTITY_FEATURES`, `GLOBAL_FEATURES`, `PLAYER_FEATURES`) define every column
of every tensor in the whole system.

### 2. Tensor schemas — `anvil/schemas/tensors.py`

The data contracts: `Example` (one decision window, pre-batch) and `Batch`
(the padded/collated version fed to the GPU). Small and boring by design —
every other module imports these two TypedDicts.

### 3. Card encoder — `anvil/encoder/cards.py`

Turns card names into dense vectors for the transformer. Three channels fused
by a 2-layer MLP: frozen text embeddings, structured pool features (CMC,
power/toughness, etc.), and a learned ID lookup for memorization.
Hidden-identity entities get learned null vectors (no information leak).

### 4. State token assembly — `anvil/state/tokens.py`

`StateAssembler` — builds the transformer input sequence from all the pieces:
`[STATE] [PLAN] entity_0 ... entity_N history_0 ... history_K`. The `[STATE]`
token is the read-out used by the value head and pointer query. Entity features
fuse with card vectors here (not in CardEncoder), so dynamic fields (tapped,
counters, count) stay separate from static card text.

### 5. The model — `anvil/policy/model.py`

**AnvilNet** — the central class. A pre-LN transformer with:
- Policy head: pointer over candidates (index 0 = PASS).
- Target decoder: autoregressive over T_MAX+1 slots (entities, players, STOP).
- X head, one-field bool/num heads, value head (win-probability).
- Combat heads: factorized per-candidate-row attack/block.
- `forward()` (teacher-forced training), `act()` (autoregressive inference,
  also handles Gumbel sampling), `losses()` (all CE/BCE terms with masking).

### 6. Dataset — `anvil/training/dataset.py`

The most Magic-specific module. Reads stored trajectories and yields one
`Example` per decision "window". Defines the task taxonomy:
`priority`, `mull_keep`, `mull_tuck`, `trigger`, `binary`, `number`,
`attack`, `block`. Handles candidate resolution from expert labels,
SA-level ambiguity, combat label reconstruction from later-in-combat
observations, and batch padding via `collate()`.

### 7. Trajectory store — `anvil/store/trajectories.py`

The I/O layer for game recordings. Each game is one zstd-compressed JSONL
frame. The store indexes frames by (file, offset, compressed length) for
random access without scanning. Key classes: `GameTrajectory` (decoded game),
`open_store()` (opens a dir or comma-separated list for replay mixing),
`mu_for_game()` (sampled-action records for self-play).

### 8. BC training loop — `anvil/training/train.py`

The entry point for **growing your first checkpoint**. Builds the net,
opens `PriorityWindows` datasets (train/val/valpair splits), runs standard
AdamW + cosine LR, and periodically evaluates. The headline metric is
`agree_honest` — fraction of non-forced decisions where the model matches
the expert. Run with:
```
uv run python -m anvil.training.train --store data/trajectories/... --embed data/embeddings/... --pool-manifest data/pool/pool-...json
```

### 9. V-trace learner — `anvil/training/rl.py`

The RL machinery that improves the policy from self-play data. Key pieces:
- `composite_logp()`: sum of log-probs over all labeled heads (the V-trace
  importance-sampling ratio).
- `vtrace_targets()`: per-trajectory corrected value targets and advantages.
- Two forward passes per trajectory: pass A (no-grad for targets/ratios),
  pass B (grad for policy gradient + value regression + entropy hinge).
- Segmented forward passes for VRAM elasticity. Mu tripwire detects
  serve/loader pipeline skew.

### 10. Self-play loop driver — `anvil/training/selfplay.py`

Orchestrates the full improvement cycle on one GPU. Each iteration:
1. Serve sampled on the current checkpoint
2. Generate N games (mirror + optional heuristic opponents)
3. Ingest mu records into the trajectory store
4. Train V-trace on a replay mixture of recent stores
5. Guard-check (KL drift, entropy, veto rate, cast count)
6. Periodically run arms vs the heuristic with paired seeds

### 11. Bridge server — `anvil/bridge/server.py`

The gRPC inference server. Featurizes wire observations (same code path as
the training dataset), runs the model, and returns answers. Uses a `_Batcher`
for GPU micro-batching across concurrent worker streams. Writes mu records
during sampled self-play generation.

### 12. Sampling — `anvil/policy/sampling.py`

Generates Gumbel noise that pushes `act()` from greedy (argmax) to sampled
exploration for self-play. Also records mu records — JSON-serialized
behavior-policy action probabilities that the V-trace learner consumes.
The record format must stay in lockstep with `apply_mu_labels()` in rl.py.