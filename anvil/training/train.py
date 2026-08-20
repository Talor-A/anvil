"""Behavior cloning (BC) training loop.

`uv run python -m anvil.training.train` -- reads trajectory stores, builds an
AnvilNet (`anvil.policy.model`), and trains it via supervised learning on
expert decisions. This is how the policy learns to play Magic: the Gathering.

Why is training a standalone script instead of a method on AnvilNet? Because
training is a *recipe*, not part of the architecture. The same AnvilNet class
powers different training regimes (BC, RL, fine-tuning) from different scripts.
This module bakes one specific recipe -- the data mix, loss weighting,
learning-rate schedule, and evaluation strategy that produce our production BC
policy.

The reader already knows the pieces this script assembles:

  - Transform (`anvil.encoder.transform`) turns raw observations into dense
    float32 tensors (entity rows, globals, players, history).
  - CardEncoder (`anvil.encoder.cards`) wraps those rows with learned embeddings.
  - AnvilNet (`anvil.policy.model`) stacks a transformer trunk on top, with
    separate heads for each rung-1 decision type (policy, target, X, value,
    bool, num, combat).
  - PriorityWindows (`anvil.training.dataset`) streams decision windows out of
    a TrajectoryStore, yielding one example per priority decision.

main() flow:

    build_net(embed, card_manifest) # CardEncoder + AnvilNet
    PriorityWindows × 3           # train / val / valpair splits
    AdamW + cosine LR + warmup

    loop over train_loader:
        batch: entities, candidates, labels, ...
        losses = net.losses(batch, pass_weight=...)  # dict: loss, policy, target, x, value, ...
        losses["loss"].backward()
        clip_grad_norm_(1.0)
        opt.step()

        if step % eval_every == 0 or step is final:
            evaluate(net, val_loader)
            evaluate(net, vp_loader)         # held-out pair games

── Metrics ──────────────────────────────────────────────────────────

`evaluate()` is where the module's metrics discipline lives. The headline
number is `agree_honest` -- accuracy *excluding* windows with only one legal
option (those are trivially correct). This is the metric runs get compared on.

The family:

  agree_honest     agreement on multi-option windows
  agree_raw        agreement on *all* priority windows (includes forced choices)
  agree_nonpass    agreement on non-PASS decisions within the honest set
  acc_target/X     per-task targeting heads
  acc_mull/trigger/bool/number   one-field decision heads
  acc_atk_row/win, acc_blk_row/win   combat heads (row-level per candidate
                   and window-level exact match)
  value_bce        binary cross-entropy of win-probability vs outcome

Every eval row (and per-100-step training loss) lands in `metrics.jsonl`.
Checkpoints (state_dict + full config) go to `last.pt`.

── The pass-weight trick ────────────────────────────────────────────

PASS is ~90% of decisions but carries near-zero skill signal -- a model that
always picks PASS would score 90% "raw" agreement but play terribly.
`--pass-weight` (default 0.1) scales down the PASS contribution to the loss:

    ┌─────────────┬──────────┬──────────────┐
    │ Decisions   │   Share  │ Loss weight  │
    ├─────────────┼──────────┼──────────────┤
    │ PASS        │   ~90%   │  pass-weight │
    │ non-PASS    │   ~10%   │  1.0         │
    └─────────────┴──────────┴──────────────┘

With --pass-weight 0.1, PASS contributes ~9x less gradient per decision than
a mulligan or combat choice. The model learns to *act* instead of defaulting
to pass.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from anvil.encoder.actiontext import ACTION_TEXT_VERSION
from anvil.encoder.cards import CardEncoder
from anvil.encoder.cardtext import pool_features
from anvil.encoder.transform import (
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HISTORY_K,
    PLAYER_FEATURES,
    TRANSFORM_VERSION,
)
from anvil.policy.model import AnvilNet
from anvil.schemas.manifests import TrainConfig
from anvil.torch.utils import get_torch_device
from anvil.training.dataset import PriorityWindows, collate, default_methods
from anvil.utils.paths import stamp_name

REPO = Path(__file__).parents[1]


def build_net(embedding_stem: str, pool_manifest: str, n_methods: int) -> AnvilNet:
    m = json.loads(Path(pool_manifest).read_text())
    meta = json.loads(Path(f"{embedding_stem}.json").read_text())
    feats = torch.from_numpy(pool_features(m, meta["names"]))
    return AnvilNet(
        CardEncoder(embedding_stem, feats),
        n_entity_features=len(ENTITY_FEATURES),
        n_global=len(GLOBAL_FEATURES),
        n_players=2,
        n_player_features=len(PLAYER_FEATURES),
        n_methods=n_methods,
        history_k=HISTORY_K,
    )


@torch.no_grad()
def evaluate(net: AnvilNet, loader: DataLoader, device: str, max_batches: int) -> dict:
    net.eval()
    agree = raw = 0
    n_honest = n_raw = 0
    agree_np = n_np = 0
    agree_host = n_host = n_masked = 0
    tgt_ok = tgt_n = 0
    tuck_ok = tuck_n = 0
    x_ok = x_n = 0
    vsum = vn = 0.0
    of_ok = {t: 0 for t in ("mull", "trigger", "binary", "number")}
    of_n = {t: 0 for t in ("mull", "trigger", "binary", "number")}
    cmb_ok = {k: 0 for k in ("atk_row", "atk_win", "atk_tgt", "cmb_count", "blk_row", "blk_win")}
    cmb_n = dict(cmb_ok)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device, dtype=torch.bfloat16):
            out = net(batch)
        prio = batch["task"] == 0
        pred = out["policy_logits"].argmax(1)
        valid = batch["label"] >= 0  # action-text label resolved
        lab = batch["label"].clamp(min=0)
        ok = (pred == lab) & prio & valid
        multi = (batch["cand_mask"].sum(1) > 1) & prio  # single-legal-option exclusion
        raw += ok.sum().item()
        n_raw += (prio & valid).sum().item()
        agree += (ok & multi).sum().item()
        n_honest += (multi & valid).sum().item()
        nonpass = (lab > 0) & prio & valid
        agree_np += (ok & nonpass).sum().item()
        n_np += nonpass.sum().item()
        n_masked += (prio & ~valid).sum().item()
        # host-level basis (M1 continuity): the chosen candidate's HOST row vs
        # the expert's, defined on ALL multi windows incl. SA-masked ones
        pred_row = batch["cand_rows"].gather(1, pred.unsqueeze(1)).squeeze(1)
        host_ok = torch.where(pred == 0, batch["label_row"] == -1, pred_row == batch["label_row"])
        agree_host += (host_ok & multi).sum().item()
        n_host += multi.sum().item()
        tm = batch["tgt_labels"] >= 0
        tok = (out["tgt_logits"].argmax(-1) == batch["tgt_labels"]) & tm
        tuck = (batch["task"] == 2).unsqueeze(-1)
        tgt_ok += (tok & ~tuck).sum().item()
        tgt_n += (tm & ~tuck).sum().item()
        tuck_ok += (tok & tuck).sum().item()
        tuck_n += (tm & tuck).sum().item()
        xm = batch["x_val"] >= 0
        x_ok += (out["x_logits"].argmax(-1)[xm] == batch["x_val"][xm]).sum().item()
        x_n += xm.sum().item()
        bpred = out["bool_logit"] > 0
        btrue = batch["bool_label"] == 1
        for tid, name in ((1, "mull"), (3, "trigger"), (4, "binary")):
            m = (batch["task"] == tid) & (batch["bool_label"] >= 0)
            of_ok[name] += ((bpred == btrue) & m).sum().item()
            of_n[name] += m.sum().item()
        nm = (batch["task"] == 5) & (batch["num_label"] >= 0) & (batch["forced"] == 0)
        of_ok["number"] += ((out["num_logits"].argmax(-1) == batch["num_label"]) & nm).sum().item()
        of_n["number"] += nm.sum().item()
        # combat per-tag (D5): row agreement + window exact-set agreement.
        # No forced exclusion — every candidate row is a real yes/no (the
        # forced-empty windows never left the loader).
        am = batch["atk_label"] >= 0
        rok = ((out["atk_logits"] > 0) == (batch["atk_label"] == 1)) & am
        cmb_ok["atk_row"] += rok.sum().item()
        cmb_n["atk_row"] += am.sum().item()
        awm = am.any(1)
        cmb_ok["atk_win"] += (rok.sum(1) == am.sum(1))[awm].sum().item()
        cmb_n["atk_win"] += awm.sum().item()
        atm = batch["atk_tgt_labels"] >= 0
        cmb_ok["atk_tgt"] += (
            ((out["atk_tgt_logits"].argmax(-1) == batch["atk_tgt_labels"]) & atm).sum().item()
        )
        cmb_n["atk_tgt"] += atm.sum().item()
        cm = batch["cmb_count_label"] >= 0
        cmb_ok["cmb_count"] += (
            ((out["cmb_count_logits"].argmax(-1) == batch["cmb_count_label"]) & cm).sum().item()
        )
        cmb_n["cmb_count"] += cm.sum().item()
        bm = batch["blk_label"] >= 0
        brok = (out["blk_logits"].argmax(-1) == batch["blk_label"]) & bm
        cmb_ok["blk_row"] += brok.sum().item()
        cmb_n["blk_row"] += bm.sum().item()
        bwm = bm.any(1)
        cmb_ok["blk_win"] += (brok.sum(1) == bm.sum(1))[bwm].sum().item()
        cmb_n["blk_win"] += bwm.sum().item()
        vm = batch["has_outcome"].bool()
        if vm.any():
            vsum += torch.nn.functional.binary_cross_entropy_with_logits(
                out["value_logit"][vm], batch["won"][vm].float(), reduction="sum"
            ).item()
            vn += vm.sum().item()
    net.train()
    # per-metric ns alongside every rate: the D5 matrix compares runs, and a
    # rate without its sample size hides the noise floor (nonpass at n~1.4K
    # has SE ~1.2% — arms closer than that are indistinguishable)
    return {
        "agree_honest": agree / max(n_honest, 1),  # THE number (forced excluded;
        # SA basis since M2 D2)
        "agree_honest_host": agree_host / max(n_host, 1),  # M1-continuity basis
        "agree_raw": raw / max(n_raw, 1),
        "agree_nonpass": agree_np / max(n_np, 1),
        "n_host": n_host,
        "n_masked": n_masked,
        "acc_target": tgt_ok / max(tgt_n, 1),
        "acc_x": x_ok / max(x_n, 1),
        "value_bce": vsum / max(vn, 1),
        "eval_windows": n_raw,
        "n_honest": n_honest,
        "n_nonpass": n_np,
        "n_target": tgt_n,
        "n_x": x_n,
        "n_value": int(vn),
        "acc_tuck": tuck_ok / max(tuck_n, 1),
        "n_tuck": tuck_n,
        **{f"acc_{t}": of_ok[t] / max(of_n[t], 1) for t in of_ok},
        **{f"n_{t}": of_n[t] for t in of_n},
        **{f"acc_{k}": cmb_ok[k] / max(cmb_n[k], 1) for k in cmb_ok},
        **{f"n_{k}": cmb_n[k] for k in cmb_n},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--store",
        default="data/trajectories/d3pilot-20260704-175219",
        help="store dir, or comma-separated dirs with disjoint game "
        "indices (pilot + extension read as one corpus)",
    )
    ap.add_argument("--embed", default="data/embeddings/cf2ca6ba-qwen3")
    ap.add_argument("--pool-manifest", default="data/pool/pool-cf2ca6ba.json")
    ap.add_argument("--out", default=None)
    # 512 OOMs on a 24GB card sharing with the desktop: big-board windows hit
    # 150+ entity tokens and attention memory is quadratic in sequence length
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--steps", type=int, default=20000)
    # Sweep results: 1.0->0.3 pass-weight bought +7.3pp nonpass
    # for -0.4pp honest, 0.3->0.1 another +3.7pp for -1.3pp; targets/X/value
    # flat throughout. 0.1 = action-rich prior; the honest cost is the
    # pass boundary, recalibratable post-hoc via a PASS-logit offset
    ap.add_argument("--pass-weight", type=float, default=0.1)
    ap.add_argument(
        "--null-text",
        action="store_true",
        help="zero the card-text embedding buffer (text-channel ablation: "
        "does rung-1 use text at all, or only features+ID+dynamics?)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--max-games", type=int, default=None, help="train-subset cap (learning curves)"
    )
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=60)
    # mid-run evals stay cheap (trajectory shape); the final eval is the number
    # runs get compared on. 600 batches ~ 154K windows -> nonpass SE ~0.33%,
    # X-head n in the hundreds; resolves ~1% arm differences in the D5 matrix
    ap.add_argument("--final-eval-batches", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = get_torch_device()
    out_dir = Path(a.out or f"data/training/{stamp_name('run')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = default_methods()
    net = build_net(a.embed, a.pool_manifest, len(methods)).to(device)
    if a.null_text:
        with torch.no_grad():
            net.cards.text.zero_()  # type: ignore[operator]
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    train_ds = PriorityWindows(
        a.store, a.embed, methods, split="train", seed=a.seed, max_games=a.max_games
    )
    val_ds = PriorityWindows(a.store, a.embed, methods, split="val", shuffle_games=False)
    vp_ds = PriorityWindows(a.store, a.embed, methods, split="valpair", shuffle_games=False)
    train = DataLoader(
        train_ds,
        batch_size=a.batch,
        collate_fn=collate,
        num_workers=a.workers,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val = DataLoader(val_ds, batch_size=a.batch, collate_fn=collate, num_workers=4)
    vp = DataLoader(vp_ds, batch_size=a.batch, collate_fn=collate, num_workers=4)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < a.warmup:
            return a.lr * step / a.warmup
        t = (step - a.warmup) / max(a.steps - a.warmup, 1)
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    config = {
        **vars(a),
        "params": n_params,
        "methods_version": 1,
        "action_text_version": ACTION_TEXT_VERSION,
        "transform_version": TRANSFORM_VERSION,
        "embed_meta": json.loads(Path(f"{a.embed}.json").read_text()),
    }
    del config["out"]
    config = TrainConfig(**config).model_dump(mode="json")
    (out_dir / "config.json").write_text(json.dumps(config, indent=1, default=str) + "\n")
    metrics = open(out_dir / "metrics.jsonl", "a")  # noqa: SIM115 -- long-lived metrics append handle closed at end of main
    print(f"[train] {n_params / 1e6:.1f}M params -> {out_dir}")

    step = 0
    t0 = time.time()
    win_seen = 0
    while step < a.steps:
        for batch in train:
            if step >= a.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device, dtype=torch.bfloat16):
                L = net.losses(batch, pass_weight=a.pass_weight)
            opt.zero_grad(set_to_none=True)
            L["loss"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            win_seen += batch["label"].shape[0]
            step += 1

            if step % 100 == 0:
                row = {
                    "step": step,
                    **{
                        k: float(L[k].detach())
                        for k in (
                            "loss",
                            "policy",
                            "target",
                            "x",
                            "value",
                            "bool",
                            "num",
                            "atk",
                            "cmb_count",
                            "atk_tgt",
                            "blk",
                        )
                    },
                    "lr": lr_at(step),
                    "windows": win_seen,
                    "wall_s": round(time.time() - t0, 1),
                }
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                if step % 500 == 0:
                    print(
                        f"[train] step {step}: loss {row['loss']:.3f} "
                        f"({win_seen / (time.time() - t0):.0f} win/s)"
                    )
            if step % a.eval_every == 0 or step == a.steps:
                nb = a.final_eval_batches if step == a.steps else a.eval_batches
                ev = {"step": step, "split": "val", **evaluate(net, val, device, nb)}
                ep = {"step": step, "split": "valpair", **evaluate(net, vp, device, nb)}
                for row in (ev, ep):
                    metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                print(
                    f"[eval] step {step}: honest {ev['agree_honest']:.4f} "
                    f"(host {ev['agree_honest_host']:.4f}) "
                    f"raw {ev['agree_raw']:.4f} nonpass {ev['agree_nonpass']:.4f} "
                    f"tgt {ev['acc_target']:.4f} "
                    f"atk {ev['acc_atk_row']:.4f}/{ev['acc_atk_win']:.4f} "
                    f"blk {ev['acc_blk_row']:.4f} | valpair honest {ep['agree_honest']:.4f}"
                )
                torch.save(
                    {"step": step, "model": net.state_dict(), "config": config}, out_dir / "last.pt"
                )
    print(f"[train] done: {step} steps, {win_seen} windows, {(time.time() - t0) / 3600:.2f}h")


if __name__ == "__main__":
    main()
