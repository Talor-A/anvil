"""I/O layer for game recordings: read, index, and serve compressed game frames.

Every self-play game is recorded as one independent zstd-compressed JSONL frame.
The store indexes frames so any game can be read by index *without* scanning
the whole corpus — essential when a training run consumes 50K+ games and you
want random access or streaming iteration.

Directory layout:

```
data/trajectories/<run_id>/
  manifest.json     provenance pins (run hashes, schema version, pool version)
  obs-NNNN.zst      compressed frame files — one zstd frame per game, JSONL
                    records inside (header, decisions, rets, end)
  index.jsonl       one line per game: (file, offset, clen, seed, record count)
  games.jsonl       per-game outcome records — the TRUE winner lives here, not
                    in the frame end record (see winner_seat() below)
  mu.jsonl          optional: behavior-policy action records for RL training
  labels.jsonl      optional: rollout-label aggregates for fork-point games
```

---

### How frames work

A game is a sequence of JSONL records inside one zstd frame:

```
record 0: {"k": "game", "g": 0, "sv": 1, ...}     # header
record 1..N-2:                                         # decisions + rets
  {"k": "dec", "s": 0, "m": "play", ...}            # decision event
  {"k": "ret", "s": 0, "v": 7, ...}                 # response (same seq)
  {"k": "dec", "s": 1, "m": "play", ...}            # nested decision
record N-1: {"k": "end", "winner": 0, ...}          # end record
```

Decisions *nest* — a parent’s ret arrives after its children’s decisions:

```mermaid
sequenceDiagram
    participant Game
    participant Engine
    Game->>Engine: dec s=0 (host action)
    Engine-->>Game: ret s=0
    Note over Game: --- children now fire ---
    Game->>Engine: dec s=1 (entity action inside host window)
    Engine-->>Game: ret s=1
```

This means training-history reconstruction (which answers were *visible* when
each decision fired) must track the *position in the record stream*, not just
“the answer eventually arrived.” That’s why `_pos` and `_retpos` exist: they
give each decision its place in the stream, and the dataset module uses those
positions to build the correct causal window.

---

### winner_seat(): the winner is not in the frame

There is a subtle gotcha. The end record carries `"winner": 0` — but for games
recorded before mid-2026, that field is the *post-elimination index into the
live player list*, which is almost always 0 (the last survivor). The end
record winner was wrong **~50% of the time**. The real winner comes from the harness progress logs,
merged at ingest time into `games.jsonl`. Always call `winner_seat()`, never
read `end["winner"]`.

---

### open_store() and MultiStore

`open_store()` accepts a single directory or a comma-separated list. When
you pass multiple directories, you get a `MultiStore` — a logical union
that presents several runs as one contiguous game corpus. This is how
replay mixing works: separately stored extension runs (D3 pilot games 0–50K, D6
games 50K–...) are read together.

Game indices must be disjoint — the intended shape is that extension runs
continue the same seed stream, so game index N always means “the Nth
deterministic game” regardless of which store it’s in.

---

### ingest: copy-and-index, not re-encode

The `ingest` subcommand collects worker frame files from a harness run
directory, copies them into the store (renumbered), and builds the index.
It does *not* re-encode: frames are read back by (file, offset, clen), so
orphaned bytes from crashed games (frames that never got an index entry)
are naturally skipped. The corpus is regenerable from seeds + heuristic,
so there is deliberately no backup story.

Key classes:
- `GameTrajectory` — one decoded game: header, decisions (with rets joined), end, marks
- `TrajectoryStore` — reads games from one store directory
- `MultiStore` — reads several stores as one corpus

Key functions:
- `open_store(spec)` — opens a store dir or comma-separated list
- `decode_frame(data)` — decompresses + parses one game frame into components
- `winner_seat(g)` — true winner from the outcome record (not the buggy end field)
- `mu_for_game(g)` — behavior-policy records keyed by (game, sequence id)
- `ingest(...)` — consolidates a run into the store
- `status(root)` — prints a summary of a store

Connects to prior modules: the dataset module (training/dataset.py) streams
from these stores to build task-labeled PyTorch examples; the tensor schemas
(schemas/tensors.py) define the output shapes those examples fill. This module
is the extract step — take raw game frames, produce structured Python objects
the dataset loader can iterate.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

import zstandard

_SEAT = re.compile(r"\((\d+)\)")  # "Anvil(2)-dc-864160" / "Heur(1)-..." -> seat

OBS_SCHEMA_VERSION = 1
TRAJECTORIES_DIR = Path(__file__).parents[2] / "data/trajectories"


@dataclasses.dataclass
class GameTrajectory:
    """One game's decoded frame: header, decisions (answers joined), end.
    marks: fork-point marker records (M2 D4 rollout labels) with _pos set —
    a label's training window is the first priority dec after its mark."""

    header: dict[str, Any]
    decisions: list[dict[str, Any]]  # "dec" records; ret joined as ["ret"]
    end: dict[str, Any] | None
    index: dict[str, Any]  # the index.jsonl entry (seed, lengths, ...)
    marks: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    @property
    def game_index(self) -> int:
        return self.header["g"]


def decode_frame(data: bytes) -> tuple[dict, list[dict], dict | None, list[dict]]:
    """Decode one game frame -> (header, decisions-with-ret-joined, end, marks)."""
    records = [
        json.loads(line)
        for line in zstandard.ZstdDecompressor()
        .decompress(data, max_output_size=1 << 30)
        .splitlines()
    ]
    if not records or records[0].get("k") != "game":
        raise ValueError("frame does not start with a game header record")
    header = records[0]
    if header["sv"] != OBS_SCHEMA_VERSION:
        raise ValueError(f"schema version {header['sv']} != reader version {OBS_SCHEMA_VERSION}")
    decisions: list[dict] = []
    marks: list[dict] = []
    by_seq: dict[int, dict] = {}
    end = None
    for pos, r in enumerate(records[1:]):
        kind = r.get("k")
        if kind == "mark":
            r["_pos"] = pos
            marks.append(r)
        elif kind == "dec":
            # _pos/_retpos: record-stream positions (in-memory only, never
            # serialized). Decisions NEST — a parent's ret can land after its
            # children's decs — and the serve-time history ring back-fills
            # hosts only at ret time, so training history must know WHEN each
            # answer arrived, not just that it eventually did (M2 D2 fix for
            # the nested-window skew documented in Obs.java).
            r["_pos"] = pos
            decisions.append(r)
            by_seq[r["s"]] = r
        elif kind == "ret":
            if r["s"] in by_seq:  # ret without dec = stale-thread record; drop
                by_seq[r["s"]]["ret"] = r["v"]
                by_seq[r["s"]]["_retpos"] = pos
                if "oi" in r:  # exact SA-level option index (logged since 2026-07-10)
                    by_seq[r["s"]]["oi"] = r["oi"]
        elif kind == "end":
            end = r
    return header, decisions, end, marks


class TrajectoryStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.index: list[dict] = [
            json.loads(line) for line in (self.root / "index.jsonl").read_text().splitlines()
        ]
        self._by_game = {e["g"]: e for e in self.index}
        # Per-game outcome records (harness progress logs, merged at ingest).
        # These carry the TRUE winner: the frame end-record's "winner" field
        # is broken pre-fork-fix (derived from the post-elimination live
        # player list -> ~always 0). Never read
        # end["winner"] for outcomes — use winner_seat().
        self.outcomes: dict[int, dict] = {}
        games_path = self.root / "games.jsonl"
        if games_path.exists():
            for line in games_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.outcomes[r["i"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue

    def winner_seat(self, g: int) -> int | None:
        """The seat index of the winner, or None for draws or unparseable games."""
        r = self.outcomes.get(g)
        if not r or r.get("status") != "won" or not r.get("winner"):
            return None
        m = _SEAT.search(r["winner"])
        return int(m.group(1)) - 1 if m else None

    def mu_for_game(self, g: int) -> dict[int, dict] | None:
        """Behavior-policy records for game g, keyed by dec seq (M2 D6
        sampled-actor stores); None when the store carries no mu.jsonl.
        Lazy whole-file load — RL iteration stores are small (~10^5 recs)."""
        if not hasattr(self, "_mu"):
            self._mu: dict[int, dict[int, dict]] | None = None
            self.mu_meta: dict | None = None
            path = self.root / "mu.jsonl"
            if path.exists():
                self._mu = {}
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    if r.get("k") == "meta":
                        self.mu_meta = r
                        continue
                    self._mu.setdefault(r["g"], {})[r["s"]] = r
        return None if self._mu is None else self._mu.get(g, {})

    def __len__(self) -> int:
        return len(self.index)

    def game_indices(self) -> list[int]:
        return sorted(self._by_game)

    def game(self, g: int) -> GameTrajectory:
        entry = self._by_game[g]
        with open(self.root / entry["file"], "rb") as f:
            f.seek(entry["off"])
            data = f.read(entry["clen"])
        header, decisions, end, marks = decode_frame(data)
        if header["g"] != g:
            raise ValueError(f"index says game {g}, frame header says {header['g']}")
        return GameTrajectory(header, decisions, end, entry, marks)

    def games(self, skip_undecodable: bool = False) -> Iterator[GameTrajectory]:
        """Yields each game in index order, skipping frames that fail to decode
        when skip_undecodable is True — training pipelines want the 50K good
        games, not an exception on a truncated write."""
        for g in self.game_indices():
            try:
                yield self.game(g)
            except Exception:
                if not skip_undecodable:
                    raise

    def iter_decisions(
        self, method: str | None = None, by: str | None = None
    ) -> Iterator[tuple[dict, dict]]:
        """Yield (game_header, dec_record) across the store, streaming."""
        for traj in self.games(skip_undecodable=True):
            for dec in traj.decisions:
                if method is not None and dec["m"] != method:
                    continue
                if by is not None and dec.get("by") != by:
                    continue
                yield traj.header, dec


class MultiStore:
    """Multiple TrajectoryStore directories presented as a single contiguous corpus.

    Each store contributes its games to a unified view — the Nth game in the
    combined index always refers to the same deterministic game, no matter
    which physical directory it lives in.

    This requires that game indices never overlap across stores. The intended
    layout is sequential extension runs:

        D3 pilot   → games [0, 50000)
        D6 extension → games [50000, ...)

    If two stores both claim game 17, MultiStore raises on construction,
    because game 17 cannot mean two different games.

    Why not just copy everything into one directory? Because stores can live
    on different volumes, be added incrementally as new runs complete, and
    be queried independently — the union is a reader convenience, not a
    physical merge.
    """

    def __init__(self, roots):
        self.stores = [TrajectoryStore(r) for r in roots]
        self._store_of: dict[int, TrajectoryStore] = {}
        for s in self.stores:
            for g in s.game_indices():
                if g in self._store_of:
                    raise ValueError(
                        f"game {g} present in both {self._store_of[g].root} and "
                        f"{s.root} — extension runs must use disjoint index ranges"
                    )
                self._store_of[g] = s

    def __len__(self) -> int:
        return len(self._store_of)

    def game_indices(self) -> list[int]:
        return sorted(self._store_of)

    def game(self, g: int) -> GameTrajectory:
        return self._store_of[g].game(g)

    def winner_seat(self, g: int) -> int | None:
        return self._store_of[g].winner_seat(g)

    def mu_for_game(self, g: int) -> dict[int, dict] | None:
        return self._store_of[g].mu_for_game(g)

    def games(self, skip_undecodable: bool = False) -> Iterator[GameTrajectory]:
        for g in self.game_indices():
            try:
                yield self.game(g)
            except Exception:
                if not skip_undecodable:
                    raise


def open_store(spec) -> TrajectoryStore | MultiStore:
    """One store dir, a comma-separated string of dirs, or a list of dirs."""
    if isinstance(spec, str) and "," in spec:
        spec = spec.split(",")
    if isinstance(spec, (list, tuple)):
        return MultiStore(spec) if len(spec) > 1 else TrajectoryStore(spec[0])
    return TrajectoryStore(spec)


def ingest(
    run_dir: Path | str,
    dest: Path | str | None = None,
    pool_version: str | None = None,
    verify: bool = False,
    forks: bool = False,
) -> Path:
    """Consolidate a harness run's worker observation files into the store.

    forks=True (M4 D3): ingest the FORK-SESSION frames (obs-forks.zst,
    -forkobs runs) instead of the mainline frames. Drill-run mainlines
    NEVER enter training ingest — under -drillstop they record a
    Forge-computed pseudo-winner (D2.2 pinned hazard). Fork frames carry
    their own true outcomes in the end record (driver-computed winner,
    registered-players index), from which games.jsonl is synthesized;
    per-fork-point aggregates are cross-checked against labels.jsonl.
    """
    run_dir = Path(run_dir)
    run_manifest = json.loads((run_dir / "run.json").read_text())
    run_id = run_manifest["run_id"] + ("-forks" if forks else "")
    dest_path = Path(dest) if dest else TRAJECTORIES_DIR / run_id
    if (dest_path / "manifest.json").exists():
        sys.exit(f"store already exists at {dest_path}; ingest is one-shot (delete it to re-ingest)")
    dest_path.mkdir(parents=True, exist_ok=True)

    src_name = "obs-forks.zst" if forks else "obs.zst"
    index_entries: list[dict] = []
    n_files = 0
    total_clen = 0
    total_rlen = 0
    seen_games: set[int] = set()
    worker_files = sorted(run_dir.glob(f"workers/inv-*/{src_name}"))
    for src in worker_files:
        idx_path = src.with_name(src_name[:-4] + ".idx.jsonl")
        if not idx_path.exists():
            print(f"[ingest] WARNING: {src} has no index sidecar, skipping", file=sys.stderr)
            continue
        fname = f"obs-{n_files:04d}.zst"
        size = src.stat().st_size
        kept = 0
        for line in idx_path.read_text().splitlines():
            e = json.loads(line)
            if e["off"] + e["clen"] > size:
                print(
                    f"[ingest] WARNING: game {e['g']} frame extends past EOF in {src}, dropped",
                    file=sys.stderr,
                )
                continue
            if e["g"] in seen_games:
                # a re-issued game (worker crash path); first complete frame wins
                continue
            seen_games.add(e["g"])
            index_entries.append({"file": fname, **e})
            total_clen += e["clen"]
            total_rlen += e["rlen"]
            kept += 1
        if kept:
            shutil.copy2(src, dest_path / fname)
            n_files += 1

    if not index_entries:
        sys.exit(
            f"no {'fork ' if forks else ''}observation frames found under "
            f"{run_dir}/workers/ — was the run launched with "
            f"{'--fork-obs' if forks else '--obs'}?"
        )

    index_entries.sort(key=lambda e: e["g"])
    with open(dest_path / "index.jsonl", "w") as f:
        for e in index_entries:
            f.write(json.dumps(e) + "\n")

    fork_agg: dict[tuple[int, int], dict] = {}
    quarantined: list[int] = []
    if forks:
        # Synthesize games.jsonl from the fork frames' own end records —
        # the harness progress logs carry MAINLINE outcomes, which under
        # -drillstop are pseudo-winners and must never be joined. The end
        # record's winner is the driver-computed registered-players index
        # (the same tally that feeds labels.jsonl w[]); winner_seat() needs
        # the player NAME, taken from the frame header.
        by_file: dict[str, list[dict]] = {}
        for e in index_entries:
            by_file.setdefault(e["file"], []).append(e)
        rows: dict[int, dict] = {}
        for fname, entries in sorted(by_file.items()):
            data = (dest_path / fname).read_bytes()
            for e in entries:
                try:
                    if e["clen"] == 0 or e["rlen"] > (1 << 30):
                        # phantom rows (zero bytes to the file — fd-death class)
                        # and RAW_CAP runaways
                        # (undecodable under the reader's 1 GiB output cap)
                        raise ValueError("phantom or runaway frame")
                    header, _, end, _ = decode_frame(data[e["off"] : e["off"] + e["clen"]])
                except Exception as ex:  # noqa: BLE001
                    # One bad completion frame must cost one frame, not the
                    # ingest (a failed drill ingest killed the d6-run10
                    # driver). Quarantine loudly: no outcome row -> the game
                    # never trains; dropped from the index below.
                    quarantined.append(e["g"])
                    print(
                        f"[ingest] QUARANTINE fork frame g={e['g']} ({type(ex).__name__}: {ex})",
                        file=sys.stderr,
                    )
                    continue
                fk = header.get("fork") or {}
                agg = fork_agg.setdefault(
                    (fk.get("pg", -1), fk.get("fp", -1)),
                    {"w": [0] * len(header["players"]), "draw": 0, "crash": 0},
                )
                if end is None:
                    print(
                        f"[ingest] WARNING: fork game {e['g']} has no end "
                        f"record (killed worker?); no outcome row",
                        file=sys.stderr,
                    )
                    continue
                winner = None
                if end["status"] == "won" and 0 <= end["winner"] < len(header["players"]):
                    winner = header["players"][end["winner"]]["name"]
                    agg["w"][end["winner"]] += 1
                elif end["status"] == "crash":
                    agg["crash"] += 1
                else:
                    agg["draw"] += 1
                rows[e["g"]] = {
                    "i": e["g"],
                    "status": end["status"],
                    "winner": winner,
                    "turns": end["turns"],
                    "fork": fk,
                }
        with open(dest_path / "games.jsonl", "w") as f:
            for i in sorted(rows):
                f.write(json.dumps(rows[i]) + "\n")
        print(f"[ingest] {len(rows)} fork-completion outcome rows synthesized from end records")
        if quarantined:
            bad = set(quarantined)
            index_entries = [e for e in index_entries if e["g"] not in bad]
            with open(dest_path / "index.jsonl", "w") as f:
                for e in index_entries:
                    f.write(json.dumps(e) + "\n")
            print(
                f"[ingest] WARNING: {len(bad)} fork frame(s) quarantined "
                f"and dropped from the index",
                file=sys.stderr,
            )
    else:
        # merge per-game outcome records (the harness progress logs)
        outcomes: dict[int, dict] = {}
        for f_ in sorted(run_dir.glob("workers/inv-*/games.jsonl")):
            for line in f_.read_text().splitlines():
                try:
                    r = json.loads(line)
                    outcomes[r["i"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue
        with open(dest_path / "games.jsonl", "w") as f:
            for i in sorted(outcomes):
                f.write(json.dumps(outcomes[i]) + "\n")

    # merge rollout-label records (M2 D4 labeler runs); keyed (i, fp),
    # first record wins on chunk re-issue like the frame rule above
    label_rows: dict[tuple[int, int], dict] = {}
    for f_ in sorted(run_dir.glob("workers/inv-*/labels.jsonl")):
        for line in f_.read_text().splitlines():
            try:
                r = json.loads(line)
                label_rows.setdefault((r["i"], r["fp"]), r)
            except (json.JSONDecodeError, KeyError):
                continue
    if label_rows:
        with open(dest_path / "labels.jsonl", "w") as f:
            for key in sorted(label_rows):
                f.write(json.dumps(label_rows[key]) + "\n")
        print(f"[ingest] {len(label_rows)} rollout-label records -> labels.jsonl")

    labels_check = None
    if forks and label_rows:
        # Consistency diagnostic: the per-fork-point frame aggregates must
        # reproduce the labels row (same tally, two paths). Copy crashes
        # never open a frame, so frame crashes <= labels crash; wins and
        # draws must match exactly. A mismatch usually means a mid-block
        # worker re-issue mixed two attempts — frames stay individually
        # valid (each is a completed game with its own outcome), so this
        # warns and records rather than dropping.
        mismatched = []
        for (pg, fp), agg in sorted(fork_agg.items()):
            lab = label_rows.get((pg, fp))
            if lab is None:
                mismatched.append({"pg": pg, "fp": fp, "why": "no labels row"})
                continue
            if agg["w"] != lab["w"] or agg["draw"] != lab["draw"] or agg["crash"] > lab["crash"]:
                mismatched.append(
                    {
                        "pg": pg,
                        "fp": fp,
                        "frames": agg,
                        "labels": {"w": lab["w"], "draw": lab["draw"], "crash": lab["crash"]},
                    }
                )
        labels_check = {"fork_points": len(fork_agg), "mismatched": len(mismatched)}
        if mismatched:
            print(
                f"[ingest] WARNING: {len(mismatched)}/{len(fork_agg)} fork "
                f"points disagree with labels.jsonl: {mismatched[:5]}",
                file=sys.stderr,
            )
        else:
            print(
                f"[ingest] labels cross-check: {len(fork_agg)} fork points "
                f"all agree with labels.jsonl"
            )

    # merge behavior-policy records (M2 D6 sampled actors; server-side file,
    # run-level), keyed (g, s). A re-issued game (first attempt crashed
    # mid-game, e.g. worker OOM) answers AGAIN from s=0 — identical records
    # are fine (same seeded noise), but any CONFLICTING duplicate means the
    # attempts diverged and a (g, s) join against the kept frame would build
    # a chimeric action stream (found live: d6-run1 game 237, out-of-bounds
    # candidate label). Conflicted games lose ALL their mu records — the RL
    # loader then skips them as no_mu.
    mu_src = run_dir / "mu.jsonl"
    if mu_src.exists():
        meta = None
        mu_rows: dict[tuple[int, int], dict] = {}
        conflicted: set[int] = set()
        for line in mu_src.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("k") == "meta":
                meta = r
                continue
            if "g" in r and "s" in r:
                prev = mu_rows.get((r["g"], r["s"]))
                if prev is not None and prev != r:
                    conflicted.add(r["g"])
                mu_rows[(r["g"], r["s"])] = r
        if conflicted:
            n_drop = sum(1 for (g, _) in mu_rows if g in conflicted)
            print(
                f"[ingest] WARNING: {len(conflicted)} game(s) with "
                f"conflicting mu attempts (diverged re-issue) — dropping "
                f"their {n_drop} records: {sorted(conflicted)}",
                file=sys.stderr,
            )
            mu_rows = {k: v for k, v in mu_rows.items() if k[0] not in conflicted}
        with open(dest_path / "mu.jsonl", "w") as f:
            if meta is not None:
                f.write(json.dumps(meta) + "\n")
            for key in sorted(mu_rows):
                f.write(json.dumps(mu_rows[key]) + "\n")
        print(f"[ingest] {len(mu_rows)} behavior-policy records -> mu.jsonl")

    if pool_version is None:
        pool_version = run_manifest.get("pool_version")
    if pool_version is None:
        print(
            "[ingest] WARNING: no pool version in run.json or --pool-version; "
            "provenance is incomplete",
            file=sys.stderr,
        )
    manifest = {
        "run_id": run_id,
        "source": "drill-forks" if forks else "selfplay-heuristic",
        "obs_schema": OBS_SCHEMA_VERSION,
        "pool_version": pool_version,
        "games": len(index_entries),
        "decisions": sum(e["recs"] - 2 for e in index_entries),  # minus game+end records
        "bytes_compressed": total_clen,
        "bytes_raw": total_rlen,
        # run pins, verbatim (fork/jar/anvil hashes, seeds, decks, flags)
        "run": {
            k: run_manifest[k]
            for k in (
                "purpose",
                "created",
                "fork_commit",
                "fork_dirty",
                "anvil_commit",
                "jar_sha256",
                "protocol_version",
                "decks",
                "pairs_sha256",
                "n_pairs",
                "games_per_pair",
                "format",
                "seed_base",
                "games",
                "bridge",
                "tags",
            )
            if k in run_manifest
        },
    }
    if forks:
        # The drill provenance tag: loaders and mixing logic key off this —
        # fork-frame games are position-initialized and must stay
        # distinguishable from full-game trajectories.
        manifest["drill"] = {
            "parent_run": run_manifest["run_id"],
            "drill_file": run_manifest.get("drill_file"),
            "drill_source": run_manifest.get("drill_source"),
            "labels_check": labels_check,
            "quarantined": quarantined,
        }
    (dest_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if verify:
        store = TrajectoryStore(dest_path)
        n_dec = 0
        for traj in store.games():
            if traj.end is None:
                print(
                    f"[ingest] WARNING: game {traj.game_index} has no end record", file=sys.stderr
                )
            n_dec += len(traj.decisions)
        print(f"[ingest] verified: {len(store)} games, {n_dec} decisions decode cleanly")

    ratio = total_rlen / total_clen if total_clen else 0
    print(
        f"[ingest] {run_id}: {len(index_entries)} games -> {dest_path}\n"
        f"[ingest] {total_rlen / 1e6:.1f} MB raw -> {total_clen / 1e6:.1f} MB "
        f"({ratio:.1f}x, {total_clen / max(len(index_entries), 1) / 1e3:.0f} KB/game)"
    )
    return dest_path


def status(root: Path | str) -> None:
    store = TrajectoryStore(root)
    m = store.manifest
    print(
        f"{m.run_id}: {m.games} games, {m.decisions} decisions, "
        f"schema v{m.obs_schema}, pool {m.pool_version}"
    )
    print(
        f"  {m.bytes_raw / 1e6:.1f} MB raw / {m.bytes_compressed / 1e6:.1f} MB compressed "
        f"({m.bytes_raw / max(m.bytes_compressed, 1):.1f}x), "
        f"{m.bytes_compressed / max(m.games, 1) / 1e3:.0f} KB/game"
    )
