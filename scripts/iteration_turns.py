"""Per-iteration turn distribution for a self-play run.

Reads monitor.jsonl to discover each iteration's run dirs, then aggregates
turn counts from the merged games.jsonl. Prints a table and writes JSON for
easy plotting.

Usage:
    python scripts/iteration_turns.py data/training/pauper-red-vs-islands
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def iteration_rows(monitor_path: Path) -> list[dict]:
    rows = []
    for line in open(monitor_path):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def turn_stats(games_path: Path) -> dict:
    turns = []
    for line in open(games_path):
        r = json.loads(line)
        if r.get("status") == "won" and "turns" in r:
            turns.append(r["turns"])
    if not turns:
        return {"games": 0}
    c = Counter(turns)
    return {
        "games": len(turns),
        "turns_min": min(turns),
        "turns_max": max(turns),
        "turns_mean": round(statistics.mean(turns), 2),
        "turns_median": statistics.median(turns),
        "turns_stdev": round(statistics.stdev(turns), 2) if len(turns) > 1 else 0.0,
        "distribution": {str(k): v for k, v in sorted(c.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-iteration turn distribution")
    ap.add_argument("run_dir", type=Path, help="e.g. data/training/pauper-red-vs-islands")
    ap.add_argument("--json", type=Path, default=None, help="optional JSON output path")
    args = ap.parse_args()

    monitor_path = args.run_dir / "monitor.jsonl"
    if not monitor_path.exists():
        raise SystemExit(f"no monitor file: {monitor_path}")

    results = []
    for row in iteration_rows(monitor_path):
        iteration = row.get("iteration")
        runs = row.get("run", [])
        # monitor row stores a list of run dirs; usually one for fixed-deck mode.
        stats: dict[str, dict] = {}
        for run in runs:
            run_dir = Path(run)
            games_path = run_dir / "games.jsonl"
            if games_path.exists():
                stats[run_dir.name] = turn_stats(games_path)
            else:
                stats[run_dir.name] = {"games": 0, "note": "not yet complete"}

        results.append({"iteration": iteration, "runs": stats})

    # Print table
    print(f"{'iter':>4} {'games':>6} {'min':>4} {'med':>4} {'mean':>6} {'max':>4} {'std':>5}")
    print("-" * 40)
    for r in results:
        for run_name, s in r["runs"].items():
            if s.get("games", 0) == 0:
                continue
            print(
                f"{r['iteration']:>4} {s['games']:>6} {s['turns_min']:>4} "
                f"{s['turns_median']:>4} {s['turns_mean']:>6.1f} {s['turns_max']:>4} "
                f"{s['turns_stdev']:>5.1f}"
            )

    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
