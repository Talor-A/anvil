#!/usr/bin/env python3
"""Pauper two-deck BC corpus visualizations.

Produces a self-contained HTML report under
`data/training/pauper-bc-twodecks/report.html` (or --out) with:
- Game-level summary: win rate, turns, game length, status distribution.
- Deck-vs-deck matchup matrix.
- Most-played / least-played cards (casts, battlefield presence).
- Deck-specific cast tables and "never cast" spot checks.
- Token and sideboard usage.
- Land drop / curve timing.
- Combat frequency and attacker/blocker counts.
- Decision-method frequency.
- Per-card first-appearance turn (who shows up when).
- Data-quality spot checks: pass-with-options, host-not-in-obs, stuck games.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from anvil.store.trajectories import TrajectoryStore


def _fmt_pct(n: int, total: int) -> str:
    return f"{100 * n / max(total, 1):.1f}%"


def _parse_deck(path: Path) -> tuple[Counter, Counter]:
    """Return (main, side) card counters from a .dck file."""
    main: Counter = Counter()
    side: Counter = Counter()
    section = "header"
    for line in path.read_text().splitlines():
        if line.startswith("["):
            section = "side" if line.strip() == "[Sideboard]" else "other"
            continue
        if not line.strip():
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        qty, name = parts
        (side if section == "side" else main)[name] += int(qty)
    return main, side


def gather(store_dir: Path):
    store = TrajectoryStore(store_dir)
    outcomes = []
    for line in (store_dir / "games.jsonl").read_text().splitlines():
        try:
            outcomes.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # card-level stats
    cast_counts = Counter()
    cast_by_deck: dict[str, Counter] = defaultdict(Counter)
    zone_card_counts: dict[str, Counter] = defaultdict(Counter)
    card_first_turn: dict[str, list[int]] = defaultdict(list)
    card_deck_counts: dict[str, Counter] = defaultdict(Counter)
    token_counts = Counter()
    attacker_counts: list[int] = []
    blocker_counts: list[int] = []
    combat_games = set()
    method_counts = Counter()
    priority_options_counts: list[int] = []
    land_drop_turn: dict[str, list[int]] = defaultdict(list)

    # outcome aggregates
    deck_records: dict[str, dict] = defaultdict(lambda: {"w": 0, "n": 0, "turns": []})
    profile_pair_counts = Counter()
    matchup_counts: dict[tuple[str, str], dict] = defaultdict(lambda: {"w": 0, "n": 0})

    # data-quality counters
    pass_total = 0
    pass_with_options = 0
    cast_total = 0
    host_not_in_obs = 0
    mull_total = 0
    mull_kept = 0
    stuck_games = 0
    slow_games = 0
    max_priority_reps: list[int] = []

    deck_mains: dict[str, Counter] = {}
    deck_sides: dict[str, Counter] = {}
    pool_dir = Path("data/pool/pauper/decks")
    for deck in ("pau-874705", "pau-877589"):
        main, side = _parse_deck(pool_dir / f"{deck}.dck")
        deck_mains[deck] = main
        deck_sides[deck] = side

    for o in outcomes:
        decks = o["decks"]
        winner = o["winner"]
        pair = tuple(decks)
        matchup_counts[pair]["n"] += 1
        for i, d in enumerate(decks):
            deck_records[d]["n"] += 1
            deck_records[d]["turns"].append(o["turns"])
            if winner == f"Anvil({i + 1})-{d}":
                deck_records[d]["w"] += 1
                matchup_counts[pair]["w"] += 1
        profile_pair_counts[tuple(o.get("profiles", []))] += 1
        if o["turns"] > 50 or o["ms"] > 60000:
            slow_games += 1
        if o.get("status") in ("draw_clock", "no_outcome"):
            stuck_games += 1

    for g in store.game_indices():
        traj = store.game(g)
        header = traj.header
        deck_of_seat = {i: p["deck"] for i, p in enumerate(header["players"])}
        seen_this_game: dict[str, int] = {}
        priority_turn_p = []
        for d in traj.decisions:
            m = d.get("m", "?")
            method_counts[m] += 1
            if m == "mulliganKeepHand":
                mull_total += 1
                if d.get("ret"):
                    mull_kept += 1
            obs = d.get("obs")
            if obs is None:
                continue
            turn = obs.get("glob", {}).get("turn", -1)
            ents = obs.get("ents", [])
            id_name = {e.get("e"): e.get("n") for e in ents}
            id_ctrl = {e.get("e"): e.get("c") for e in ents}
            for e in ents:
                name = e.get("n")
                if not name:
                    continue
                if e.get("tok"):
                    token_counts[name] += 1
                    continue
                zone = e.get("z")
                zone_card_counts[zone][name] += 1
                if name not in seen_this_game:
                    seen_this_game[name] = turn
                owner = e.get("c", -1)
                if 0 <= owner < len(deck_of_seat):
                    card_deck_counts[name][deck_of_seat[owner]] += 1
                if zone == "battlefield" and name in (
                    "Plains",
                    "Mountain",
                    "Forest",
                    "Island",
                    "Swamp",
                ):
                    key = f"{deck_of_seat.get(owner, '?')}-{name}"
                    if key not in land_drop_turn or turn < land_drop_turn[key][-1]:
                        land_drop_turn[key].append(turn)
            # combat counts from post-declaration obs
            atk_ents = [e for e in ents if "atk" in e]
            blk_ents = [e for e in ents if "blk" in e]
            if atk_ents or blk_ents:
                combat_games.add(g)
            if atk_ents:
                attacker_counts.append(len(atk_ents))
            if blk_ents:
                blocker_counts.append(len(blk_ents))
            if m == "chooseSpellAbilityToPlay":
                opts = d.get("opts") or []
                priority_options_counts.append(len(opts))
                priority_turn_p.append((turn, d.get("p", -1)))
                ret = d.get("ret")
                if ret is None:
                    pass_total += 1
                    if len(opts) > 1:
                        pass_with_options += 1
                else:
                    cast_total += 1
                    plans = ret if isinstance(ret, list) else [ret]
                    for plan in plans:
                        if isinstance(plan, dict):
                            host = plan.get("e")
                            if host is None:
                                continue
                            if host not in id_name:
                                host_not_in_obs += 1
                                continue
                            name = id_name[host]
                            cast_counts[name] += 1
                            owner = id_ctrl.get(host)
                            deck = deck_of_seat.get(owner)
                            if deck:
                                cast_by_deck[deck][name] += 1
        reps = Counter(priority_turn_p)
        max_priority_reps.append(max(reps.values()) if reps else 0)
        for name, turn in seen_this_game.items():
            card_first_turn[name].append(turn)

    # spot-check derived tables
    never_cast_rows = {deck: [] for deck in deck_mains}
    for deck, main in deck_mains.items():
        for card in main:
            if cast_by_deck[deck][card] == 0:
                never_cast_rows[deck].append((main[card], card))
    sideboard_cast_rows = []
    for deck, side in deck_sides.items():
        for card, qty in side.items():
            c = cast_by_deck[deck][card]
            if c > 0:
                sideboard_cast_rows.append((deck, qty, card, c))

    return {
        "store": store_dir.name,
        "n_games": len(outcomes),
        "outcomes": outcomes,
        "deck_records": dict(deck_records),
        "matchup_counts": {f"{a} vs {b}": v for (a, b), v in matchup_counts.items()},
        "profile_pair_counts": dict(profile_pair_counts),
        "method_counts": dict(method_counts),
        "cast_counts": dict(cast_counts),
        "cast_by_deck": {d: dict(c) for d, c in cast_by_deck.items()},
        "zone_card_counts": {z: dict(c) for z, c in zone_card_counts.items()},
        "card_first_turn": {
            n: {"mean": sum(v) / len(v), "n": len(v), "median": sorted(v)[len(v) // 2]}
            for n, v in card_first_turn.items()
        },
        "attacker_counts": attacker_counts,
        "blocker_counts": blocker_counts,
        "combat_games": len(combat_games),
        "priority_options_counts": priority_options_counts,
        "token_counts": dict(token_counts),
        "quality": {
            "pass_total": pass_total,
            "pass_with_options": pass_with_options,
            "cast_total": cast_total,
            "host_not_in_obs": host_not_in_obs,
            "mull_total": mull_total,
            "mull_kept": mull_kept,
            "slow_games": slow_games,
            "stuck_games": stuck_games,
            "max_priority_reps": max_priority_reps,
        },
        "never_cast_rows": never_cast_rows,
        "sideboard_cast_rows": sideboard_cast_rows,
    }


def histogram(values: list[int], bins: list[tuple[int, int]]) -> list[tuple[str, int, float]]:
    total = len(values)
    out = []
    for lo, hi in bins:
        n = sum(1 for v in values if lo <= v <= hi)
        out.append((f"{lo}-{hi}", n, n / max(total, 1)))
    return out


def html_report(data: dict, out: Path) -> None:
    n = data["n_games"]
    outcomes = data["outcomes"]
    turns = [o["turns"] for o in outcomes]
    ms = [o["ms"] for o in outcomes]
    decisive = sum(1 for o in outcomes if o["status"] == "won")
    draws = sum(1 for o in outcomes if o["status"] == "draw")
    draw_clock = sum(1 for o in outcomes if o.get("draw_clock"))

    # deck records table
    deck_rows = ""
    for d, rec in sorted(data["deck_records"].items()):
        avg_turns = sum(rec["turns"]) / len(rec["turns"])
        deck_rows += f"<tr><td>{d}</td><td>{rec['w']}/{rec['n']}</td><td>{_fmt_pct(rec['w'], rec['n'])}</td><td>{avg_turns:.1f}</td></tr>\n"

    # matchup table
    matchup_rows = ""
    for pair, rec in data["matchup_counts"].items():
        matchup_rows += f"<tr><td>{pair}</td><td>{rec['w']}/{rec['n']}</td><td>{_fmt_pct(rec['w'], rec['n'])}</td></tr>\n"

    # profile pairs
    profile_rows = ""
    for pair, c in (
        data["profile_pair_counts"].most_common()
        if hasattr(data["profile_pair_counts"], "most_common")
        else sorted(data["profile_pair_counts"].items(), key=lambda x: -x[1])
    ):
        profile_rows += (
            f"<tr><td>{' / '.join(pair)}</td><td>{c}</td><td>{_fmt_pct(c, n)}</td></tr>\n"
        )

    # top cast cards
    cast_rows = ""
    for name, c in Counter(data["cast_counts"]).most_common(25):
        avg_first = data["card_first_turn"].get(name, {}).get("mean", 0)
        cast_rows += f"<tr><td>{name}</td><td>{c}</td><td>{avg_first:.1f}</td></tr>\n"

    # least cast cards (among non-token cards that were observed at all)
    least_cast_rows = ""
    observed_cards = {name for z in data["zone_card_counts"].values() for name in z}
    for name, c in Counter(
        {
            name: data["cast_counts"].get(name, 0)
            for name in observed_cards
            if not name.endswith("'s Effect")
        }
    ).most_common()[:-26:-1]:
        avg_first = data["card_first_turn"].get(name, {}).get("mean", 0)
        least_cast_rows += f"<tr><td>{name}</td><td>{c}</td><td>{avg_first:.1f}</td></tr>\n"

    # top battlefield cards
    bf_rows = ""
    for name, c in Counter(data["zone_card_counts"].get("battlefield", {})).most_common(25):
        bf_rows += f"<tr><td>{name}</td><td>{c}</td></tr>\n"

    # deck-specific cast tables
    deck_cast_rows = ""
    for deck in sorted(data["cast_by_deck"]):
        cnt = Counter(data["cast_by_deck"][deck])
        deck_cast_rows += f"<h3>Casts for {deck}</h3><table><tr><th>Card</th><th>Casts</th></tr>\n"
        for name, c in cnt.most_common():
            deck_cast_rows += f"<tr><td>{name}</td><td>{c}</td></tr>\n"
        deck_cast_rows += "</table>\n"

    # never-cast spot checks
    never_cast_html = ""
    for deck in sorted(data["never_cast_rows"]):
        rows = data["never_cast_rows"][deck]
        never_cast_html += f"<h3>Maindeck cards never cast: {deck}</h3>"
        if rows:
            never_cast_html += "<table><tr><th>Qty</th><th>Card</th></tr>\n"
            for qty, card in sorted(rows, key=lambda x: -x[0]):
                never_cast_html += f"<tr><td>{qty}</td><td>{card}</td></tr>\n"
            never_cast_html += "</table>\n"
        else:
            never_cast_html += "<p>All maindeck cards were cast at least once.</p>\n"

    # sideboard casts
    side_rows = ""
    if data["sideboard_cast_rows"]:
        for deck, qty, card, casts in data["sideboard_cast_rows"]:
            side_rows += f"<tr><td>{deck}</td><td>{qty} {card}</td><td>{casts}</td></tr>\n"
    else:
        side_rows = "<tr><td colspan=3>No sideboard cards were cast.</td></tr>\n"

    # tokens
    token_rows = ""
    for name, c in Counter(data["token_counts"]).most_common(15):
        token_rows += f"<tr><td>{name}</td><td>{c}</td></tr>\n"

    # quality checks
    q = data["quality"]
    qual_rows = f"""
    <tr><td>Priority passes</td><td>{q["pass_total"]}</td></tr>
    <tr><td>Passes with &gt;1 option</td><td>{q["pass_with_options"]} ({_fmt_pct(q["pass_with_options"], q["pass_total"])})</td></tr>
    <tr><td>Priority casts</td><td>{q["cast_total"]}</td></tr>
    <tr><td>Chosen host not in obs</td><td>{q["host_not_in_obs"]}</td></tr>
    <tr><td>Mulligan decisions</td><td>{q["mull_total"]}</td></tr>
    <tr><td>Mulligans kept</td><td>{q["mull_kept"]} ({_fmt_pct(q["mull_kept"], q["mull_total"])})</td></tr>
    <tr><td>Slow/stuck games (&gt;50 turns or &gt;60s)</td><td>{q["slow_games"]}</td></tr>
    <tr><td>Draw-clock / no-outcome games</td><td>{q["stuck_games"]}</td></tr>
    <tr><td>Avg max same-turn priority reps</td><td>{sum(q["max_priority_reps"]) / max(len(q["max_priority_reps"]), 1):.1f}</td></tr>
    """

    # method table
    method_rows = ""
    total_methods = sum(data["method_counts"].values())
    for m, c in Counter(data["method_counts"]).most_common(25):
        method_rows += f"<tr><td>{m}</td><td>{c}</td><td>{_fmt_pct(c, total_methods)}</td></tr>\n"

    # turns histogram
    turn_hist = histogram(
        turns, [(0, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 100)]
    )
    turn_hist_rows = ""
    for label, count, frac in turn_hist:
        turn_hist_rows += f"<tr><td>{label}</td><td>{count}</td><td>{100 * frac:.1f}%</td></tr>\n"

    # options width histogram
    opt_hist = histogram(
        data["priority_options_counts"],
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 20), (21, 100)],
    )
    opt_hist_rows = ""
    for label, count, frac in opt_hist:
        opt_hist_rows += f"<tr><td>{label}</td><td>{count}</td><td>{100 * frac:.1f}%</td></tr>\n"

    # attacker/blocker summary
    atk = data["attacker_counts"]
    blk = data["blocker_counts"]
    combat_summary = f"""
    <p>Observations with attackers: {len(atk)}; with blockers: {len(blk)}; games touching combat: {data["combat_games"]} / {n}</p>
    <p>Avg attackers per observed attack: {sum(atk) / max(len(atk), 1):.2f}; avg blockers: {sum(blk) / max(len(blk), 1):.2f}</p>
    """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pauper BC two-deck report: {data["store"]}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2em; color: #222; }}
h1, h2 {{ border-bottom: 1px solid #ccc; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f4f4f4; }}
tr:nth-child(even) {{ background: #fafafa; }}
.num {{ text-align: right; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1em; }}
.card {{ background: #f9f9f9; border: 1px solid #ddd; padding: 1em; border-radius: 6px; }}
</style>
</head>
<body>
<h1>Pauper two-deck BC corpus report</h1>
<p>Store: <code>{data["store"]}</code> · Games: {n}</p>

<div class="summary">
  <div class="card"><strong>Decisive</strong><br>{decisive} ({_fmt_pct(decisive, n)})</div>
  <div class="card"><strong>Draws</strong><br>{draws}</div>
  <div class="card"><strong>Draw-clock hits</strong><br>{draw_clock}</div>
  <div class="card"><strong>Median turns</strong><br>{sorted(turns)[len(turns) // 2] if turns else "N/A"}</div>
  <div class="card"><strong>Mean turns</strong><br>{sum(turns) / len(turns):.1f}</div>
  <div class="card"><strong>Mean game ms</strong><br>{sum(ms) / len(ms):.0f}</div>
</div>

<h2>Deck records</h2>
<table>
<tr><th>Deck</th><th>Wins / Games</th><th>Win rate</th><th>Avg turns</th></tr>
{deck_rows}
</table>

<h2>Matchup</h2>
<table>
<tr><th>Pair</th><th>Wins / Games</th><th>Win rate (deck A)</th></tr>
{matchup_rows}
</table>

<h2>Profile pairs</h2>
<table>
<tr><th>Profile pair</th><th>Count</th><th>Share</th></tr>
{profile_rows}
</table>

<h2>Turns distribution</h2>
<table>
<tr><th>Turns</th><th>Games</th><th>Share</th></tr>
{turn_hist_rows}
</table>

<h2>Priority-option width distribution</h2>
<table>
<tr><th>Options</th><th>Windows</th><th>Share</th></tr>
{opt_hist_rows}
</table>

<h2>Most-cast cards</h2>
<table>
<tr><th>Card</th><th>Casts</th><th>Avg first seen turn</th></tr>
{cast_rows}
</table>

<h2>Least-cast observed cards</h2>
<table>
<tr><th>Card</th><th>Casts</th><th>Avg first seen turn</th></tr>
{least_cast_rows}
</table>

<h2>Most-observed on battlefield</h2>
<table>
<tr><th>Card</th><th>Observations</th></tr>
{bf_rows}
</table>

<h2>Deck-specific casts</h2>
{deck_cast_rows}

<h2>Maindeck never-cast spot check</h2>
{never_cast_html}

<h2>Sideboard cards cast</h2>
<table>
<tr><th>Deck</th><th>Card</th><th>Casts</th></tr>
{side_rows}
</table>

<h2>Tokens observed</h2>
<table>
<tr><th>Token</th><th>Observations</th></tr>
{token_rows}
</table>

<h2>Combat</h2>
{combat_summary}

<h2>Data-quality spot checks</h2>
<table>
<tr><th>Check</th><th>Value</th></tr>
{qual_rows}
</table>

<h2>Top decision methods</h2>
<table>
<tr><th>Method</th><th>Calls</th><th>Share</th></tr>
{method_rows}
</table>

</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[viz] report -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None, help="trajectory store directory")
    ap.add_argument("--out", default="data/training/pauper-bc-twodecks/report.html")
    a = ap.parse_args()
    store_dir = Path(a.store) if a.store else None
    if store_dir is None:
        # auto-detect latest pauper-bc-twodecks store
        candidates = sorted(Path("data/trajectories").glob("pauper-bc-twodecks-*"))
        if not candidates:
            raise SystemExit("no pauper-bc-twodecks store found")
        store_dir = candidates[-1]
    data = gather(store_dir)
    html_report(data, Path(a.out))


if __name__ == "__main__":
    main()
