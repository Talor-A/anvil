"""Path helpers for timestamped data directories.

Every top-level directory created under data/runs, data/training, or
data/trajectories should start with a YYYYMMDD-HHMMSS stamp so that a plain
``ls`` is chronological.  Internal subdirectories (iter-000, workers, etc.)
do not need stamps; their order is implied by the parent.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

STAMP_FMT = "%Y%m%d-%H%M%S"
_STAMP_RE = re.compile(r"^\d{8}-\d{6}")


def stamp_name(name: str, dt: _dt.datetime | None = None) -> str:
    """Return ``YYYYMMDD-HHMMSS-<name>``.

    The stamp uses local time; lexicographic order equals chronological order
    for runs created on the same machine.
    """
    ts = (dt or _dt.datetime.now()).strftime(STAMP_FMT)
    return f"{ts}-{name}"


def make_stamp_dir(parent: Path | str, name: str, dt: _dt.datetime | None = None) -> Path:
    """Create ``parent/YYYYMMDD-HHMMSS-name`` and return its path."""
    d = Path(parent) / stamp_name(name, dt)
    d.mkdir(parents=True, exist_ok=True)
    return d


def has_stamp(name: str) -> bool:
    """True if *name* already begins with a YYYYMMDD-HHMMSS stamp."""
    return bool(_STAMP_RE.match(name))


def parse_stamp(name: str) -> str | None:
    """Return the leading timestamp (or None)."""
    m = _STAMP_RE.match(name)
    return m.group(0) if m else None
