"""Strict Trackio experiment-logging wrapper (M7 dashboard wiring).

if a call fails, it propagates so the
operator sees it immediately.  Values are sanitized/flattened before they
reach Trackio, but any value that cannot be made into a flat scalar metric is
rejected rather than silently coerced.

Usage:
    from anvil.trackio_logger import init_run, log, finish

    run = init_run(name="d6-runX", group="m7-c-bundle", config=vars(args))
    log({"loss": 0.5, "agree_honest": 0.62}, step=step)
    finish()

Environment conventions:
    TRACKIO_PROJECT=anvil          -> default project name (default "anvil")
    TRACKIO_NAME=...               -> fallback run name
    TRACKIO_GROUP=...              -> fallback group
    TRACKIO_PARENT=...             -> parent namespace for child runs

Local dashboard:
    uv run trackio show
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import trackio as _trackio


def _sanitize(value: Any) -> Any:
    """Make a value safe for a Trackio config dict."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v) for k, v in value.items()}
    return str(value)


def _flatten_metrics(
    d: Mapping[str, Any], prefix: str = ""
) -> dict[str, float | int | bool | None]:
    """Flatten a nested metric dict into Trackio-compatible scalar metrics.

    Uses Trackio's slash convention for metric groups (for example,
    ``rl/mean/kl_mu``). Rejects non-scalar leaves so bad data never slips
    through silently.
    """
    out: dict[str, float | int | bool | None] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(_flatten_metrics(v, key))
        elif isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, (int, float)):
            out[key] = v
        elif v is None:
            out[key] = None
        else:
            raise TypeError(f"trackio metric '{key}' has non-scalar type {type(v).__name__}: {v!r}")
    return out


def _run_name(suggested: str | None = None) -> str:
    """Build a run name from explicit value, env, or timestamp."""
    if suggested:
        return suggested
    env_name = os.environ.get("TRACKIO_NAME")
    if env_name:
        return env_name
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"run-{ts}"


def init_run(
    name: str | None = None,
    group: str | None = None,
    config: Mapping[str, Any] | None = None,
    project: str | None = None,
    resume: str = "allow",
    embed: bool = False,
) -> Any:
    """Initialize a Trackio run.

    Args:
        name: run name.  Falls back to TRACKIO_NAME, then a timestamp.
        group: group name.  Falls back to TRACKIO_GROUP, then TRACKIO_PARENT.
        config: flat-ish dict of hyperparameters / provenance.
        project: project name.  Falls back to TRACKIO_PROJECT, then "anvil".
        resume: passed to trackio.init (default "allow" for resumable loops).
        embed: whether to auto-embed the dashboard.  Default False for CLI runs.

    Returns:
        The Trackio Run object.
    """
    run_name = _run_name(name)
    group = group or os.environ.get("TRACKIO_GROUP") or os.environ.get("TRACKIO_PARENT")
    project = project or os.environ.get("TRACKIO_PROJECT", "anvil")

    kwargs: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "resume": resume,
        "embed": embed,
    }
    if group:
        kwargs["group"] = group
    if config:
        kwargs["config"] = {str(k): _sanitize(v) for k, v in config.items()}

    run = _trackio.init(**kwargs)
    return run


def log(metrics: Mapping[str, Any], step: int | None = None) -> None:
    """Log a dict of scalar metrics to the current run.

    Nested dicts are flattened.  Non-scalar values raise TypeError.
    """
    flat = _flatten_metrics(metrics)
    if step is not None:
        _trackio.log(flat, step=step)
    else:
        _trackio.log(flat)


def finish() -> None:
    """Close the current run."""
    _trackio.finish()


def alert(title: str, message: str, level: str = "warn") -> None:
    """Surface an alert as a Trackio metric row."""
    log({"alert": f"{level}: {title} - {message}"})


def child_env(
    parent: str,
    *,
    name: str | None = None,
    iteration: int | None = None,
    step_offset: int | None = None,
) -> dict[str, str]:
    """Environment updates for a subprocess in a parent campaign.

    ``name`` lets repeated subprocesses resume one logical Trackio run;
    ``iteration`` marks the campaign boundary within that run; and
    ``step_offset`` places subprocess-local steps on a continuous axis.
    """
    env = {
        "TRACKIO_PROJECT": os.environ.get("TRACKIO_PROJECT", "anvil"),
        "TRACKIO_GROUP": parent,
        "TRACKIO_PARENT": parent,
    }
    if name is not None:
        env["TRACKIO_NAME"] = name
    if iteration is not None:
        env["TRACKIO_ITERATION"] = str(iteration)
    if step_offset is not None:
        env["TRACKIO_STEP_OFFSET"] = str(step_offset)
    return env
