"""Trackio logger metric-shape and self-play artifact tests."""

from types import SimpleNamespace

import pytest

from anvil.trackio_logger import _flatten_metrics, child_env


def test_flatten_metrics_uses_trackio_slash_groups():
    assert _flatten_metrics(
        {
            "iteration": 3,
            "census": {"bridged": 47_609, "casts_per_game": 32.53},
            "rl": {"mean": {"kl_mu": 0.0024}},
        }
    ) == {
        "iteration": 3,
        "census/bridged": 47_609,
        "census/casts_per_game": 32.53,
        "rl/mean/kl_mu": 0.0024,
    }


def test_flatten_metrics_normalizes_bools_and_rejects_non_scalars():
    assert _flatten_metrics({"healthy": True, "optional": None}) == {
        "healthy": 1,
        "optional": None,
    }
    with pytest.raises(TypeError, match="non-scalar type str"):
        _flatten_metrics({"status": "healthy"})


def test_child_env_names_one_logical_run_and_marks_iteration(monkeypatch):
    monkeypatch.setenv("TRACKIO_PROJECT", "test-project")
    assert child_env("campaign", name="campaign-train", iteration=4, step_offset=8_000) == {
        "TRACKIO_PROJECT": "test-project",
        "TRACKIO_GROUP": "campaign",
        "TRACKIO_PARENT": "campaign",
        "TRACKIO_NAME": "campaign-train",
        "TRACKIO_ITERATION": "4",
        "TRACKIO_STEP_OFFSET": "8000",
    }


def test_iteration_artifacts_share_stable_versioned_lineages(tmp_path, monkeypatch):
    from anvil.training import selfplay

    out = tmp_path / "campaign"
    it_dir = out / "iter-004"
    train_dir = it_dir / "train"
    train_dir.mkdir(parents=True)
    (out / "loop_config.json").write_text("{}")
    (it_dir / "server.log").write_text("served")
    (it_dir / "arms-report.json").write_text("{}")
    (it_dir / "drill-eval.json").write_text("{}")
    ckpt = train_dir / "last.pt"
    ckpt.write_bytes(b"model-v4")

    logged = []

    def fake_log_artifact(artifact, aliases=None):
        logged.append((artifact, aliases))

    monkeypatch.setattr(selfplay._trackio, "log_artifact", fake_log_artifact)
    selfplay._log_iteration_artifacts(
        SimpleNamespace(name="campaign"), out, it_dir, 4, ckpt, {"steps": 1234}
    )

    assert [(artifact.name, aliases) for artifact, aliases in logged] == [
        ("campaign-checkpoint", ["iter-004"]),
        ("campaign-iteration-provenance", ["iter-004"]),
    ]
    assert all(artifact.metadata["iteration"] == 4 for artifact, _ in logged)
    assert all(artifact.metadata["accepted"] is True for artifact, _ in logged)
    assert logged[0][0].metadata["learner_step"] == 1234
    assert [name for _, name in logged[0][0]._pending_files] == [
        "last.pt",
        "iteration.json",
    ]
    assert [name for _, name in logged[1][0]._pending_files] == [
        "iteration.json",
        "loop_config.json",
        "server.log",
        "arms-report.json",
        "drill-eval.json",
    ]
    assert (it_dir / "artifact-iteration.json").read_text().endswith("\n")
