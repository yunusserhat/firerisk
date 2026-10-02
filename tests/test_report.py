"""Checks that result tables cannot silently pool incompatible experiments."""

import json

import pytest

from firerisk.report import aggregate_runs


def _run(
    directory, seed, *, lr=1e-5, status="complete", protocol="protocol", split="split", smoke=False
):
    directory.mkdir()
    summary = {
        "status": status,
        "protocol_hash": protocol,
        "split_hash": split,
        "seed": seed,
        "smoke_test": smoke,
        "config": {
            "model": {"name": "siglip2", "mode": "full", "image_size": 224, "revision": "pinned"},
            "training": {"lr_backbone": lr, "workers": 2, "batch_size": 16},
        },
        "evaluation": {"test": {"raw": {"accuracy": 0.7, "macro_f1": 0.5}}},
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    return directory


def test_learning_rate_sweep_is_not_pooled_as_seed_repetitions(tmp_path):
    baseline = _run(tmp_path / "baseline", 42)
    tuned = _run(tmp_path / "tuned", 43, lr=2e-5)
    result = aggregate_runs([baseline, tuned], tmp_path / "comparison")
    assert len(result["groups"]) == 2
    assert all(group["n_seeds"] == 1 for group in result["groups"])
    assert all(group["metrics"]["macro_f1"]["ci_lower"] is None for group in result["groups"])


def test_worker_count_does_not_define_a_new_scientific_recipe(tmp_path):
    first = _run(tmp_path / "first", 42)
    second = _run(tmp_path / "second", 43)
    path = second / "summary.json"
    summary = json.loads(path.read_text())
    summary["config"]["training"]["workers"] = 4
    path.write_text(json.dumps(summary))
    result = aggregate_runs([first, second], tmp_path / "comparison")
    assert result["groups"][0]["n_seeds"] == 2


def test_same_protocol_with_a_changed_split_is_rejected(tmp_path):
    first = _run(tmp_path / "first", 42)
    second = _run(tmp_path / "second", 43, split="different")
    with pytest.raises(ValueError, match="incompatible"):
        aggregate_runs([first, second], tmp_path / "comparison")


def test_active_run_and_unverifiable_hashes_are_rejected(tmp_path):
    active = _run(tmp_path / "active", 42, status="running")
    with pytest.raises(ValueError, match="completed"):
        aggregate_runs([active], tmp_path / "comparison")
    missing = _run(tmp_path / "missing", 43, protocol=None)
    with pytest.raises(ValueError, match="missing protocol_hash"):
        aggregate_runs([missing], tmp_path / "comparison")


def test_smoke_results_cannot_pool_with_research_runs_when_explicitly_included(tmp_path):
    full = _run(tmp_path / "full", 42)
    smoke = _run(tmp_path / "smoke", 42, smoke=True)
    result = aggregate_runs([full, smoke], tmp_path / "comparison", include_smoke=True)
    assert len(result["groups"]) == 2
