"""Real prediction artifacts exercise ensemble alignment and leakage controls."""

import json

import numpy as np
import pytest

from firerisk.ensemble import ensemble_runs
from firerisk.metrics import apply_temperature, compute_metrics
from firerisk.report import aggregate_runs

CLASSES = ["high", "low"]
VALIDATION_PROBS = np.array(
    [[0.9, 0.1], [0.8, 0.2], [0.35, 0.65], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4]]
)
OTHER_PROBS = np.array([[0.6, 0.4], [0.7, 0.3], [0.9, 0.1], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])


def _member(
    directory,
    name="model_a",
    *,
    probabilities=VALIDATION_PROBS,
    reorder=False,
    seed=42,
    test_labels=None,
):
    directory.mkdir()
    labels = np.array([0, 0, 0, 1, 1, 1])
    summary = {
        "status": "complete",
        "protocol_hash": "fixed-protocol",
        "split_hash": "fixed-split",
        "class_names": CLASSES,
        "seed": seed,
        "smoke_test": False,
        "config": {
            "model": {"name": name, "mode": "full", "image_size": 224},
            "training": {"lr_backbone": 1e-5},
            "experiment": {"seed": seed},
        },
        "evaluation": {},
    }
    permutation = np.array([4, 1, 5, 0, 3, 2]) if reorder else np.arange(6)
    for split, offset in [("validation", 0), ("test", 10)]:
        actual_labels = labels if split == "validation" or test_labels is None else test_labels
        values = probabilities if split == "validation" else probabilities[:, ::-1]
        np.savez_compressed(
            directory / f"{split}_predictions.npz",
            indices=(np.arange(6) + offset)[permutation],
            labels=actual_labels[permutation],
            probabilities=values[permutation],
            logits=np.log(values[permutation]),
        )
        summary["evaluation"][split] = {"raw": compute_metrics(actual_labels, values, CLASSES)}
    (directory / "summary.json").write_text(json.dumps(summary))
    return directory


def _change_summary(directory, **updates):
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    summary.update(updates)
    path.write_text(json.dumps(summary))


def _change_predictions(directory, field, value, split="test"):
    path = directory / f"{split}_predictions.npz"
    with np.load(path, allow_pickle=False) as arrays:
        prediction = {key: arrays[key].copy() for key in arrays.files}
    prediction[field] = value
    np.savez_compressed(path, **prediction)


def test_equal_probabilities_align_indices_and_produce_reportable_artifacts(tmp_path):
    first = _member(tmp_path / "first")
    second = _member(tmp_path / "second", "model_b", probabilities=OTHER_PROBS, reorder=True)
    output = tmp_path / "ensemble"
    result = ensemble_runs([first, second], output, bootstrap_samples=20)
    expected = (VALIDATION_PROBS[:, ::-1] + OTHER_PROBS[:, ::-1]) / 2
    with np.load(output / "test_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["indices"], np.arange(6) + 10)
        np.testing.assert_allclose(saved["probabilities"], expected)
        np.testing.assert_allclose(apply_temperature(saved["logits"]), expected)
    assert result["model"]["mode"] == "ensemble"
    assert result["status"] == "complete"
    assert result["calibration"]["fit_split"] == "validation"
    assert [member["weight"] for member in result["members"]] == [0.5, 0.5]
    assert result["evaluation"]["test"]["bootstrap"]["n_bootstrap"] == 20
    assert result["mean_member_validation_macro_f1"] == pytest.approx(5 / 6)
    assert result["best_val_macro_f1"] == 1
    assert (output / "calibration.json").is_file()
    report = aggregate_runs([output], tmp_path / "report")
    assert report["groups"][0]["mode"] == "ensemble"


def test_temperature_does_not_depend_on_test_labels(tmp_path):
    first = _member(tmp_path / "first")
    second = _member(tmp_path / "second", "model_b", probabilities=OTHER_PROBS)
    original = ensemble_runs([first, second], tmp_path / "original", bootstrap_samples=0)
    changed_labels = np.array([1, 1, 1, 0, 0, 0])
    changed_first = _member(tmp_path / "changed_first", test_labels=changed_labels)
    changed_second = _member(
        tmp_path / "changed_second",
        "model_b",
        probabilities=OTHER_PROBS,
        test_labels=changed_labels,
    )
    changed = ensemble_runs(
        [changed_first, changed_second], tmp_path / "changed", bootstrap_samples=0
    )
    assert original["calibration"]["temperature"] == pytest.approx(
        changed["calibration"]["temperature"]
    )
    assert (
        original["evaluation"]["test"]["raw"]["accuracy"]
        != changed["evaluation"]["test"]["raw"]["accuracy"]
    )


@pytest.mark.parametrize(
    "field,value",
    [("protocol_hash", "different"), ("split_hash", "different"), ("class_names", ["low", "high"])],
)
def test_incompatible_protocol_split_or_class_order_is_rejected(tmp_path, field, value):
    first = _member(tmp_path / "first")
    second = _member(tmp_path / "second", "model_b")
    _change_summary(second, **{field: value})
    with pytest.raises(ValueError, match="incompatible"):
        ensemble_runs([first, second], tmp_path / "output", bootstrap_samples=0)
    assert not (tmp_path / "output").exists()


def test_duplicate_indices_and_different_labels_are_rejected(tmp_path):
    first = _member(tmp_path / "first")
    second = _member(tmp_path / "second", "model_b")
    _change_predictions(second, "indices", np.array([10, 11, 12, 13, 14, 14]))
    with pytest.raises(ValueError, match="duplicate"):
        ensemble_runs([first, second], tmp_path / "duplicate", bootstrap_samples=0)
    _change_predictions(second, "indices", np.arange(6) + 10)
    _change_predictions(second, "labels", np.array([1, 0, 0, 1, 1, 1]))
    with pytest.raises(ValueError, match="different test labels"):
        ensemble_runs([first, second], tmp_path / "labels", bootstrap_samples=0)


def test_different_observations_and_validation_test_overlap_are_rejected(tmp_path):
    first = _member(tmp_path / "first")
    second = _member(tmp_path / "second", "model_b")
    _change_predictions(second, "indices", np.arange(6) + 20)
    with pytest.raises(ValueError, match="different test example indices"):
        ensemble_runs([first, second], tmp_path / "changed", bootstrap_samples=0)
    _change_predictions(first, "indices", np.array([0, 11, 12, 13, 14, 15]))
    with pytest.raises(ValueError, match="overlap"):
        ensemble_runs([first, second], tmp_path / "overlap", bootstrap_samples=0)


def test_incomplete_or_unevaluated_member_is_rejected(tmp_path):
    first = _member(tmp_path / "first")
    second = _member(tmp_path / "second", "model_b")
    _change_summary(second, status="running")
    with pytest.raises(ValueError, match="not completed"):
        ensemble_runs([first, second], tmp_path / "running", bootstrap_samples=0)
    _change_summary(second, status="complete")
    (second / "test_predictions.npz").unlink()
    with pytest.raises(FileNotFoundError, match="evaluate"):
        ensemble_runs([first, second], tmp_path / "missing", bootstrap_samples=0)


def test_smoke_flag_and_no_overwrite_guards_are_preserved(tmp_path):
    first = _member(tmp_path / "first")
    second = _member(tmp_path / "second", "model_b")
    _change_summary(first, smoke_test=True)
    result = ensemble_runs([first, second], tmp_path / "ensemble", bootstrap_samples=0)
    assert result["smoke_test"] is True
    with pytest.raises(FileExistsError):
        ensemble_runs([first, second], tmp_path / "ensemble", bootstrap_samples=0)
    with pytest.raises(ValueError, match="Duplicate member"):
        ensemble_runs([first, first], tmp_path / "duplicate", bootstrap_samples=0)
    with pytest.raises(ValueError, match="separate"):
        ensemble_runs([first, second], first, bootstrap_samples=0)


def test_member_order_and_training_seeds_do_not_change_scientific_recipe(tmp_path):
    first_a = _member(tmp_path / "first_a", seed=42)
    first_b = _member(tmp_path / "first_b", "model_b", probabilities=OTHER_PROBS, seed=42)
    second_a = _member(tmp_path / "second_a", seed=43)
    second_b = _member(tmp_path / "second_b", "model_b", probabilities=OTHER_PROBS, seed=43)
    first_output, second_output = tmp_path / "first_ensemble", tmp_path / "second_ensemble"
    ensemble_runs([first_a, first_b], first_output, bootstrap_samples=0, seed=42)
    ensemble_runs([second_b, second_a], second_output, bootstrap_samples=0, seed=43)
    result = aggregate_runs([first_output, second_output], tmp_path / "report")
    assert len(result["groups"]) == 1
    assert result["groups"][0]["n_seeds"] == 2


def test_bootstrap_reseeding_does_not_create_a_new_training_replication(tmp_path):
    first = _member(tmp_path / "first", seed=7)
    second = _member(tmp_path / "second", "model_b", seed=7)
    output_a, output_b = tmp_path / "ensemble_a", tmp_path / "ensemble_b"
    a = ensemble_runs([first, second], output_a, bootstrap_samples=10, seed=42)
    b = ensemble_runs([second, first], output_b, bootstrap_samples=10, seed=43)
    assert a["seed"] == b["seed"] == 7
    assert a["replication_id"] == b["replication_id"]
    assert a["bootstrap_seed"] == 42 and b["bootstrap_seed"] == 43
    with pytest.raises(ValueError, match="Duplicate training seeds"):
        aggregate_runs([output_a, output_b], tmp_path / "report")


def test_mixed_seed_replication_identifier_is_order_and_bootstrap_invariant(tmp_path):
    first = _member(tmp_path / "first", seed=7)
    second = _member(tmp_path / "second", "model_b", seed=8)
    a = ensemble_runs([first, second], tmp_path / "ensemble_a", bootstrap_samples=0, seed=42)
    b = ensemble_runs([second, first], tmp_path / "ensemble_b", bootstrap_samples=0, seed=43)
    assert a["seed"] == b["seed"]
    assert a["replication_id"] == b["replication_id"]
    assert sorted(a["member_training_seeds"]) == [7, 8]
    assert a["replication_members"] == b["replication_members"]
