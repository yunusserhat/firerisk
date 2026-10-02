import json

import numpy as np
import pytest

from firerisk.metrics import apply_temperature, bootstrap_metrics, compute_metrics, fit_temperature
from firerisk.report import aggregate_runs

CLASSES = ["high", "low", "moderate", "non-burnable", "very_high", "very_low", "water"]


def test_perfect_predictions_have_expected_metrics():
    labels = np.arange(len(CLASSES))
    result = compute_metrics(labels, np.eye(len(CLASSES)), CLASSES)
    assert result["accuracy"] == result["balanced_accuracy"] == result["macro_f1"] == 1
    assert result["ece"] == result["brier"] == result["nll"] == 0
    assert result["macro_auroc"] == 1
    assert result["ordinal"]["n_ordered_pairs"] == 5
    assert result["ordinal"]["mae"] == 0
    assert result["ordinal"]["quadratic_weighted_kappa"] == 1
    json.dumps(result, allow_nan=False)


def test_calibration_values_match_hand_calculation():
    result = compute_metrics(np.array([0, 1]), np.array([[0.8, 0.2], [0.4, 0.6]]), ["a", "b"])
    assert result["brier"] == pytest.approx(0.2)
    assert result["nll"] == pytest.approx(-(np.log(0.8) + np.log(0.6)) / 2)
    assert result["ece"] == pytest.approx(0.3)
    assert result["macro_auroc"] == 1


def test_absent_classes_are_explicit_and_auc_never_nan():
    labels = np.array([0, 0])
    probabilities = np.zeros((2, len(CLASSES)))
    probabilities[:, 0] = 1
    result = compute_metrics(labels, probabilities, CLASSES)
    assert result["balanced_accuracy"] == 1
    assert result["macro_f1"] == pytest.approx(1 / len(CLASSES))
    assert result["macro_auroc"] is None
    assert result["macro_auroc_classes"] == []
    assert result["per_class"]["water"]["support"] == 0
    json.dumps(result, allow_nan=False)


def test_ordinal_errors_do_not_rank_water_or_nonburnable():
    labels = np.array([0, 0, 6, 3])
    predictions = np.array([5, 6, 0, 4])
    probabilities = np.eye(len(CLASSES))[predictions]
    ordinal = compute_metrics(labels, probabilities, CLASSES)["ordinal"]
    assert ordinal["n_risk_targets"] == 2
    assert ordinal["n_ordered_pairs"] == 1
    assert ordinal["risk_predictions_outside_order"] == 1
    assert ordinal["coverage"] == 0.5
    assert ordinal["mae"] == 3  # high -> very_low is three levels.
    assert ordinal["severe_underestimation_rate"] == 1


def test_high_risk_group_uses_both_high_and_very_high():
    labels = np.array([0, 4, 1, 6])
    predictions = np.array([4, 0, 4, 6])
    result = compute_metrics(labels, np.eye(len(CLASSES))[predictions], CLASSES)
    assert result["high_risk"]["recall"] == 1
    assert result["high_risk"]["precision"] == pytest.approx(2 / 3)


def test_bootstrap_is_reproducible_and_perfect_intervals_collapse():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.eye(2)[labels]
    result = bootstrap_metrics(labels, probabilities, ["a", "b"], n_bootstrap=30, seed=7)
    assert result == bootstrap_metrics(labels, probabilities, ["a", "b"], n_bootstrap=30, seed=7)
    assert result["metrics"]["macro_f1"] == {"estimate": 1.0, "lower": 1.0, "upper": 1.0}
    assert result["resampling_unit"] == "example"


def test_temperature_scaling_reduces_overconfidence_without_changing_predictions():
    logits = np.tile(np.array([6.0, 0.0]), (10, 1))
    labels = np.array([0] * 8 + [1] * 2)
    temperature = fit_temperature(logits, labels)
    raw = apply_temperature(logits)
    calibrated = apply_temperature(logits, temperature)
    assert temperature > 1
    assert (
        compute_metrics(labels, calibrated, ["a", "b"])["nll"]
        < compute_metrics(labels, raw, ["a", "b"])["nll"]
    )
    np.testing.assert_array_equal(raw.argmax(1), calibrated.argmax(1))
    np.testing.assert_allclose(apply_temperature(np.array([[10000.0, 9999.0]])).sum(1), 1)


@pytest.mark.parametrize(
    "probabilities", [np.array([[0.2, 0.2]]), np.array([[np.nan, 0.5]]), np.array([[1.2, -0.2]])]
)
def test_invalid_probabilities_fail_before_metrics(probabilities):
    with pytest.raises(ValueError):
        compute_metrics(np.array([0]), probabilities, ["a", "b"])


def _write_summary(tmp_path, name, seed, protocol="same", smoke=False):
    directory = tmp_path / name
    directory.mkdir()
    summary = {
        "config": {"model": {"name": "siglip2", "mode": "full", "image_size": 224}},
        "protocol_hash": protocol,
        "split_hash": "fixed-split",
        "seed": seed,
        "smoke_test": smoke,
        "evaluation": {
            "test": {
                "raw": {
                    "macro_f1": 0.6 + seed / 100,
                    "accuracy": 0.7,
                    "balanced_accuracy": 0.5,
                    "ece": 0.1,
                    "n_examples": 20,
                }
            }
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    return directory


def test_report_aggregates_independent_seeds_and_excludes_smoke(tmp_path):
    runs = [
        _write_summary(tmp_path, "a", 1),
        _write_summary(tmp_path, "b", 2),
        _write_summary(tmp_path, "smoke", 3, smoke=True),
    ]
    result = aggregate_runs(runs, tmp_path / "report")
    assert len(result["groups"]) == 1
    assert result["groups"][0]["n_seeds"] == 2
    assert result["groups"][0]["metrics"]["macro_f1"]["mean"] == pytest.approx(0.615)
    assert len(result["excluded_smoke_runs"]) == 1
    assert (tmp_path / "report" / "comparison.csv").exists()
    assert (tmp_path / "report" / "comparison.md").exists()


def test_report_rejects_incompatible_splits_and_duplicate_seeds(tmp_path):
    a = _write_summary(tmp_path, "a", 1)
    incompatible = _write_summary(tmp_path, "b", 2, protocol="different")
    with pytest.raises(ValueError, match="incompatible"):
        aggregate_runs([a, incompatible], tmp_path / "report")
    separated = aggregate_runs([a, incompatible], tmp_path / "allowed", require_same_protocol=False)
    assert len(separated["groups"]) == 2
    duplicate = _write_summary(tmp_path, "duplicate", 1)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_runs([a, duplicate], tmp_path / "duplicate-report")
