"""Classification, calibration and uncertainty metrics for FireRisk experiments.

All rates are fractions, rather than percentages. Macro F1 includes every class
in the declared label schema. Ordinal errors are defined only for predictions
and targets in the five ordered hazard classes; coverage is reported alongside
them so predictions of water/non-burnable cannot silently disappear.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

RISK_ORDER = ("very_low", "low", "moderate", "high", "very_high")


def _validate_labels(labels: Any, n_rows: int, n_classes: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != n_rows or not len(values):
        raise ValueError("Labels must be a nonempty vector matching prediction rows.")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Labels must contain integer class indices.")
    if np.any(values < 0) or np.any(values >= n_classes):
        raise ValueError("A label falls outside the declared class schema.")
    return values.astype(np.int64, copy=False)


def _validate_probabilities(
    labels: Any, probabilities: Any, class_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float64)
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be nonempty and unique.")
    if probs.ndim != 2 or probs.shape[1] != len(class_names):
        raise ValueError("Probabilities must have shape (examples, declared classes).")
    y = _validate_labels(labels, probs.shape[0], probs.shape[1])
    if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Probabilities must be finite values in [0, 1].")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("Each probability row must sum to one.")
    return y, probs


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0
    )


def reliability_bins(labels: Any, probabilities: Any, n_bins: int = 15) -> list[dict[str, Any]]:
    """Equal-width bins of maximum predicted confidence, including empty bins."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 1:
        raise ValueError("Probabilities must be a two-dimensional array.")
    y = _validate_labels(labels, len(probs), probs.shape[1])
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    assignments = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    bins = []
    for idx in range(n_bins):
        selected = assignments == idx
        count = int(selected.sum())
        bins.append(
            {
                "lower": idx / n_bins,
                "upper": (idx + 1) / n_bins,
                "count": count,
                "accuracy": float(correct[selected].mean()) if count else None,
                "confidence": float(confidence[selected].mean()) if count else None,
            }
        )
    return bins


def _core_metrics(
    y: np.ndarray, probs: np.ndarray
) -> tuple[dict[str, float], np.ndarray, dict[str, np.ndarray]]:
    k = probs.shape[1]
    predictions = probs.argmax(axis=1)
    matrix = np.bincount(y * k + predictions, minlength=k * k).reshape(k, k)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    confidence = probs.max(axis=1)
    correct = predictions == y
    assignments = np.minimum((confidence * 15).astype(int), 14)
    counts = np.bincount(assignments, minlength=15)
    bin_accuracy = _safe_divide(np.bincount(assignments, weights=correct, minlength=15), counts)
    bin_confidence = _safe_divide(
        np.bincount(assignments, weights=confidence, minlength=15), counts
    )
    # sum(p^2) - 2 p(y) + 1 avoids materializing a one-hot target matrix.
    metrics = {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1)).mean()),
        "brier": float((np.square(probs).sum(axis=1) - 2 * probs[np.arange(len(y)), y] + 1).mean()),
        "ece": float((np.abs(bin_accuracy - bin_confidence) * counts).sum() / len(y)),
    }
    return metrics, matrix, {"precision": precision, "recall": recall, "f1": f1, "support": support}


def _canonical(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _high_risk_metrics(
    y: np.ndarray, probs: np.ndarray, class_names: Sequence[str]
) -> dict[str, Any] | None:
    risk_ids = [
        idx for idx, name in enumerate(class_names) if _canonical(name) in ("high", "very_high")
    ]
    if not risk_ids:
        return None
    actual = np.isin(y, risk_ids)
    predicted = np.isin(probs.argmax(axis=1), risk_ids)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    score = probs[:, risk_ids].sum(axis=1)
    return {
        "class_names": [class_names[idx] for idx in risk_ids],
        "support": int(actual.sum()),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "auroc": float(roc_auc_score(actual, score)) if np.unique(actual).size == 2 else None,
    }


def _ordinal_metrics(
    y: np.ndarray, probs: np.ndarray, class_names: Sequence[str]
) -> dict[str, Any]:
    ranks = np.array(
        [
            RISK_ORDER.index(_canonical(name)) if _canonical(name) in RISK_ORDER else -1
            for name in class_names
        ]
    )
    true_ranks = ranks[y]
    predicted_ranks = ranks[probs.argmax(axis=1)]
    risk_targets = true_ranks >= 0
    valid = risk_targets & (predicted_ranks >= 0)
    count = int(valid.sum())
    target_count = int(risk_targets.sum())
    result: dict[str, Any] = {
        "class_order": list(RISK_ORDER),
        "n_risk_targets": target_count,
        "n_ordered_pairs": count,
        "risk_predictions_outside_order": int((risk_targets & ~valid).sum()),
        "coverage": count / target_count if target_count else None,
        "mae": None,
        "within_one_level_accuracy": None,
        "severe_underestimation_rate": None,
        "quadratic_weighted_kappa": None,
    }
    if count:
        truth, prediction = true_ranks[valid], predicted_ranks[valid]
        difference = prediction - truth
        result.update(
            mae=float(np.abs(difference).mean()),
            within_one_level_accuracy=float((np.abs(difference) <= 1).mean()),
            severe_underestimation_rate=float((difference <= -2).mean()),
        )
        # A constant identical pair has undefined chance correction.
        if np.unique(np.concatenate([truth, prediction])).size > 1:
            kappa = cohen_kappa_score(truth, prediction, labels=list(range(5)), weights="quadratic")
            result["quadratic_weighted_kappa"] = float(kappa) if np.isfinite(kappa) else None
    return result


def compute_metrics(labels: Any, probabilities: Any, class_names: Sequence[str]) -> dict[str, Any]:
    """Return JSON-safe classification and calibration results.

    AUROC is computed one-vs-rest for classes with both positive and negative
    examples. ``macro_auroc_classes`` explicitly records the averaging domain.
    NLL clips probabilities to 1e-15; Brier is the sum of squared class errors.
    """
    y, probs = _validate_probabilities(labels, probabilities, class_names)
    metrics, matrix, per_class = _core_metrics(y, probs)
    class_results = {}
    auc_values, auc_names = [], []
    for idx, name in enumerate(class_names):
        target = y == idx
        auc = float(roc_auc_score(target, probs[:, idx])) if np.unique(target).size == 2 else None
        if auc is not None:
            auc_values.append(auc)
            auc_names.append(name)
        class_results[name] = {
            "precision": float(per_class["precision"][idx]),
            "recall": float(per_class["recall"][idx]),
            "f1": float(per_class["f1"][idx]),
            "support": int(per_class["support"][idx]),
            "auroc": auc,
        }
    return {
        **metrics,
        "n_examples": int(len(y)),
        "class_names": list(class_names),
        "per_class": class_results,
        "confusion_matrix": matrix.tolist(),
        "macro_auroc": float(np.mean(auc_values)) if auc_values else None,
        "macro_auroc_classes": auc_names,
        "ece_bins": 15,
        "brier_definition": "sum of squared errors across classes",
        "high_risk": _high_risk_metrics(y, probs, class_names),
        "ordinal": _ordinal_metrics(y, probs, class_names),
    }


def bootstrap_metrics(
    labels: Any,
    probabilities: Any,
    class_names: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Stratified percentile 95% CIs for the principal scalar metrics.

    Fixed class counts condition the interval on the observed class mixture.
    These are example-level intervals; spatially dependent images require a
    geographic block bootstrap, which cannot be inferred from RGB pixels.
    """
    y, probs = _validate_probabilities(labels, probabilities, class_names)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least one.")
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == label) for label in np.unique(y)]
    estimates, _, _ = _core_metrics(y, probs)
    samples: dict[str, list[float]] = {name: [] for name in estimates}
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            [rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata]
        )
        values, _, _ = _core_metrics(y[indices], probs[indices])
        for name, value in values.items():
            samples[name].append(value)
    return {
        "method": "stratified percentile bootstrap",
        "resampling_unit": "example",
        "confidence_level": 0.95,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "metrics": {
            name: {
                "estimate": estimates[name],
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for name, values in samples.items()
        },
    }


def apply_temperature(logits: Any, temperature: float = 1.0) -> np.ndarray:
    """Return a stable NumPy softmax after scaling logits by temperature."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("Logits must be a finite array with shape (examples, classes).")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    shifted = values / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def fit_temperature(logits: Any, labels: Any) -> float:
    """Fit one scalar by validation NLL, bounded to [0.05, 20].

    The caller must supply validation logits only and apply this unchanged to
    test predictions. The fitting function cannot infer a dataset's provenance.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Calibration logits must be a finite (examples, classes) array.")
    y = _validate_labels(labels, len(values), values.shape[1])

    def objective(log_temperature: float) -> float:
        scaled = values / np.exp(log_temperature)
        return float((logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]).mean())

    optimum = minimize_scalar(objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    if not optimum.success or objective(float(optimum.x)) > objective(0.0):
        return 1.0
    return float(np.exp(optimum.x))
