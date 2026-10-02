"""Reconstruct manuscript validation tables and figures from saved predictions.

This command reads completed run artifacts only. It neither trains a model nor
loads a dataset or evaluates a checkpoint on the reserved test partition.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

CLASS_ORDER = [
    "very_low", "low", "moderate", "high", "very_high", "non-burnable", "water"
]
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high", "Nonburnable", "Water"]


def load_validation(run_dir):
    """Verify the saved raw validation metrics against observation predictions."""
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("status") != "complete" or summary.get("smoke_test"):
        raise ValueError(f"A completed research run is required at {run_dir}")
    classes = summary["class_names"]
    if len(classes) != len(CLASS_ORDER) or set(classes) != set(CLASS_ORDER):
        raise ValueError("Run class mapping does not match the declared FireRisk labels")
    raw = summary["evaluation"]["validation"]["raw"]
    with np.load(run_dir / "validation_predictions.npz", allow_pickle=False) as saved:
        labels = saved["labels"]
        logits = saved["logits"]
        indices = saved["indices"]
    if labels.shape != indices.shape or logits.shape != (len(labels), len(classes)):
        raise ValueError("Validation prediction dimensions are inconsistent")
    if not np.isfinite(logits).all() or len(np.unique(indices)) != len(indices):
        raise ValueError("Validation logits or observation indices are invalid")
    predictions = logits.argmax(axis=1)
    counts = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(counts, (labels, predictions), 1)
    if not np.array_equal(counts, raw["confusion_matrix"]):
        raise ValueError("Saved validation summary and prediction matrix do not agree")
    permutation = [classes.index(name) for name in CLASS_ORDER]
    counts = counts[np.ix_(permutation, permutation)]
    support = counts.sum(axis=1)
    predicted_support = counts.sum(axis=0)
    correct = counts.diagonal()
    precision = np.divide(correct, predicted_support, where=predicted_support > 0,
                          out=np.zeros(len(classes), dtype=float))
    recall = correct / support
    f1 = np.divide(2 * precision * recall, precision + recall, where=precision + recall > 0,
                   out=np.zeros(len(classes), dtype=float))
    metrics = {
        "accuracy": float(correct.sum() / support.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / support.sum()),
    }
    for metric, value in metrics.items():
        if not np.isclose(value, raw[metric], rtol=0, atol=1e-12):
            raise ValueError(f"Reconstructed {metric} disagrees with the run summary")
    manifest = json.loads((run_dir / "data_manifest.json").read_text())
    recorded_support = np.asarray(manifest["split_class_counts"]["validation"])
    manifest_permutation = [manifest["class_names"].index(name) for name in CLASS_ORDER]
    if not np.array_equal(support, recorded_support[manifest_permutation]):
        raise ValueError("Validation support differs from the recorded data manifest")
    if manifest["split_hash"] != summary["split_hash"] or len(labels) != raw["n_examples"]:
        raise ValueError("Validation data provenance is inconsistent")
    per_class = [
        {"class": name, "support": int(n), "precision": float(p),
         "recall": float(r), "f1": float(f)}
        for name, n, p, r, f in zip(CLASS_ORDER, support, precision, recall, f1)
    ]
    risk = counts[:5, :5]
    ordinal_distance = np.arange(5)[:, None] - np.arange(5)[None, :]
    risk_targets = int(support[:5].sum())
    group_analysis = {
        "risk_targets": risk_targets,
        "exact_risk_label_correct": int(risk.diagonal().sum()),
        "exact_risk_label_accuracy": float(risk.diagonal().sum() / risk_targets),
        "risk_targets_predicted_outside_risk_classes": int(counts[:5, 5:].sum()),
        "risk_targets_predicted_outside_risk_classes_rate":
            float(counts[:5, 5:].sum() / risk_targets),
        "risk_targets_underestimated_within_order": int(risk[ordinal_distance > 0].sum()),
        "risk_targets_at_least_two_levels_underestimated_within_order":
            int(risk[ordinal_distance >= 2].sum()),
        "within_one_level_correct_over_all_risk_targets":
            int(risk[np.abs(ordinal_distance) <= 1].sum()),
        "within_one_level_accuracy_over_all_risk_targets":
            float(risk[np.abs(ordinal_distance) <= 1].sum() / risk_targets),
        "nonburnable_and_water_targets": int(support[5:].sum()),
        "exact_nonburnable_or_water_label_correct": int(correct[5:].sum()),
        "exact_nonburnable_or_water_label_accuracy":
            float(correct[5:].sum() / support[5:].sum()),
        "denominator_note": "All risk rates use every true risk target, including predictions "
                            "of water or nonburnable. Those predictions have no ordinal rank "
                            "and are separately counted as outside the risk classes.",
    }
    off_diagonal = counts.copy()
    np.fill_diagonal(off_diagonal, 0)
    errors = []
    for flat in np.argsort(off_diagonal, axis=None)[::-1][:10]:
        true, predicted = np.unravel_index(flat, counts.shape)
        errors.append({"true": CLASS_ORDER[true], "predicted": CLASS_ORDER[predicted],
                       "count": int(counts[true, predicted]),
                       "fraction_of_true_class": float(counts[true, predicted] / support[true])})
    result = {
        "seed": summary["seed"],
        "best_epoch": summary["best_epoch"],
        "model_revision": summary["model"]["resolved_revision"],
        "trainable_parameters": summary["model"]["trainable_parameters"],
        "metrics": metrics,
        "per_class": per_class,
        "confusion_counts": counts.tolist(),
        "confusion_fraction_of_true_class": (counts / support[:, None]).tolist(),
        "group_analysis": group_analysis,
        "largest_off_diagonal_counts": errors,
    }
    return summary, result, indices, labels


def plot_comparison(runs, output_dir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 10, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.85), layout="constrained")
    for axis, (name, result) in zip(axes, runs.items()):
        matrix = np.asarray(result["confusion_fraction_of_true_class"])
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="cividis")
        axis.set_title(name, pad=8)
        axis.set_xticks(range(len(CLASS_ORDER)), labels=CLASS_LABELS,
                        rotation=45, ha="right", rotation_mode="anchor")
        axis.set_yticks(range(len(CLASS_ORDER)), labels=CLASS_LABELS)
        axis.set_xlabel("Predicted category")
        axis.set_ylabel("True category")
        axis.tick_params(length=0)
        for row in range(len(CLASS_ORDER)):
            for column in range(len(CLASS_ORDER)):
                value = matrix[row, column]
                axis.text(column, row, f"{100 * value:.1f}", ha="center", va="center",
                          color="white" if value < 0.5 else "black", fontsize=6.4)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.80, fraction=0.025, pad=0.025)
    colorbar.formatter = PercentFormatter(xmax=1, decimals=0)
    colorbar.update_ticks()
    colorbar.set_label("Share of the true category")
    figure.savefig(output_dir / "validation-confusion-comparison.pdf", metadata={
        "Title": "FireRisk validation confusion matrices", "CreationDate": None})
    figure.savefig(output_dir / "validation-confusion-comparison.png", dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-run", required=True, type=Path)
    parser.add_argument("--full-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Directory for PDF, PNG and JSON")
    args = parser.parse_args()
    frozen_summary, frozen, frozen_indices, frozen_labels = load_validation(args.frozen_run)
    full_summary, full, full_indices, full_labels = load_validation(args.full_run)
    if frozen_summary["model"]["mode"] != "linear" or full_summary["model"]["mode"] != "full":
        raise ValueError("Expected a frozen encoder run and a fully adapted run")
    for key in ["name", "resolved_revision", "image_size"]:
        if frozen_summary["model"][key] != full_summary["model"][key]:
            raise ValueError(f"The two encoder recipes differ in {key}")
    for key in ["split_hash", "protocol_hash", "class_names"]:
        if frozen_summary[key] != full_summary[key]:
            raise ValueError(f"The two runs have incompatible {key}")
    frozen_order = np.argsort(frozen_indices)
    full_order = np.argsort(full_indices)
    if not (np.array_equal(frozen_indices[frozen_order], full_indices[full_order])
            and np.array_equal(frozen_labels[frozen_order], full_labels[full_order])):
        raise ValueError("The two runs do not contain the same validation observations")
    if [p["support"] for p in frozen["per_class"]] != [p["support"] for p in full["per_class"]]:
        raise ValueError("The two runs have different validation class supports")
    analysis = {
        "schema_version": 1,
        "partition": "validation",
        "n_examples": len(frozen_indices),
        "split_hash": frozen_summary["split_hash"],
        "protocol_hash": frozen_summary["protocol_hash"],
        "class_order": CLASS_ORDER,
        "interpretation": "Single training seed. Checkpoints selected on this validation "
                          "partition. Not independent test results or a controlled adaptation ablation.",
        "runs": {"Frozen encoder": frozen, "Full adaptation": full},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    plot_comparison(analysis["runs"], args.output)
    print(f"Verified {len(frozen_indices)} validation observations and saved {args.output}")


if __name__ == "__main__":
    main()
