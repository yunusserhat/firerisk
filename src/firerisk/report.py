"""Export compatible experiment runs as paper-ready CSV and Markdown tables."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .metrics import compute_metrics, reliability_bins

REPORT_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_auroc",
    "nll",
    "brier",
    "ece",
    "high_risk_recall",
)


def _summary_paths(paths: Iterable[str | Path]) -> list[Path]:
    found: set[Path] = set()
    for pattern in paths:
        matches = glob.glob(str(pattern), recursive=True)
        if not matches and Path(pattern).exists():
            matches = [str(pattern)]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                found.update(p.resolve() for p in path.rglob("summary.json"))
            elif path.is_file():
                found.add(path.resolve())
    if not found:
        raise ValueError("No summary.json files were found in the supplied run paths.")
    return sorted(found)


def _run_record(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("config", {})
    model = dict(config.get("model", {}))
    if isinstance(summary.get("model"), dict):
        model.update(summary["model"])
    experiment = config.get("experiment", {})
    evaluation = summary.get("evaluation", {}).get("test", {})
    metrics = evaluation.get("raw", {})
    if not metrics:
        raise ValueError(f"{path} does not contain held-out test.raw metrics.")
    seed = summary.get("seed", experiment.get("seed"))
    if seed is None:
        raise ValueError(f"{path} does not record its training seed.")
    # A model family is not an experimental condition: changing learning rate,
    # epochs, LoRA settings or effective batch size creates a different recipe.
    training = dict(config.get("training", {}))
    training.pop("workers", None)  # Throughput setting, with worker RNG seeding.
    recipe = {
        "model": config.get("model", {}),
        "training": training,
        "world_size": summary.get("world_size", 1),
        "effective_batch_size": summary.get("effective_batch_size"),
        "smoke_test": bool(summary.get("smoke_test", False)),
    }
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    record: dict[str, Any] = {
        "run": str(path.parent),
        "protocol_hash": summary.get("protocol_hash"),
        "split_hash": summary.get("split_hash"),
        "recipe_hash": recipe_hash,
        "smoke_test": bool(summary.get("smoke_test", False)),
        "model": model.get(
            "name", summary.get("model") if isinstance(summary.get("model"), str) else "unknown"
        ),
        "mode": model.get("mode", "unknown"),
        "image_size": model.get("image_size"),
        "seed": int(seed),
        "best_epoch": summary.get("best_epoch"),
        "best_val_macro_f1": summary.get("best_val_macro_f1"),
        "training_seconds": summary.get("training_seconds"),
        "peak_gpu_memory_gb": summary.get("peak_gpu_memory_gb"),
        "n_test_examples": metrics.get("n_examples"),
    }
    for name in REPORT_METRICS:
        record[name] = (
            metrics.get(name)
            if name != "high_risk_recall"
            else (metrics.get("high_risk") or {}).get("recall")
        )
    bootstrap = evaluation.get("bootstrap", {}).get("metrics", {})
    for name in ("accuracy", "macro_f1"):
        interval = bootstrap.get(name, {})
        record[f"{name}_example_ci_lower"] = interval.get("lower")
        record[f"{name}_example_ci_upper"] = interval.get("upper")
    return record


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _statistics(values: Sequence[float]) -> dict[str, Any]:
    from scipy.stats import t

    array = np.asarray(values, dtype=float)
    n = len(array)
    if not n:
        return {"n": 0, "mean": None, "std": None, "ci_lower": None, "ci_upper": None}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if n > 1 else None
    half_width = float(t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else None
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_lower": mean - half_width if half_width is not None else None,
        "ci_upper": mean + half_width if half_width is not None else None,
    }


def _format_metric(value: dict[str, Any]) -> str:
    if value["mean"] is None:
        return "—"
    if value["std"] is None:
        return f"{value['mean']:.4f}"
    return f"{value['mean']:.4f} ± {value['std']:.4f}"


def aggregate_runs(
    run_paths: Iterable[str | Path],
    output_dir: str | Path,
    require_same_protocol: bool = True,
    include_smoke: bool = False,
) -> dict[str, Any]:
    """Aggregate held-out raw metrics across independent training seeds.

    Incompatible protocols are rejected by default. If explicitly allowed,
    their results remain in separate groups. A missing hash is always rejected
    because compatibility cannot be established. Smoke experiments are omitted
    unless requested, and repeated seeds within a group are rejected.
    """
    rows, excluded = [], []
    for path in _summary_paths(run_paths):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("smoke_test", False) and not include_smoke:
            excluded.append(str(path))
            continue
        if summary.get("status", "complete") != "complete":
            raise ValueError(f"{path} is not a completed training run.")
        row = _run_record(path, summary)
        if not row["protocol_hash"] or not row["split_hash"]:
            raise ValueError(
                f"{path} is missing protocol_hash or split_hash; comparison cannot verify compatibility."
            )
        rows.append(row)
    if not rows:
        raise ValueError("No complete research runs remain after excluding smoke tests.")
    protocols = {(row["protocol_hash"], row["split_hash"]) for row in rows}
    if require_same_protocol and len(protocols) > 1:
        raise ValueError(
            "Runs use incompatible protocols or splits. Compare separately, or explicitly allow mixed protocols."
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(
            row[name]
            for name in (
                "protocol_hash",
                "split_hash",
                "model",
                "mode",
                "image_size",
                "recipe_hash",
            )
        )
        grouped[key].append(row)
    aggregates, csv_aggregates = [], []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        seeds = [row["seed"] for row in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                f"Duplicate training seeds in model group {key[2:]}; repeated runs are not independent seeds."
            )
        statistics = {
            name: _statistics(
                [
                    float(row[name])
                    for row in group
                    if row[name] is not None and np.isfinite(row[name])
                ]
            )
            for name in REPORT_METRICS
        }
        metadata = dict(
            zip(("protocol_hash", "split_hash", "model", "mode", "image_size", "recipe_hash"), key)
        )
        metadata["smoke_test"] = group[0]["smoke_test"]
        aggregate = {
            **metadata,
            "n_seeds": len(seeds),
            "seeds": sorted(seeds),
            "metrics": statistics,
        }
        aggregates.append(aggregate)
        flat = {**metadata, "n_seeds": len(seeds), "seeds": ",".join(map(str, sorted(seeds)))}
        for name, stats in statistics.items():
            flat.update({f"{name}_{field}": value for field, value in stats.items()})
        csv_aggregates.append(flat)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "runs.csv", rows)
    _write_csv(output / "comparison.csv", csv_aggregates)
    result = {
        "schema_version": 1,
        "evaluation": "held-out test, raw probabilities",
        "confidence_interval": "95% Student-t interval for the mean across independent training seeds",
        "excluded_smoke_runs": excluded,
        "groups": aggregates,
    }
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown = [
        "Results use held-out test predictions before temperature scaling. Values are fractions.",
        "",
        "| Model | Mode | Size | Seeds | Accuracy | Macro F1 | Balanced accuracy | ECE | Macro F1 mean 95% CI | Protocol | Recipe | Smoke |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for group in aggregates:
        metrics = group["metrics"]
        ci = metrics["macro_f1"]
        interval = (
            f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]" if ci["ci_lower"] is not None else "—"
        )
        model_name = str(group["model"]).replace("|", "\\|")
        markdown.append(
            f"| {model_name} | {group['mode']} | {group['image_size']} | {group['n_seeds']} "
            f"| {_format_metric(metrics['accuracy'])} | {_format_metric(metrics['macro_f1'])} "
            f"| {_format_metric(metrics['balanced_accuracy'])} | {_format_metric(metrics['ece'])} "
            f"| {interval} | {str(group['protocol_hash'])[:12]} | {str(group['recipe_hash'])[:12]} | {'yes' if group['smoke_test'] else 'no'} |"
        )
    markdown.extend(
        [
            "",
            "The ± value is the sample standard deviation across seeds. Seed-mean confidence intervals use a Student-t distribution and require at least two independent seeds.",
            "Example-level bootstrap intervals in runs.csv describe finite test-sample uncertainty and are separate from uncertainty across training seeds.",
            "Compatible split and protocol hashes are required. Geographic independence is not established by a random image split.",
            "Different model or optimization settings remain in separate recipe groups; seed repeats must share one recipe.",
        ]
    )
    if excluded:
        markdown.append(f"{len(excluded)} smoke run(s) were excluded.")
    (output / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return result


def run_report(args: Any) -> dict[str, Any]:
    """Adapter for the main argparse CLI."""
    result = aggregate_runs(
        args.runs,
        args.output,
        require_same_protocol=not getattr(args, "allow_mixed_protocols", False),
        include_smoke=getattr(args, "include_smoke", False),
    )
    print(f"Wrote comparison.csv, comparison.md, comparison.json and runs.csv to {args.output}")
    return result


def plot_history(history_path: str | Path, output_dir: str | Path) -> Path:
    """Export validation learning curves without reading any held-out predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads(Path(history_path).read_text())
    if not history:
        raise ValueError("Training history is empty")
    epochs = [h["epoch"] for h in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Training CE")
    axes[0].set(xlabel="Epoch", ylabel="Training loss")
    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], label="Validation macro F1")
    axes[1].plot(epochs, [h["val_accuracy"] for h in history], label="Validation accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "learning_curves.png"
    figure.savefig(path, dpi=200)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path


def plot_predictions(
    prediction_path: str | Path, class_names: Sequence[str], output_dir: str | Path
) -> list[Path]:
    """Save confusion and reliability plots from an evaluation NPZ artifact."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires the optional matplotlib dependency.") from error
    with np.load(prediction_path, allow_pickle=False) as saved:
        if "labels" not in saved or "probabilities" not in saved:
            raise ValueError("Prediction NPZ must contain labels and probabilities.")
        labels, probabilities = saved["labels"], saved["probabilities"]
    metrics = compute_metrics(labels, probabilities, class_names)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(8, 7), layout="constrained")
    heatmap = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(heatmap, ax=axis, label="Examples")
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Held-out confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(
            column,
            row,
            str(value),
            ha="center",
            va="center",
            color="white" if value > matrix.max() / 2 else "black",
        )
    confusion_path = output / "confusion_matrix.png"
    figure.savefig(confusion_path, dpi=200)
    figure.savefig(confusion_path.with_suffix(".pdf"))
    plt.close(figure)
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, support, out=np.zeros_like(matrix, dtype=float), where=support > 0
    )
    figure, axis = plt.subplots(figsize=(8, 7), layout="constrained")
    heatmap = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(heatmap, ax=axis, label="Fraction within true class")
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Row-normalized confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for (row, column), value in np.ndenumerate(normalized):
        axis.text(
            column,
            row,
            f"{value:.0%}",
            ha="center",
            va="center",
            color="white" if value > 0.5 else "black",
        )
    normalized_path = output / "confusion_matrix_normalized.png"
    figure.savefig(normalized_path, dpi=200)
    figure.savefig(normalized_path.with_suffix(".pdf"))
    plt.close(figure)
    bins = [item for item in reliability_bins(labels, probabilities) if item["count"]]
    figure, (axis, counts_axis) = plt.subplots(
        2,
        1,
        figsize=(6, 7),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.plot(
        [item["confidence"] for item in bins],
        [item["accuracy"] for item in bins],
        "o-",
        label="Observed",
    )
    axis.set(
        ylabel="Empirical accuracy",
        ylim=(0, 1),
        xlim=(0, 1),
        title=f"Reliability diagram (ECE={metrics['ece']:.4f})",
    )
    axis.legend()
    counts_axis.bar(
        [(item["lower"] + item["upper"]) / 2 for item in bins],
        [item["count"] for item in bins],
        width=1 / 15,
    )
    counts_axis.set(xlabel="Predicted confidence", ylabel="Examples")
    reliability_path = output / "reliability.png"
    figure.savefig(reliability_path, dpi=200)
    figure.savefig(reliability_path.with_suffix(".pdf"))
    plt.close(figure)
    return [confusion_path, reliability_path]
