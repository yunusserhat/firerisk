"""Offline, fixed-weight ensembling of compatible held-out prediction artifacts.

The ensemble rule is declared before inspecting test results. No model weights,
GPU, dataset downloads or test-label fitting are needed. A scalar temperature
may be fitted only to the averaged validation probabilities.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .metrics import apply_temperature, bootstrap_metrics, compute_metrics, fit_temperature


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as handle:
        temp = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temp, path)


def _atomic_predictions(path: Path, prediction: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".npz") as handle:
        temp = Path(handle.name)
        np.savez_compressed(handle, **prediction)
    os.replace(temp, path)


def _read_predictions(path: Path, class_names: Sequence[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prediction artifact: {path}; evaluate the completed member first."
        )
    with np.load(path, allow_pickle=False) as arrays:
        required = {"indices", "labels", "logits", "probabilities"}
        if not required.issubset(arrays.files):
            raise ValueError(f"{path} must contain {sorted(required)}.")
        prediction = {key: arrays[key].copy() for key in required}
    indices = prediction["indices"]
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{path} indices must be a one-dimensional integer array.")
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{path} contains invalid or duplicate example indices.")
    if prediction["labels"].ndim != 1 or len(prediction["labels"]) != len(indices):
        raise ValueError(f"{path} labels and indices differ in length.")
    # Full metric validation checks class shape, integer labels, finite values
    # and probability normalization before we consume an artifact.
    compute_metrics(prediction["labels"], prediction["probabilities"], class_names)
    if (
        prediction["logits"].shape != prediction["probabilities"].shape
        or not np.isfinite(prediction["logits"]).all()
    ):
        raise ValueError(f"{path} has incompatible or nonfinite logits.")
    order = np.argsort(indices)
    return {key: values[order] for key, values in prediction.items()}


def _member_recipe(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("config", {})
    training = dict(config.get("training", {}))
    training.pop("workers", None)
    return {
        "model": config.get("model", summary.get("model", {})),
        "training": training,
        "world_size": summary.get("world_size", 1),
        "effective_batch_size": summary.get("effective_batch_size"),
    }


def ensemble_runs(
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    *,
    plots: bool = False,
) -> dict[str, Any]:
    """Average raw probabilities equally and export a standalone result run.

    Members must be completed and evaluated on the same protocol, exact class
    schema, and validation/test observations. Source row order may differ and
    is aligned using unique original indices. Calibration uses validation only;
    bootstrap intervals describe the raw test ensemble. This function never
    chooses members or learns ensemble weights from test labels.

    The bootstrap seed does not define a training replication. ``summary.seed``
    is the shared member training seed, or a stable identifier for their recipe
    and training-seed mapping. Across-ensemble aggregate confidence intervals
    require disjoint underlying member checkpoints; changing bootstrap seeds or
    partially overlapping member lists does not supply independent replications.
    """
    paths = [Path(path).expanduser().resolve() for path in run_dirs]
    output = Path(output_dir).expanduser().resolve()
    if len(paths) < 2:
        raise ValueError("An ensemble requires at least two completed member runs.")
    if len(set(paths)) != len(paths):
        raise ValueError(
            "Duplicate member run directories would change the declared equal weights."
        )
    if output in paths:
        raise ValueError("The ensemble output must be separate from its member runs.")
    if (output / "summary.json").exists():
        raise FileExistsError(f"Ensemble output already exists: {output}; choose a new directory.")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if plots and importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("Requested figures require matplotlib.")
    summaries, predictions, members = [], {"validation": [], "test": []}, []
    class_names, protocol_hash, split_hash = None, None, None
    for path in paths:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise ValueError(f"Member run is not completed: {path}")
        member_seed = summary.get("seed")
        if not isinstance(member_seed, int) or isinstance(member_seed, bool) or member_seed < 0:
            raise ValueError(f"{path} does not record a valid member training seed.")
        names = summary.get("class_names")
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) for name in names)
        ):
            raise ValueError(f"{path} does not record a valid class schema.")
        if not summary.get("protocol_hash") or not summary.get("split_hash"):
            raise ValueError(f"{path} lacks protocol or split hashes.")
        if class_names is None:
            class_names = names
            protocol_hash, split_hash = summary["protocol_hash"], summary["split_hash"]
        elif (
            names != class_names
            or summary["protocol_hash"] != protocol_hash
            or summary["split_hash"] != split_hash
        ):
            raise ValueError(f"Member uses incompatible protocol, split or class schema: {path}")
        artifact_hashes = {}
        for split in predictions:
            if not summary.get("evaluation", {}).get(split, {}).get("raw"):
                raise ValueError(f"{path} does not record completed {split} evaluation.")
            artifact = path / f"{split}_predictions.npz"
            prediction = _read_predictions(artifact, class_names)
            if predictions[split]:
                reference = predictions[split][0]
                if not np.array_equal(prediction["indices"], reference["indices"]):
                    raise ValueError(f"Members have different {split} example indices: {path}")
                if not np.array_equal(prediction["labels"], reference["labels"]):
                    raise ValueError(
                        f"Members have different {split} labels after index alignment: {path}"
                    )
            predictions[split].append(prediction)
            artifact_hashes[f"{split}_predictions_sha256"] = _hash_file(artifact)
        if np.intersect1d(
            predictions["validation"][-1]["indices"], predictions["test"][-1]["indices"]
        ).size:
            raise ValueError(f"Validation and test example indices overlap in {path}.")
        summaries.append(summary)
        members.append(
            {
                "run_dir": str(path),
                "seed": summary.get("seed"),
                "model": summary.get("model", summary.get("config", {}).get("model", {})),
                "weight": 1.0 / len(paths),
                "summary_sha256": _hash_file(path / "summary.json"),
                **artifact_hashes,
            }
        )
    assert class_names is not None
    averaged = {}
    for split, split_members in predictions.items():
        probabilities = np.mean(
            [member["probabilities"] for member in split_members], axis=0, dtype=np.float64
        )
        averaged[split] = {
            "indices": split_members[0]["indices"],
            "labels": split_members[0]["labels"],
            "logits": np.log(np.clip(probabilities, 1e-15, 1.0)),
            "probabilities": probabilities,
        }
    validation = averaged["validation"]
    temperature = fit_temperature(validation["logits"], validation["labels"])
    evaluation = {}
    for split, prediction in averaged.items():
        evaluation[split] = {
            "raw": compute_metrics(prediction["labels"], prediction["probabilities"], class_names),
            "temperature": temperature,
            "calibrated": compute_metrics(
                prediction["labels"],
                apply_temperature(prediction["logits"], temperature),
                class_names,
            ),
        }
    if bootstrap_samples:
        evaluation["test"]["bootstrap"] = bootstrap_metrics(
            averaged["test"]["labels"],
            averaged["test"]["probabilities"],
            class_names,
            bootstrap_samples,
            seed,
        )
    # Order and member run paths/training seeds do not define an ensemble recipe.
    # Different model/optimization conditions do, and are retained in report hashes.
    recipes = sorted(
        [_member_recipe(summary) for summary in summaries],
        key=lambda recipe: json.dumps(recipe, sort_keys=True),
    )
    replication_members = sorted(
        [
            {"recipe": _member_recipe(summary), "training_seed": summary["seed"]}
            for summary in summaries
        ],
        key=lambda member: json.dumps(member, sort_keys=True),
    )
    replication_id = hashlib.sha256(
        json.dumps(replication_members, sort_keys=True).encode()
    ).hexdigest()
    member_training_seeds = [summary["seed"] for summary in summaries]
    report_seed = (
        member_training_seeds[0]
        if len(set(member_training_seeds)) == 1
        else int(replication_id[:15], 16)
    )
    model = {
        "backend": "offline",
        "name": "equal_weight_probability_ensemble",
        "mode": "ensemble",
        "image_size": None,
        "member_recipes": recipes,
        "n_members": len(paths),
        "weight_rule": "equal arithmetic mean of raw class probabilities",
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "artifact_kind": "offline_probability_ensemble",
        "config": {"model": model, "training": {}, "experiment": {"seed": report_seed}},
        "model": model,
        "seed": report_seed,
        "bootstrap_seed": seed,
        "replication_id": replication_id,
        "member_training_seeds": member_training_seeds,
        "replication_members": replication_members,
        "protocol_hash": protocol_hash,
        "split_hash": split_hash,
        "class_names": class_names,
        "smoke_test": any(summary.get("smoke_test", False) for summary in summaries),
        "members": members,
        "member_selection": "caller-declared fixed member list; no test-label fitting",
        "uncertainty_notes": [
            "Example bootstrap intervals describe finite test-sample uncertainty, conditional on the fixed member list.",
            "Across-ensemble seed-mean confidence intervals require disjoint underlying member checkpoints; overlapping members or bootstrap reseeding do not create independent training replications.",
        ],
        "best_epoch": None,
        "best_val_macro_f1": evaluation["validation"]["raw"]["macro_f1"],
        "mean_member_validation_macro_f1": float(
            np.mean(
                [
                    compute_metrics(member["labels"], member["probabilities"], class_names)[
                        "macro_f1"
                    ]
                    for member in predictions["validation"]
                ]
            )
        ),
        "calibration": {
            "fit_split": "validation",
            "temperature": temperature,
            "log_probability_floor": 1e-15,
        },
        "evaluation": evaluation,
    }
    output.mkdir(parents=True, exist_ok=True)
    for split, prediction in averaged.items():
        _atomic_predictions(output / f"{split}_predictions.npz", prediction)
    _atomic_json(output / "config.json", summary["config"])
    _atomic_json(output / "calibration.json", summary["calibration"])
    if plots:
        from .report import plot_predictions

        summary["figures"] = [
            str(path.relative_to(output))
            for path in plot_predictions(
                output / "test_predictions.npz", class_names, output / "figures" / "test"
            )
        ]
    _atomic_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Equal-weight offline ensemble of completed FireRisk test runs"
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Explicit member run directories")
    parser.add_argument("--output", required=True, help="New ensemble result directory")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bootstrap seed; does not define an independent training replication",
    )
    parser.add_argument(
        "--plots", action="store_true", help="Save held-out confusion and reliability figures"
    )
    args = parser.parse_args(argv)
    summary = ensemble_runs(
        args.runs, args.output, args.bootstrap_samples, args.seed, plots=args.plots
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "n_members": len(summary["members"]),
                "test_macro_f1": summary["evaluation"]["test"]["raw"]["macro_f1"],
                "temperature": summary["calibration"]["temperature"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
