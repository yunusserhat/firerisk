"""Command line entry points. Expensive work is always an explicit subcommand."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .storage import configure_cache_environment, inspect_storage, resolve_storage_root


def _common(parser):
    parser.add_argument(
        "--config", default="configs/siglip2_base.yaml", help="YAML experiment preset"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted YAML override; repeatable",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Storage-aware FireRisk research benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["doctor", "prepare", "train", "status"]:
        p = sub.add_parser(name)
        _common(p)
        if name == "train":
            p.add_argument("--run-dir")
            p.add_argument("--resume", help="Same run's last.pt; resumes at the next epoch")
    p = sub.add_parser(
        "evaluate", help="Evaluate selected best.pt; temperature is fitted on validation only"
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--bootstrap-samples", type=int)
    p = sub.add_parser("compare", help="Export compatible held-out results across models and seeds")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-mixed-protocols", action="store_true")
    p.add_argument("--include-smoke", action="store_true")
    p = sub.add_parser("predict", help="Classify an individual RGB aerial image")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--image", required=True)
    args = parser.parse_args(argv)
    if args.command in {"doctor", "prepare", "train", "status"}:
        config = load_config(args.config, args.set)
    elif args.command in {"evaluate", "predict"}:
        config = json.loads((Path(args.run_dir) / "config.json").read_text())
    else:
        from .report import run_report

        run_report(args)
        return
    root = resolve_storage_root(config["storage"])
    configure_cache_environment(root)
    if args.command == "status":
        rows = []
        for path in sorted((root / "runs").glob("*/summary.json")):
            summary = json.loads(path.read_text())
            history_path = path.parent / "history.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            rows.append(
                {
                    "run": path.parent.name,
                    "status": summary.get("status"),
                    "completed_epochs": history[-1]["epoch"] if history else 0,
                    "best_val_macro_f1": max((h["val_macro_f1"] for h in history), default=None),
                    "smoke_test": summary.get("smoke_test", False),
                    "test_evaluated": "test" in summary.get("evaluation", {}),
                    "path": str(path.parent),
                }
            )
        print(json.dumps(rows, indent=2))
        return
    if args.command == "doctor":
        from .runtime import environment_info

        state = {
            "storage": inspect_storage(root),
            "environment": environment_info(),
            "data_revision": config["data"]["revision"],
            "expected_dataset_gib": 11575727336 / 2**30,
            "access_mode": config["data"]["access_mode"],
            "expected_encoded_pack_gib": 11575727336 / 2**30
            if config["data"]["access_mode"] == "packed"
            else 0,
        }
        print(json.dumps(state, indent=2))
        return
    if args.command == "predict":
        _predict(config, args, root)
        return
    # Preparation runs once before torchrun; shared manifests must already exist for DDP.
    import os

    from .data import prepare_data

    if (
        int(os.environ.get("WORLD_SIZE", "1")) > 1
        and not (root / "data" / "manifest.json").exists()
    ):
        raise RuntimeError("Run 'firerisk prepare' before launching torchrun")
    prepared = prepare_data(config)
    if args.command == "prepare":
        print(json.dumps({k: v for k, v in prepared.manifest.items() if k != "shards"}, indent=2))
    elif args.command == "train":
        from .engine import train

        train(config, prepared, args.run_dir, args.resume)
    elif args.command == "evaluate":
        from .engine import evaluate_run
        from .report import plot_history, plot_predictions

        metrics = evaluate_run(args.run_dir, prepared, args.split, args.bootstrap_samples)
        plot_predictions(
            Path(args.run_dir) / f"{args.split}_predictions.npz",
            prepared.class_names,
            Path(args.run_dir) / "figures" / args.split,
        )
        plot_history(Path(args.run_dir) / "history.json", Path(args.run_dir) / "figures")
        print(json.dumps(metrics, indent=2))


def _predict(config, args, root):
    import torch
    from PIL import Image, ImageOps

    from .engine import load_model_state
    from .metrics import apply_temperature
    from .models import build_model, build_transforms

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    names = summary["class_names"]
    _, transform, _ = build_transforms(config, root / "hub")
    model = build_model(config, len(names), root / "hub")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    load_model_state(model, checkpoint["model"])
    model.eval()
    with Image.open(args.image) as image, torch.no_grad():
        logits = model(transform(ImageOps.exif_transpose(image)).unsqueeze(0)).numpy()
    calibration = run_dir / "calibration.json"
    temperature = (
        json.loads(calibration.read_text())["temperature"] if calibration.exists() else 1.0
    )
    probs = apply_temperature(logits, temperature)[0]
    print(
        json.dumps(
            {
                "prediction": names[int(probs.argmax())],
                "temperature": temperature,
                "probabilities": dict(zip(names, probs.tolist())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
