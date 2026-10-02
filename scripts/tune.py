"""Validation-only search. Freeze the recipe before evaluating a selected run."""

import argparse
import itertools
import json

from firerisk.config import load_config
from firerisk.data import prepare_data
from firerisk.engine import train
from firerisk.runtime import json_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/siglip2_base.yaml")
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-5, 3e-5])
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--balance", nargs="+", choices=["none", "weighted_loss"], default=["none"])
    args = parser.parse_args()
    base = load_config(args.config)
    prepared = prepare_data(base)
    candidates = list(itertools.product(args.lr, args.weight_decay, args.balance))
    search_id = json_hash({"base": base, "candidates": candidates})[:12]
    output = prepared.root / "tuning" / search_id
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "plan.json",
        {
            "base": base,
            "candidates": candidates,
            "selection_metric": "validation.macro_f1",
            "test_accessed": False,
        },
    )
    results = []
    for number, (lr, decay, balance) in enumerate(candidates):
        name = f"{base['experiment']['name']}-search-{search_id}-{number}"
        config = load_config(
            args.config,
            [
                f"experiment.name={name}",
                f"training.lr_backbone={lr}",
                f"training.weight_decay={decay}",
                f"training.class_balance={balance}",
            ],
        )
        run = prepared.root / "runs" / f"{name}-seed{config['experiment']['seed']}"
        summary_path = run / "summary.json"
        done = (
            summary_path.exists()
            and json.loads(summary_path.read_text()).get("status") == "complete"
        )
        if not done:
            resume = str(run / "last.pt") if (run / "last.pt").exists() else None
            train(config, prepared, run_dir=run, resume=resume)
        summary = json.loads(summary_path.read_text())
        results.append(
            {
                "run": str(run),
                "lr": lr,
                "weight_decay": decay,
                "balance": balance,
                "validation_macro_f1": summary["best_val_macro_f1"],
            }
        )
        write_json(output / "results.json", results)
    selected = max(results, key=lambda r: r["validation_macro_f1"])
    write_json(output / "selected.json", selected)
    print(
        f"Validation-selected recipe: {selected}. Repeat with independent training seeds before final test reporting."
    )


if __name__ == "__main__":
    main()
