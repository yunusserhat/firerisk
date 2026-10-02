"""Offline two-GPU engine check: torchrun --nproc_per_node=2 this_file.py DIR.

This helper uses tiny image-independent features, so no model or dataset weights
are downloaded. Normal pytest tests remain CPU-only and do not invoke torchrun.
"""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_engine import SmallPreparedData  # noqa: E402

from firerisk import engine  # noqa: E402
from firerisk.config import DEFAULTS  # noqa: E402
from firerisk.models import VisionClassifier  # noqa: E402


def tiny_model(config, num_classes, cache_dir):
    return VisionClassifier(
        nn.Sequential(nn.Linear(4, 6), nn.Tanh()),
        6,
        num_classes,
        backend="timm",
        mode="full",
        dropout=0.1,
        metadata={"fixture": True},
    )


def main():
    run_dir = Path(sys.argv[1]).resolve()
    config = deepcopy(DEFAULTS)
    config["storage"].update(root=str(run_dir.parent), min_free_gb=0)
    config["model"].update(pretrained=False, gradient_checkpointing=False)
    config["data"]["max_shards"] = 1
    config["training"].update(
        epochs=2,
        batch_size=2,
        eval_batch_size=2,
        accumulation_steps=2,
        workers=0,
        precision="fp32",
        patience=5,
        max_train_batches=None,
    )
    engine.build_model = tiny_model
    engine.build_transforms = lambda *args: (None, None, {"fixture": True})
    prepared = SmallPreparedData(run_dir.parent)
    engine.train(config, prepared, run_dir)
    if int(os.environ.get("RANK", "0")) == 0:
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["world_size"] == 2 and summary["status"] == "complete"
        predictions = np.load(run_dir / "validation_predictions.npz")
        assert len(predictions["indices"]) == len(set(predictions["indices"])) == 2
        checkpoint = torch.load(run_dir / "last.pt", weights_only=False, map_location="cpu")
        assert len(checkpoint["rng_states"]) == 2
        print("Offline two-GPU DDP smoke passed", flush=True)


if __name__ == "__main__":
    main()
