"""Small CPU integration runs test checkpoint selection and exact continuation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from firerisk import engine
from firerisk.config import DEFAULTS
from firerisk.models import VisionClassifier


class FeatureDataset(Dataset):
    def __init__(self, prepared, indices):
        self.prepared = prepared
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        return {
            "pixel_values": self.prepared.features[index],
            "labels": int(self.prepared.labels[index]),
            "index": index,
        }


class SmallPreparedData:
    def __init__(self, root):
        self.root = root
        self.class_names = ["a", "b"]
        self.labels = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1])
        self.features = torch.tensor(
            [[0.1 + i / 15, 0.9 - i / 12, (i % 3) / 4, -0.5 + i / 8] for i in range(9)],
            dtype=torch.float32,
        )
        self.splits = {
            "train": np.arange(5),
            "validation": np.array([5, 6]),
            "test": np.array([7, 8]),
        }
        self.train_class_counts = np.bincount(self.labels[self.splits["train"]], minlength=2)
        self.manifest = {"split_hash": "offline-fixed-split", "repo_id": "offline-fixture"}
        self.requested_splits = []

    def build_dataset(self, split, transform):
        self.requested_splits.append(split)
        return FeatureDataset(self, self.splits[split])


@pytest.fixture
def tiny_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = deepcopy(DEFAULTS)
    config["storage"].update(root=str(tmp_path), min_free_gb=0)
    config["experiment"].update(name="tiny-integration", seed=17)
    config["model"].update(pretrained=False, mode="full", gradient_checkpointing=False)
    config["data"]["max_shards"] = 1
    config["training"].update(
        epochs=3,
        batch_size=2,
        eval_batch_size=2,
        accumulation_steps=2,
        workers=0,
        lr_backbone=0.003,
        lr_head=0.01,
        weight_decay=0.01,
        warmup_fraction=0,
        label_smoothing=0,
        clip_grad_norm=100,
        precision="fp32",
        patience=5,
        max_train_batches=None,
        max_eval_batches=None,
    )
    config["evaluation"].update(bootstrap_samples=0, calibrate=False)

    def make_tiny_model(config, num_classes, cache_dir):
        return VisionClassifier(
            nn.Sequential(nn.Linear(4, 6), nn.Tanh()),
            6,
            num_classes,
            backend="timm",
            mode=config["model"]["mode"],
            dropout=0.25,
            metadata={"fixture": True},
        )

    monkeypatch.setattr(engine, "build_model", make_tiny_model)
    monkeypatch.setattr(engine, "build_transforms", lambda *args: (None, None, {"fixture": True}))
    monkeypatch.setattr(engine, "environment_info", lambda: {"fixture": "cpu"})
    prepared = SmallPreparedData(tmp_path)
    return config, prepared


def _cpu_train(config, prepared, run_dir, resume=None):
    return engine._train(config, prepared, run_dir, resume, torch.device("cpu"), 0, 1)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_training_saves_best_and_resumable_last_without_touching_test(tiny_engine, tmp_path):
    config, prepared = tiny_engine
    run = _cpu_train(config, prepared, tmp_path / "run")
    assert prepared.requested_splits == ["train", "validation", "validation"]
    assert not (run / "test_predictions.npz").exists()
    best, last = _load(run / "best.pt"), _load(run / "last.pt")
    assert best["epoch"] <= last["epoch"] == 2
    assert "optimizer" not in best and "optimizer" in last
    assert all(key in last for key in ["scheduler", "scaler", "rng_states", "history"])
    assert last["scheduler"]["last_epoch"] == 6  # Three batches, two updates per epoch.
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["smoke_test"]
    assert summary["best_epoch"] == best["epoch"] + 1
    assert (run / "validation_predictions.npz").exists()


def test_best_checkpoint_selected_using_validation_macro_f1(tiny_engine, tmp_path, monkeypatch):
    config, prepared = tiny_engine
    sequence = iter([0.25, 0.80, 0.60, 0.80])
    monkeypatch.setattr(
        engine, "compute_metrics", lambda *args: {"macro_f1": next(sequence), "accuracy": 0.5}
    )
    run = _cpu_train(config, prepared, tmp_path / "selection")
    best, last = _load(run / "best.pt"), _load(run / "last.pt")
    assert best["epoch"] == 1 and best["best_score"] == 0.80
    assert last["epoch"] == 2


def test_epoch_resume_exactly_matches_uninterrupted_weights_and_history(
    tiny_engine, tmp_path, monkeypatch
):
    config, prepared = tiny_engine
    complete = _cpu_train(deepcopy(config), prepared, tmp_path / "complete")
    interrupted = tmp_path / "interrupted"
    original_save = engine.atomic_save

    class SimulatedInterruption(Exception):
        pass

    def save_then_interrupt(path, payload, min_free_gb):
        original_save(path, payload, min_free_gb)
        if Path(path).name == "last.pt" and payload["epoch"] == 0:
            raise SimulatedInterruption("Interrupted after a complete epoch checkpoint")

    with monkeypatch.context() as temporary:
        temporary.setattr(engine, "atomic_save", save_then_interrupt)
        with pytest.raises(SimulatedInterruption):
            _cpu_train(deepcopy(config), prepared, interrupted)
    _cpu_train(deepcopy(config), prepared, interrupted, interrupted / "last.pt")
    expected, resumed = _load(complete / "last.pt"), _load(interrupted / "last.pt")
    assert expected["history"] == resumed["history"]
    for name in expected["model"]:
        assert torch.equal(expected["model"][name], resumed["model"][name]), name
    assert expected["scheduler"] == resumed["scheduler"]
    assert torch.equal(
        expected["rng_states"][0]["loader_generator"], resumed["rng_states"][0]["loader_generator"]
    )


def test_resume_rejects_changed_scientific_protocol_and_optimization(tiny_engine, tmp_path):
    config, prepared = tiny_engine
    run = _cpu_train(deepcopy(config), prepared, tmp_path / "resume-validation")
    changed = deepcopy(config)
    changed["training"]["lr_head"] *= 2
    with pytest.raises(ValueError, match="training"):
        _cpu_train(changed, prepared, run, run / "last.pt")
    prepared.manifest["split_hash"] = "another-split"
    with pytest.raises(ValueError, match="protocol"):
        _cpu_train(deepcopy(config), prepared, run, run / "last.pt")


def test_early_stopped_last_checkpoint_resume_performs_no_more_updates(
    tiny_engine, tmp_path, monkeypatch
):
    config, prepared = tiny_engine
    config["training"].update(epochs=6, patience=1)
    monkeypatch.setattr(engine, "compute_metrics", lambda *args: {"macro_f1": 0.5, "accuracy": 0.5})
    run = _cpu_train(deepcopy(config), prepared, tmp_path / "early-stop")
    initial = _load(run / "last.pt")
    assert initial["epoch"] == 1 and initial["stale"] == 1
    _cpu_train(deepcopy(config), prepared, run, run / "last.pt")
    resumed = _load(run / "last.pt")
    assert resumed["epoch"] == initial["epoch"]
    assert resumed["history"] == initial["history"]
    assert resumed["scheduler"] == initial["scheduler"]


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_uneven_accumulation_matches_large_batch_cross_entropy_gradient(weighted, smoothing):
    torch.manual_seed(3)
    x = torch.randn(5, 4)
    targets = torch.tensor([0, 0, 0, 1, 1])
    weights = torch.tensor([0.25, 2.0]) if weighted else None
    large = nn.Linear(4, 2)
    accumulated = deepcopy(large)
    engine.loss_function(large(x), targets, smoothing, weights).backward()
    for start, stop in [(0, 2), (2, 4), (4, 5)]:
        loss = engine.loss_function(
            accumulated(x[start:stop]), targets[start:stop], smoothing, weights
        )
        (loss * ((stop - start) / len(x))).backward()
    for expected, actual in zip(large.parameters(), accumulated.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-5)


def test_probe_checkpoint_omits_frozen_backbone_and_restores_trainable_head(tiny_engine, tmp_path):
    config, _ = tiny_engine
    config["model"]["mode"] = "linear"
    model = engine.build_model(config, 2, tmp_path)
    state = engine.checkpoint_state(model)
    assert state and all(name.startswith("head.") for name in state)
    before = model.head[-1].weight.clone()
    with torch.no_grad():
        model.head[-1].weight.add_(1)
    engine.load_model_state(model, state)
    assert torch.equal(before, model.head[-1].weight)
    broken = dict(state)
    broken.pop("head.2.weight")
    with pytest.raises(RuntimeError, match="missing trainable"):
        engine.load_model_state(model, broken)


def test_distributed_eval_sampler_has_no_padding_or_duplicates():
    shards = [list(engine.DistributedEvalSampler(5, rank, 3)) for rank in range(3)]
    assert shards == [[0, 3], [1, 4], [2]]
    assert sorted(index for shard in shards for index in shard) == list(range(5))


@pytest.mark.parametrize("local_indices", [[4, 0], []])
def test_distributed_predictions_sync_canonical_buffers_before_uneven_shard_evaluation(
    monkeypatch, local_indices
):
    torch.manual_seed(31)
    model = VisionClassifier(
        nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.ReLU()),
        6,
        2,
        backend="timm",
        mode="full",
        dropout=0,
        metadata={"fixture": True},
    )
    bn = model.backbone[1]
    bn.running_mean.copy_(torch.arange(6) / 10)
    bn.running_var.fill_(1.5)
    bn.num_batches_tracked.fill_(7)
    canonical = deepcopy(model).eval()
    canonical_buffers = [buffer.clone() for buffer in canonical.buffers()]
    bn.running_mean.fill_(20)
    bn.running_var.fill_(9)
    bn.num_batches_tracked.fill_(99)
    features = torch.randn(5, 4)
    labels = np.array([0, 1, 0, 1, 1])
    remote_indices = [index for index in range(5) if index not in local_indices]
    local_loader = [
        {
            "pixel_values": features[index : index + 1],
            "labels": torch.tensor([labels[index]]),
            "index": torch.tensor([index]),
        }
        for index in local_indices
    ]
    with torch.no_grad():
        expected = canonical(features).numpy()
    broadcast_calls = []

    def synchronize_buffer(buffer, src):
        assert src == 0
        buffer.copy_(canonical_buffers[len(broadcast_calls)])
        broadcast_calls.append(buffer)

    def assert_synchronized_before_forward(module, args):
        assert len(broadcast_calls) == len(canonical_buffers)
        assert all(torch.equal(a, b) for a, b in zip(module.buffers(), canonical_buffers))

    def gather_uneven_shards(shards, local):
        assert len(broadcast_calls) == len(canonical_buffers)
        shards[:] = [
            local,
            {
                "logits": expected[remote_indices],
                "labels": labels[remote_indices],
                "indices": np.array(remote_indices),
            },
        ]

    monkeypatch.setattr(engine.dist, "broadcast", synchronize_buffer)
    monkeypatch.setattr(engine.dist, "all_gather_object", gather_uneven_shards)
    model.register_forward_pre_hook(assert_synchronized_before_forward)
    result = engine.collect_predictions(model, local_loader, torch.device("cpu"), world=2)
    np.testing.assert_array_equal(result["indices"], np.arange(5))
    np.testing.assert_array_equal(result["labels"], labels)
    np.testing.assert_allclose(result["logits"], expected, atol=1e-6)
    assert len(broadcast_calls) == len(canonical_buffers)


def test_evaluation_is_explicit_and_preserves_training_checkpoint(
    tiny_engine, tmp_path, monkeypatch
):
    config, prepared = tiny_engine
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    run = _cpu_train(config, prepared, tmp_path / "evaluate")
    before = (run / "best.pt").read_bytes()
    result = engine.evaluate_run(run, prepared, "test", bootstrap_samples=0)
    assert result["raw"]["n_examples"] == 2
    assert prepared.requested_splits[-1] == "test"
    assert (run / "test_predictions.npz").exists()
    assert (run / "best.pt").read_bytes() == before


def test_evaluation_rejects_incomplete_runs_and_negative_bootstrap_before_model_load(
    tiny_engine, tmp_path, monkeypatch
):
    config, prepared = tiny_engine
    run = _cpu_train(config, prepared, tmp_path / "evaluation-validation")

    def unexpected_model_load(*args):
        raise AssertionError("Invalid evaluation must fail before loading any weights")

    monkeypatch.setattr(engine, "build_model", unexpected_model_load)
    with pytest.raises(ValueError, match="nonnegative"):
        engine.evaluate_run(run, prepared, bootstrap_samples=-1)
    summary = json.loads((run / "summary.json").read_text())
    summary["status"] = "running"
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="completed"):
        engine.evaluate_run(run, prepared)


def test_ddp_preflight_broadcasts_existing_run_error_to_every_rank(
    tiny_engine, tmp_path, monkeypatch
):
    config, prepared = tiny_engine
    run = tmp_path / "existing-run"
    run.mkdir()
    (run / "summary.json").write_text("{}")
    shared_error = []

    def fake_broadcast(objects, src):
        if objects[0] is not None:
            shared_error[:] = objects
        else:
            objects[:] = shared_error

    def unexpected_barrier():
        raise AssertionError("Ranks must raise before waiting at a barrier")

    monkeypatch.setattr(engine.dist, "broadcast_object_list", fake_broadcast)
    monkeypatch.setattr(engine.dist, "barrier", unexpected_barrier)
    for rank in [0, 1]:
        with pytest.raises(FileExistsError, match="Run already exists"):
            engine._train(deepcopy(config), prepared, run, None, torch.device("cpu"), rank, 2)
