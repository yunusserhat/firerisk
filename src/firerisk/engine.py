"""Single GPU / torchrun training with validation-only checkpoint selection."""

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler, WeightedRandomSampler
from tqdm.auto import tqdm

from .metrics import apply_temperature, bootstrap_metrics, compute_metrics, fit_temperature
from .models import build_model, build_transforms
from .runtime import (
    environment_info,
    json_hash,
    restore_rng,
    rng_state,
    seed_everything,
    seed_worker,
    write_json,
)
from .storage import check_free_space


class DistributedEvalSampler(Sampler):
    """Partition without padding: no duplicated observations in reported metrics."""

    def __init__(self, size, rank=0, world_size=1):
        self.indices = list(range(rank, size, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def distributed_setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return device, rank, world


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 is unsupported on this GPU; select fp16 or fp32")
    return torch.autocast(device_type="cuda", dtype=dtype)


def parameter_groups(model, config):
    """Head/backbone rates with no decay on biases, norms and scalar parameters."""
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        section = "head" if name.startswith("head.") else "backbone"
        decay = param.ndim > 1 and not name.endswith("bias")
        key = (section, decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": config[f"lr_{section}"],
                "weight_decay": config["weight_decay"] if decay else 0.0,
            }
        groups[key]["params"].append(param)
    return list(groups.values())


def loss_function(logits, labels, smoothing=0.0, weights=None, mixup_alpha=0.0):
    """Average per-example CE, including class weights, over the sample count.

    PyTorch's weighted mean divides by the sum of target class weights. Using
    that reduction separately in each microbatch changes the effective loss
    when gradient accumulation splits a heterogeneous batch. A per-example
    mean preserves the same objective for full and accumulated batches.
    Mixup is performed on images by the train loop.
    """
    return nn.functional.cross_entropy(
        logits.float(), labels, weight=weights, label_smoothing=smoothing, reduction="none"
    ).mean()


def checkpoint_state(model):
    """Adapters/probes store trainable weights and buffers, never duplicate frozen weights."""
    model = unwrap(model)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    names.update(n for n, _ in model.named_buffers())
    return {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if n in names}


def load_model_state(model, state):
    model = unwrap(model)
    result = model.load_state_dict(state, strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = trainable.intersection(result.missing_keys)
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing trainable={missing}, unexpected={result.unexpected_keys}"
        )


def atomic_save(path, payload, min_free_gb=5.0):
    path = Path(path)
    tensors = payload.get("model", {})
    estimate = sum(v.numel() * v.element_size() for v in tensors.values())
    # Adam moments plus serialization slack; temporary replaces old file atomically.
    check_free_space(path.parent, max(estimate * 4, 2**20), min_free_gb)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


@torch.no_grad()
def collect_predictions(model, loader, device, precision="fp32", max_batches=None, world=1):
    # Evaluate the unwrapped module to avoid DDP forward collectives with uneven eval shards.
    model = unwrap(model)
    model.eval()
    if world > 1:
        # Canonical rank-zero BatchNorm statistics are the ones saved in best.pt.
        # All ranks must evaluate that same model even with uneven eval partitions.
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    logits, labels, indices = [], [], []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        with autocast_context(device, precision):
            output = model(batch["pixel_values"].to(device, non_blocking=True))
        logits.append(output.float().cpu().numpy())
        labels.append(np.asarray(batch["labels"]))
        indices.append(np.asarray(batch["index"]))
    result = {
        "logits": np.concatenate(logits) if logits else np.empty((0, model.head[-1].out_features)),
        "labels": np.concatenate(labels) if labels else np.empty(0, dtype=int),
        "indices": np.concatenate(indices) if indices else np.empty(0, dtype=int),
    }
    if world > 1:
        shards = [None] * world
        dist.all_gather_object(shards, result)
        result = {k: np.concatenate([s[k] for s in shards]) for k in result}
    order = np.argsort(result["indices"])
    return {k: v[order] for k, v in result.items()}


def make_loader(dataset, config, train=False, labels=None, rank=0, world=1, generator=None):
    t = config["training"]
    if train and world > 1:
        if t["class_balance"] == "weighted_sampler":
            raise ValueError("weighted_sampler requires single GPU; use weighted_loss with DDP")
        sampler = DistributedSampler(
            dataset, world, rank, shuffle=True, seed=config["experiment"]["seed"], drop_last=False
        )
    elif train and t["class_balance"] == "weighted_sampler":
        counts = np.bincount(labels)
        sampler = WeightedRandomSampler(
            1.0 / counts[labels], len(labels), replacement=True, generator=generator
        )
    elif not train and world > 1:
        sampler = DistributedEvalSampler(len(dataset), rank, world)
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=t["batch_size"] if train else t["eval_batch_size"],
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=t["workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    return loader


def train(config, prepared, run_dir=None, resume=None):
    """Train or resume from a trusted local last.pt created by this codebase.

    Exact resume restores NumPy/Python RNG state and therefore uses torch.load
    with weights_only=False for last.pt. Inference-only best.pt uses the safer
    weights_only=True loader and does not contain serialized RNG objects.
    """
    device, rank, world = distributed_setup()
    try:
        return _train(config, prepared, run_dir, resume, device, rank, world)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(config, prepared, run_dir, resume, device, rank, world):
    t = config["training"]
    seed = config["experiment"]["seed"]
    seed_everything(seed + rank, t["deterministic"])
    root = prepared.root
    run_dir = (
        Path(run_dir) if run_dir else root / "runs" / f"{config['experiment']['name']}-seed{seed}"
    )
    preflight_error = None
    if rank == 0:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / "summary.json").exists() and resume is None:
                raise FileExistsError(
                    f"Run already exists: {run_dir}; choose a new name or --resume"
                )
        except OSError as error:
            preflight_error = (type(error).__name__, str(error))
    if world > 1:
        # Share rank zero's filesystem failure before anybody enters a barrier.
        # Otherwise the other ranks can wait indefinitely for an exited leader.
        errors = [preflight_error]
        dist.broadcast_object_list(errors, src=0)
        preflight_error = errors[0]
    if preflight_error:
        error_type, message = preflight_error
        if error_type == "FileExistsError":
            raise FileExistsError(message)
        raise RuntimeError(f"Run directory preflight failed: {message}")
    train_tf, eval_tf, preprocessing = build_transforms(config, root / "hub")
    resolved = preprocessing.get("resolved_revision")
    if resolved:
        config["model"]["revision"] = resolved
    model = build_model(config, len(prepared.class_names), root / "hub").to(device)
    generator = torch.Generator().manual_seed(seed + rank)
    train_loader = make_loader(
        prepared.build_dataset("train", train_tf),
        config,
        True,
        prepared.labels[prepared.splits["train"]],
        rank,
        world,
        generator,
    )
    val_loader = make_loader(
        prepared.build_dataset("validation", eval_tf),
        config,
        rank=rank,
        world=world,
        generator=generator,
    )
    if not len(train_loader):
        raise ValueError("Training dataset is empty")
    optimizer = torch.optim.AdamW(parameter_groups(model, t))
    num_batches = min(len(train_loader), t["max_train_batches"] or len(train_loader))
    updates_per_epoch = math.ceil(num_batches / t["accumulation_steps"])
    total_updates = updates_per_epoch * t["epochs"]
    warmup = int(total_updates * t["warmup_fraction"])

    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and t["precision"] == "fp16"
    )
    weights = None
    if t["class_balance"] == "weighted_loss":
        counts = prepared.train_class_counts.astype(float)
        weights = torch.tensor(
            np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0),
            dtype=torch.float32,
            device=device,
        )
    split_hash = prepared.manifest.get("split_hash") or json_hash(
        {k: v.tolist() for k, v in prepared.splits.items()}
    )
    protocol_hash = json_hash(
        {
            "data_revision": config["data"]["revision"],
            "split_hash": split_hash,
            "class_names": prepared.class_names,
        }
    )
    smoke = bool(
        config["data"]["max_shards"]
        or t["max_train_batches"]
        or t["max_eval_batches"]
        or not config["model"]["pretrained"]
    )
    summary = {
        "config": config,
        "protocol_hash": protocol_hash,
        "split_hash": split_hash,
        "seed": seed,
        "model": getattr(model, "metadata", config["model"]),
        "world_size": world,
        "effective_batch_size": t["batch_size"] * t["accumulation_steps"] * world,
        "smoke_test": smoke,
        "class_names": prepared.class_names,
        "evaluation": {},
    }
    start_epoch, best_score, stale, history, elapsed_before = 0, -1.0, 0, [], 0.0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if checkpoint["protocol_hash"] != protocol_hash:
            raise ValueError("Resume checkpoint uses a different data protocol")
        # Allow storage relocation, but require identical scientific / optimization settings.
        for section in ["model", "training", "data", "experiment"]:
            if checkpoint["config"][section] != config[section]:
                raise ValueError(f"Resume configuration differs: {section}")
        if checkpoint.get("world_size", 1) != world:
            raise ValueError("Exact resume requires the original world size")
        load_model_state(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_score, stale, history = (
            checkpoint["best_score"],
            checkpoint["stale"],
            checkpoint["history"],
        )
        elapsed_before = checkpoint["training_seconds"]
        states = checkpoint["rng_states"]
        restore_rng(states[rank]["rng"])
        generator.set_state(states[rank]["loader_generator"].cpu())
        if not (run_dir / "best.pt").exists():
            raise FileNotFoundError("Resume requires best.pt in the same run directory")
    if rank == 0:
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "environment.json", environment_info())
        write_json(run_dir / "data_manifest.json", prepared.manifest)
        write_json(run_dir / "preprocessing.json", preprocessing)
        write_json(run_dir / "summary.json", {**summary, "status": "running"})
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    # A last.pt saved at the patience boundary already completed early stopping.
    # Resuming it should regenerate final artifacts without extra optimization.
    stop_epoch = t["epochs"] if stale < t["patience"] else start_epoch
    for epoch in range(start_epoch, stop_epoch):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen = 0.0, 0
        iterator = tqdm(
            train_loader,
            total=num_batches,
            desc=f"epoch {epoch + 1}/{t['epochs']}",
            disable=rank != 0,
        )
        for step, batch in enumerate(iterator):
            if step >= num_batches:
                break
            images = batch["pixel_values"].to(device, non_blocking=True)
            targets = batch["labels"].to(device, non_blocking=True)
            block_start = (step // t["accumulation_steps"]) * t["accumulation_steps"]
            block_size = min(t["accumulation_steps"], num_batches - block_start)
            # Correct weighting for an incomplete final microbatch / accumulation window.
            sample_count = (
                len(train_loader.sampler)
                if train_loader.sampler is not None
                else len(train_loader.dataset)
            )
            block_samples = sum(
                min(t["batch_size"], max(0, sample_count - j * t["batch_size"]))
                for j in range(block_start, block_start + block_size)
            )
            should_step = step + 1 == num_batches or (step + 1) % t["accumulation_steps"] == 0
            context = model.no_sync() if world > 1 and not should_step else nullcontext()
            with context:
                with autocast_context(device, t["precision"]):
                    if t["mixup_alpha"] > 0:
                        lam = float(np.random.beta(t["mixup_alpha"], t["mixup_alpha"]))
                        permutation = torch.randperm(images.size(0), device=device)
                        mixed = lam * images + (1 - lam) * images[permutation]
                        logits = model(mixed)
                        loss = lam * loss_function(
                            logits, targets, t["label_smoothing"], weights
                        ) + (1 - lam) * loss_function(
                            logits, targets[permutation], t["label_smoothing"], weights
                        )
                    else:
                        logits = model(images)
                        loss = loss_function(logits, targets, t["label_smoothing"], weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {step}")
                scaler.scale(loss * (len(targets) / block_samples)).backward()
            total_loss += float(loss.detach()) * len(targets)
            seen += len(targets)
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], t["clip_grad_norm"]
                )
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        if world > 1:
            totals = torch.tensor([total_loss, seen], device=device, dtype=torch.float64)
            dist.all_reduce(totals)
            total_loss, seen = totals.tolist()
        prediction = collect_predictions(
            model, val_loader, device, t["precision"], t["max_eval_batches"], world
        )
        metrics = compute_metrics(
            prediction["labels"], apply_temperature(prediction["logits"], 1.0), prepared.class_names
        )
        score = metrics["macro_f1"]
        improved = score > best_score + t["min_delta"]
        best_score = max(best_score, score) if improved else best_score
        stale = 0 if improved else stale + 1
        entry = {
            "epoch": epoch + 1,
            "train_loss": total_loss / seen,
            "val_macro_f1": score,
            "val_accuracy": metrics["accuracy"],
            "lr": [g["lr"] for g in optimizer.param_groups],
        }
        history.append(entry)
        local_rng = {"rng": rng_state(), "loader_generator": generator.get_state()}
        states = [local_rng]
        if world > 1:
            states = [None] * world
            dist.all_gather_object(states, local_rng)
        if rank == 0:
            payload = {
                "model": checkpoint_state(model),
                "config": config,
                "protocol_hash": protocol_hash,
                "class_names": prepared.class_names,
                "epoch": epoch,
                "best_score": best_score,
                "world_size": world,
            }
            if improved:
                atomic_save(run_dir / "best.pt", payload, config["storage"]["min_free_gb"])
                summary["best_epoch"] = epoch + 1
            if t["save_last"]:
                payload.update(
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(),
                    rng_states=states,
                    stale=stale,
                    history=history,
                    training_seconds=elapsed_before + time.monotonic() - start,
                )
                atomic_save(run_dir / "last.pt", payload, config["storage"]["min_free_gb"])
            write_json(run_dir / "history.json", history)
            print(json.dumps(entry), flush=True)
        if world > 1:
            dist.barrier()
        if stale >= t["patience"]:
            break
    if rank == 0:
        best = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        load_model_state(model, best["model"])
        # Rank zero evaluates the complete validation set after training; test remains untouched.
        loader = make_loader(prepared.build_dataset("validation", eval_tf), config)
        validation = collect_predictions(
            model, loader, device, t["precision"], t["max_eval_batches"]
        )
        probs = apply_temperature(validation["logits"], 1.0)
        np.savez_compressed(
            run_dir / "validation_predictions.npz", **validation, probabilities=probs
        )
        summary.update(
            status="complete",
            best_epoch=best["epoch"] + 1,
            best_val_macro_f1=best["best_score"],
            training_seconds=elapsed_before + time.monotonic() - start,
            peak_gpu_memory_gb=torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0,
        )
        summary["evaluation"]["validation"] = {
            "raw": compute_metrics(validation["labels"], probs, prepared.class_names)
        }
        write_json(run_dir / "summary.json", summary)
        print(f"Completed training: {run_dir}", flush=True)
    if world > 1:
        dist.barrier()
    return run_dir


def evaluate_run(run_dir, prepared, split="test", bootstrap_samples=None):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError("Evaluation requires a completed training run")
    config = summary["config"]
    count = (
        config["evaluation"]["bootstrap_samples"]
        if bootstrap_samples is None
        else bootstrap_samples
    )
    if not isinstance(count, int) or count < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    seed_everything(config["experiment"]["seed"], config["training"]["deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, transform, _ = build_transforms(config, prepared.root / "hub")
    model = build_model(config, len(prepared.class_names), prepared.root / "hub").to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    expected = json_hash(
        {
            "data_revision": config["data"]["revision"],
            "split_hash": prepared.manifest.get("split_hash")
            or json_hash({k: v.tolist() for k, v in prepared.splits.items()}),
            "class_names": prepared.class_names,
        }
    )
    if checkpoint["protocol_hash"] != expected:
        raise ValueError("Evaluation data protocol differs from training")
    load_model_state(model, checkpoint["model"])
    prediction = collect_predictions(
        model,
        make_loader(prepared.build_dataset(split, transform), config),
        device,
        config["training"]["precision"],
        config["training"]["max_eval_batches"],
    )
    probabilities = apply_temperature(prediction["logits"], 1.0)
    result = {"raw": compute_metrics(prediction["labels"], probabilities, prepared.class_names)}
    if config["evaluation"]["calibrate"]:
        val = collect_predictions(
            model,
            make_loader(prepared.build_dataset("validation", transform), config),
            device,
            config["training"]["precision"],
            config["training"]["max_eval_batches"],
        )
        temperature = fit_temperature(val["logits"], val["labels"])
        calibrated = apply_temperature(prediction["logits"], temperature)
        result.update(
            temperature=temperature,
            calibrated=compute_metrics(prediction["labels"], calibrated, prepared.class_names),
        )
        write_json(
            run_dir / "calibration.json", {"temperature": temperature, "fit_split": "validation"}
        )
    if count:
        result["bootstrap"] = bootstrap_metrics(
            prediction["labels"],
            probabilities,
            prepared.class_names,
            count,
            config["experiment"]["seed"],
        )
    np.savez_compressed(
        run_dir / f"{split}_predictions.npz", **prediction, probabilities=probabilities
    )
    summary["evaluation"][split] = result
    write_json(run_dir / "summary.json", summary)
    return result
