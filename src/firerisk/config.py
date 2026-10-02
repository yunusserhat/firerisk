"""Validated YAML configurations with explicit inheritance and dotted overrides."""

import math
import re
from copy import deepcopy
from pathlib import Path

import yaml

DEFAULTS = {
    "experiment": {"name": "siglip2-base-full", "seed": 42},
    "storage": {"root": None, "min_free_gb": 5.0},
    "data": {
        "repo_id": "blanchon/FireRisk",
        "revision": "234b2e7fe6be2da773472e83bd4d42cc9815a630",
        "split_seed": 2026,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "deduplicate": True,
        "audit_hash": "pixels",
        "max_shards": None,
        "cache_row_groups": 2,
        "access_mode": "packed",
    },
    "model": {
        "backend": "timm",
        "name": "vit_base_patch16_siglip_224.v2_webli",
        "revision": None,
        "mode": "full",
        "image_size": 224,
        "dropout": 0.1,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "gradient_checkpointing": True,
        "pretrained": True,
    },
    "training": {
        "epochs": 30,
        "batch_size": 32,
        "eval_batch_size": 64,
        "accumulation_steps": 2,
        "workers": 4,
        "lr_backbone": 1e-5,
        "lr_head": 3e-4,
        "weight_decay": 0.05,
        "warmup_fraction": 0.1,
        "label_smoothing": 0.05,
        "class_balance": "none",
        "clip_grad_norm": 1.0,
        "precision": "bf16",
        "patience": 7,
        "min_delta": 0.0001,
        "deterministic": True,
        "save_last": True,
        "max_train_batches": None,
        "max_eval_batches": None,
        "mixup_alpha": 0.0,
    },
    "evaluation": {"bootstrap_samples": 1000, "calibrate": True},
}


def merge(base, update, prefix=""):
    result = deepcopy(base)
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be a mapping")
            result[key] = merge(base[key], value, f"{prefix}{key}.")
        else:
            result[key] = value
    return result


def load_config(path=None, overrides=(), _seen=None):
    config = deepcopy(DEFAULTS)
    if path:
        path = Path(path).resolve()
        seen = set() if _seen is None else _seen
        if path in seen:
            raise ValueError(f"Circular config inheritance: {path}")
        seen.add(path)
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("Configuration YAML must be a mapping")
        parent = raw.pop("extends", None)
        if parent:
            config = load_config(path.parent / parent, _seen=seen)
        config = merge(config, raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value: {item}")
        dotted, value = item.split("=", 1)
        nested = yaml.safe_load(value)
        for key in reversed(dotted.split(".")):
            nested = {key: nested}
        config = merge(config, nested)
    validate_config(config)
    return config


def validate_config(c):
    t, d, m = c["training"], c["data"], c["model"]
    if not isinstance(c["experiment"]["name"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", c["experiment"]["name"]
    ):
        raise ValueError("experiment.name must be a safe directory basename")
    for key in ["seed"]:
        if type(c["experiment"][key]) is not int or c["experiment"][key] < 0:
            raise ValueError("experiment.seed must be a nonnegative integer")
    for key in ["epochs", "batch_size", "eval_batch_size", "accumulation_steps", "patience"]:
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if type(t["workers"]) is not int or t["workers"] < 0:
        raise ValueError("training.workers must be nonnegative")
    if not (
        0 < d["val_fraction"] < 1
        and 0 < d["test_fraction"] < 1
        and d["val_fraction"] + d["test_fraction"] < 1
    ):
        raise ValueError("Validation and test fractions must be positive and sum to < 1")
    if m["mode"] not in {"full", "linear", "lora"}:
        raise ValueError("model.mode must be full, linear, or lora")
    if m["backend"] not in {"siglip", "timm"}:
        raise ValueError("model.backend must be siglip or timm")
    if d["access_mode"] not in {"packed", "parquet"}:
        raise ValueError("data.access_mode must be packed or parquet")
    if type(d["cache_row_groups"]) is not int or d["cache_row_groups"] < 1:
        raise ValueError("data.cache_row_groups must be a positive integer")
    if d["max_shards"] is not None and (type(d["max_shards"]) is not int or d["max_shards"] < 1):
        raise ValueError("data.max_shards must be a positive integer or null")
    if t["precision"] not in {"bf16", "fp16", "fp32"}:
        raise ValueError("training.precision must be bf16, fp16, or fp32")
    if t["class_balance"] not in {"none", "weighted_loss", "weighted_sampler"}:
        raise ValueError("Use one imbalance treatment: none, weighted_loss, weighted_sampler")
    for key in ["max_train_batches", "max_eval_batches"]:
        if t[key] is not None and (type(t[key]) is not int or t[key] < 1):
            raise ValueError(f"training.{key} must be positive or null")
    if not 0 <= t["label_smoothing"] < 1 or not 0 <= t["warmup_fraction"] < 1:
        raise ValueError("Smoothing and warmup fractions must be in [0,1)")
    if t["lr_backbone"] <= 0 or t["lr_head"] <= 0 or t["mixup_alpha"] < 0:
        raise ValueError("Learning rates must be positive; mixup must be nonnegative")
    for key in [
        "lr_backbone",
        "lr_head",
        "weight_decay",
        "clip_grad_norm",
        "mixup_alpha",
        "min_delta",
        "warmup_fraction",
        "label_smoothing",
    ]:
        if not isinstance(t[key], (int, float)) or not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative")
    if t["clip_grad_norm"] == 0:
        raise ValueError("training.clip_grad_norm must be positive")
    if (
        not isinstance(c["storage"]["min_free_gb"], (int, float))
        or not math.isfinite(c["storage"]["min_free_gb"])
        or c["storage"]["min_free_gb"] < 0
    ):
        raise ValueError("storage.min_free_gb must be finite and nonnegative")
    if (
        type(c["evaluation"]["bootstrap_samples"]) is not int
        or c["evaluation"]["bootstrap_samples"] < 0
    ):
        raise ValueError("bootstrap_samples must be nonnegative")
