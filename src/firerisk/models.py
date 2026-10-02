"""Vision-only classifiers and checkpoint-specific image preprocessing.

The fixed-resolution SigLIP2 checkpoints use the ``SiglipVisionModel``
architecture. NaFlex has a different input representation and is deliberately
rejected instead of silently treating patches as ordinary image tensors.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model_cache import preflight_model_download


def _model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Accept either the whole experiment configuration or its model section."""
    model = config.get("model", config)
    if not isinstance(model, Mapping):
        raise TypeError("model configuration must be a mapping")
    return dict(model)


def _validate(config: dict[str, Any]) -> tuple[str, str, str]:
    backend = str(config.get("backend", "siglip"))
    name = str(config.get("name", "google/siglip2-base-patch16-224"))
    mode = str(config.get("mode", "full"))
    if backend not in {"siglip", "timm"}:
        raise ValueError(f"Unsupported backend {backend!r}; use siglip or timm")
    if mode not in {"linear", "full", "lora"}:
        raise ValueError(f"Unsupported mode {mode!r}; use linear, full, or lora")
    if "naflex" in name.lower():
        raise ValueError("SigLIP2 NaFlex is not supported; select a FixRes checkpoint")
    if config.get("image_size") is not None and int(config["image_size"]) < 1:
        raise ValueError("image_size must be positive")
    dropout = float(config.get("dropout", 0.1))
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    return backend, name, mode


class VisionClassifier(nn.Module):
    """Keep a shared classifier API for transformer and timm backbones."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        *,
        backend: str,
        mode: str,
        dropout: float,
        metadata: dict[str, Any],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Dropout(dropout), nn.Linear(feature_dim, num_classes)
        )
        self.backend = backend
        self.mode = mode
        self.metadata = metadata
        if mode == "linear":
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> VisionClassifier:
        super().train(mode)
        # A linear probe must freeze BatchNorm buffers and stochastic layers as
        # well as weights. Parent training loops routinely call model.train().
        if self.mode == "linear":
            self.backbone.eval()
        return self

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.backend == "siglip":
            native_size = int(self.backbone.config.image_size)
            interpolate = tuple(pixel_values.shape[-2:]) != (native_size, native_size)
            outputs = self.backbone(
                pixel_values=pixel_values,
                interpolate_pos_encoding=interpolate,
                return_dict=True,
            )
            # SigLIP's learned attention pooling is part of the pretrained
            # vision backbone. An arbitrary token mean discards that pooling.
            return outputs.pooler_output
        return self.backbone(pixel_values)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear":
            with torch.no_grad():
                features = self._features(pixel_values)
        else:
            features = self._features(pixel_values)
        return self.head(features)


def _siglip_config(config: dict[str, Any], cache_dir: str | Path | None) -> Any:
    from transformers import AutoConfig, SiglipVisionConfig

    if not bool(config.get("pretrained", True)) and config.get("vision_config"):
        return SiglipVisionConfig(**config["vision_config"])
    whole_config = AutoConfig.from_pretrained(
        config.get("name", "google/siglip2-base-patch16-224"),
        revision=config.get("revision"),
        cache_dir=cache_dir,
        local_files_only=bool(config.get("local_files_only", False)),
        trust_remote_code=False,
    )
    if whole_config.model_type not in {"siglip", "siglip_vision_model"}:
        raise ValueError(
            f"Expected SigLIP FixRes architecture, received {whole_config.model_type!r}. "
            "SigLIP2 NaFlex is not supported."
        )
    vision_config = getattr(whole_config, "vision_config", whole_config)
    # Nested config objects do not consistently inherit this Hub provenance.
    vision_config._commit_hash = getattr(whole_config, "_commit_hash", None)
    return vision_config


def _timm_config(name: str) -> dict[str, Any]:
    import timm

    pretrained_config = timm.models.get_pretrained_cfg(name)
    if pretrained_config is None:
        raise ValueError(f"No pretrained preprocessing configuration for timm model {name!r}")
    return pretrained_config.to_dict()


def _timm_revision(
    config: dict[str, Any], pretrained_config: dict[str, Any], cache_dir: str | Path | None
) -> tuple[dict[str, Any], str | None]:
    """Resolve before loading so weights and recorded provenance use one commit."""
    if not bool(config.get("pretrained", True)):
        return {}, None
    repo_id = pretrained_config.get("hf_hub_id")
    if not repo_id:
        if config.get("revision"):
            raise ValueError("This timm checkpoint does not support a Hugging Face revision")
        return {}, None
    from huggingface_hub import hf_hub_download

    cached_config = Path(
        hf_hub_download(
            repo_id=str(repo_id).split("@")[0],
            filename="config.json",
            revision=config.get("revision"),
            cache_dir=cache_dir,
            local_files_only=bool(config.get("local_files_only", False)),
        )
    )
    # Hub snapshot paths are .../snapshots/<immutable SHA>/config.json.
    commit = cached_config.parent.name
    return {"hf_hub_id": f"{str(repo_id).split('@')[0]}@{commit}"}, commit


def _add_lora(backbone: nn.Module, config: dict[str, Any], backend: str) -> nn.Module:
    from peft import LoraConfig, inject_adapter_in_model

    rank = int(config.get("lora_rank", 8))
    if rank < 1:
        raise ValueError("lora_rank must be positive")
    targets = ["q_proj", "v_proj"] if backend == "siglip" else ["qkv"]
    if not any(name.split(".")[-1] in targets for name, _ in backbone.named_modules()):
        raise ValueError(
            f"No {targets} attention projections found; timm LoRA requires a qkv-based ViT"
        )
    adapter = LoraConfig(
        r=rank,
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=targets,
        bias="none",
    )
    # Preserve the image-only forward signature: PeftModel's generic
    # feature-extraction forward otherwise introduces input_ids.
    return inject_adapter_in_model(adapter, backbone)


def build_model(
    config: Mapping[str, Any], num_classes: int, cache_dir: str | Path | None = None
) -> VisionClassifier:
    """Build a vision-only classifier, including parameter-efficient adapters.

    ``pretrained=False`` and ``vision_config={...}`` create an entirely offline
    small SigLIP model, useful for integration tests. Public experiment presets
    should always use pretrained checkpoints.
    """
    reserve_gb = float(config.get("storage", {}).get("min_free_gb", 5.0))
    config = _model_config(config)
    backend, name, mode = _validate(config)
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    pretrained = bool(config.get("pretrained", True))
    resolved_revision = None
    if backend == "siglip":
        from transformers import SiglipVisionModel

        vision_config = _siglip_config(config, cache_dir)
        image_size = int(config.get("image_size") or vision_config.image_size)
        if image_size != int(vision_config.image_size) and image_size % int(
            vision_config.patch_size
        ):
            raise ValueError("Custom image_size must be divisible by the SigLIP patch_size")
        resolved_revision = getattr(vision_config, "_commit_hash", None)
        if pretrained:
            preflight_model_download(
                name,
                resolved_revision or config.get("revision"),
                cache_dir,
                backend=backend,
                min_free_gb=reserve_gb,
                local_files_only=bool(config.get("local_files_only", False)),
            )
            backbone = SiglipVisionModel.from_pretrained(
                name,
                config=vision_config,
                revision=resolved_revision or config.get("revision"),
                cache_dir=cache_dir,
                local_files_only=bool(config.get("local_files_only", False)),
                use_safetensors=True,
            )
        else:
            backbone = SiglipVisionModel(vision_config)
        feature_dim = int(vision_config.hidden_size)
        if mode == "lora":
            backbone = _add_lora(backbone, config, backend)
        if bool(config.get("gradient_checkpointing", False)) and mode != "linear":
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if mode == "lora":
                # Safe even when patch embeddings are frozen; this also keeps
                # adapter gradients working with older checkpoint internals.
                backbone.enable_input_require_grads()
    else:
        import timm

        pretrained_config = _timm_config(name)
        image_size = int(config.get("image_size") or pretrained_config["input_size"][-1])
        overlay, resolved_revision = _timm_revision(config, pretrained_config, cache_dir)
        if pretrained and pretrained_config.get("hf_hub_id"):
            preflight_model_download(
                str(pretrained_config["hf_hub_id"]).split("@")[0],
                resolved_revision,
                cache_dir,
                backend=backend,
                filename=pretrained_config.get("hf_hub_filename") or "pytorch_model.bin",
                min_free_gb=reserve_gb,
                local_files_only=bool(config.get("local_files_only", False)),
            )
        kwargs: dict[str, Any] = {}
        # timm interpolates pretrained positional embeddings while loading
        # these ViTs, including DINOv2's native 518px checkpoint at 224px.
        if name.split(".")[0].startswith(("vit_", "deit_", "beit_")):
            kwargs["img_size"] = image_size
        backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            pretrained_cfg_overlay=overlay,
            cache_dir=cache_dir,
            **kwargs,
        )
        feature_dim = int(backbone.num_features)
        if mode == "lora":
            backbone = _add_lora(backbone, config, backend)
        if bool(config.get("gradient_checkpointing", False)) and mode != "linear":
            if not hasattr(backbone, "set_grad_checkpointing"):
                raise ValueError(f"Gradient checkpointing is unavailable for {name}")
            backbone.set_grad_checkpointing(True)

    metadata = {
        "backend": backend,
        "name": name,
        "mode": mode,
        "pretrained": pretrained,
        "requested_revision": config.get("revision"),
        "resolved_revision": resolved_revision,
        "image_size": image_size,
        "feature_dim": feature_dim,
    }
    model = VisionClassifier(
        backbone,
        feature_dim,
        num_classes,
        backend=backend,
        mode=mode,
        dropout=float(config.get("dropout", 0.1)),
        metadata=metadata,
    )
    metadata["total_parameters"] = sum(p.numel() for p in model.parameters())
    metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model


class ConvertRGB:
    """Pickle-safe transform for multiprocessing DataLoaders."""

    def __call__(self, image: Any) -> Any:
        return image.convert("RGB")


class RandomRightAngleRotation:
    """Aerial images lack a preferred compass orientation; avoid padded corners."""

    def __call__(self, image: Any) -> Any:
        return image.rotate(90 * int(torch.randint(0, 4, ()).item()), expand=False)


class SiglipPreprocess:
    """Delegate resizing and normalization to the official checkpoint processor."""

    def __init__(self, processor: Any, image_size: int) -> None:
        self.processor = processor
        self.image_size = image_size

    def __call__(self, image: Any) -> torch.Tensor:
        return self.processor(
            images=image.convert("RGB"),
            size={"height": self.image_size, "width": self.image_size},
            return_tensors="pt",
        )["pixel_values"][0]


def build_transforms(
    config: Mapping[str, Any], cache_dir: str | Path | None = None
) -> tuple[Any, Any, dict[str, Any]]:
    """Return mild aerial training augmentation and deterministic official eval.

    Every transform maps a PIL image to a float32 CHW tensor. Evaluation never
    uses random cropping; augmentation settings are included in run metadata.
    """
    config = _model_config(config)
    backend, name, _ = _validate(config)
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if backend == "siglip":
        from transformers import AutoImageProcessor, SiglipImageProcessor

        vision_config = _siglip_config(config, cache_dir)
        image_size = int(config.get("image_size") or vision_config.image_size)
        if image_size != int(vision_config.image_size) and image_size % int(
            vision_config.patch_size
        ):
            raise ValueError("Custom image_size must be divisible by the SigLIP patch_size")
        resolved_revision = getattr(vision_config, "_commit_hash", None)
        if not bool(config.get("pretrained", True)) and config.get("vision_config"):
            processor = SiglipImageProcessor(
                size={"height": image_size, "width": image_size}, resample=2
            )
        else:
            processor = AutoImageProcessor.from_pretrained(
                name,
                revision=resolved_revision or config.get("revision"),
                cache_dir=cache_dir,
                local_files_only=bool(config.get("local_files_only", False)),
                use_fast=False,
                trust_remote_code=False,
            )
        mean, std = list(processor.image_mean), list(processor.image_std)
        # PIL's integer resampling values map directly to torchvision modes.
        interpolation = {
            0: "nearest",
            1: "lanczos",
            2: "bilinear",
            3: "bicubic",
            4: "box",
            5: "hamming",
        }[int(processor.resample)]
        eval_transform = SiglipPreprocess(processor, image_size)
        processor_metadata = processor.to_dict()
    else:
        import timm

        pretrained_config = _timm_config(name)
        image_size = int(config.get("image_size") or pretrained_config["input_size"][-1])
        _, resolved_revision = _timm_revision(config, pretrained_config, cache_dir)
        data_config = timm.data.resolve_data_config(
            {"input_size": (3, image_size, image_size)}, pretrained_cfg=pretrained_config
        )
        mean, std = list(data_config["mean"]), list(data_config["std"])
        interpolation = str(data_config["interpolation"])
        eval_transform = transforms.Compose(
            [ConvertRGB(), timm.data.create_transform(**data_config, is_training=False)]
        )
        processor_metadata = {**data_config, "input_size": [3, image_size, image_size]}

    train_transform = transforms.Compose(
        [
            ConvertRGB(),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode(interpolation),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRightAngleRotation(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    metadata = {
        "backend": backend,
        "name": name,
        "requested_revision": config.get("revision"),
        "resolved_revision": resolved_revision,
        "image_size": image_size,
        "input_size": [3, image_size, image_size],
        "mean": mean,
        "std": std,
        "interpolation": interpolation,
        "processor": processor_metadata,
        "train_augmentation": {
            "random_resized_crop_scale": [0.8, 1.0],
            "random_resized_crop_ratio": [0.9, 1.1],
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "right_angle_rotation": True,
        },
    }
    return train_transform, eval_transform, metadata
