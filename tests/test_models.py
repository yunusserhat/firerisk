"""Offline tests exercise actual attention, adapters, pooling, and gradients."""

from __future__ import annotations

import copy
import json

import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from firerisk.models import build_model, build_transforms


@pytest.fixture
def tiny_siglip() -> dict:
    return {
        "backend": "siglip",
        "name": "offline-tiny-siglip",
        "pretrained": False,
        "mode": "full",
        "dropout": 0.0,
        "image_size": 32,
        "vision_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 8,
            "attention_dropout": 0.0,
        },
    }


def test_siglip_full_updates_vision_weights(tiny_siglip):
    torch.manual_seed(7)
    model = build_model({"model": tiny_siglip}, num_classes=7)
    model.train()
    weight = model.backbone.vision_model.embeddings.patch_embedding.weight
    before = weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 7)
    F.cross_entropy(logits, torch.tensor([1, 5])).backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    optimizer.step()
    assert not torch.equal(before, weight)
    assert not any("text" in name for name, _ in model.named_parameters())
    assert model.metadata["trainable_parameters"] == model.metadata["total_parameters"]
    json.dumps(model.metadata)


def test_linear_probe_changes_head_only_and_preserves_eval(tiny_siglip):
    tiny_siglip["mode"] = "linear"
    model = build_model(tiny_siglip, num_classes=7)
    before = {name: tensor.clone() for name, tensor in model.backbone.state_dict().items()}
    head_before = model.head[-1].weight.detach().clone()
    model.train()
    assert model.training and model.head.training
    assert not model.backbone.training
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    F.cross_entropy(model(torch.randn(2, 3, 32, 32)), torch.tensor([0, 6])).backward()
    optimizer.step()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(
        torch.equal(before[name], tensor) for name, tensor in model.backbone.state_dict().items()
    )
    assert not torch.equal(head_before, model.head[-1].weight)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_siglip_lora_receives_gradients_with_frozen_embeddings(tiny_siglip, checkpointing):
    tiny_siglip.update(mode="lora", lora_rank=2, lora_alpha=4, gradient_checkpointing=checkpointing)
    model = build_model(tiny_siglip, num_classes=7).train()
    named = dict(model.backbone.named_parameters())
    trainable = {name: param for name, param in named.items() if param.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    assert 0 < model.metadata["trainable_parameters"] < model.metadata["total_parameters"]
    # Keep the pretrained multihead pooling projection intact.
    assert not model.backbone.vision_model.head.attention.in_proj_weight.requires_grad
    before = {name: param.detach().clone() for name, param in trainable.items()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    F.cross_entropy(model(torch.randn(2, 3, 32, 32)), torch.tensor([2, 4])).backward()
    assert all(param.grad is not None for param in trainable.values())
    assert all(torch.isfinite(param.grad).all() for param in trainable.values())
    optimizer.step()
    assert any(not torch.equal(before[name], param) for name, param in trainable.items())
    assert all(param.grad is None for name, param in named.items() if "lora_" not in name)


def test_siglip_resolution_interpolates_positions(tiny_siglip):
    tiny_siglip["image_size"] = 64
    model = build_model(tiny_siglip, num_classes=7).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 7)


def test_offline_siglip_preprocessing_matches_official_processor(tiny_siglip):
    train_transform, eval_transform, metadata = build_transforms(tiny_siglip)
    image = Image.new("RGB", (47, 53), (127, 0, 255))
    result = eval_transform(image)
    official = eval_transform.processor(
        images=image, size={"height": 32, "width": 32}, return_tensors="pt"
    )["pixel_values"][0]
    assert torch.equal(result, official)
    assert result.shape == (3, 32, 32)
    assert result.dtype == torch.float32
    assert metadata["mean"] == metadata["std"] == [0.5, 0.5, 0.5]
    assert metadata["interpolation"] == "bilinear"
    assert torch.equal(result, eval_transform(image))
    assert train_transform(image.convert("L")).shape == (3, 32, 32)
    json.dumps(metadata)


def test_local_siglip_checkpoint_loads_without_network(tiny_siglip, tmp_path):
    from transformers import SiglipImageProcessor

    source = build_model(tiny_siglip, num_classes=7)
    source.backbone.save_pretrained(tmp_path)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(tmp_path)
    config = copy.deepcopy(tiny_siglip)
    config.update(name=str(tmp_path), pretrained=True, local_files_only=True)
    config.pop("vision_config")
    restored = build_model(config, num_classes=7)
    assert torch.equal(
        source.backbone.vision_model.embeddings.patch_embedding.weight,
        restored.backbone.vision_model.embeddings.patch_embedding.weight,
    )
    _, eval_transform, metadata = build_transforms(config)
    assert eval_transform(Image.new("RGB", (32, 32))).shape == (3, 32, 32)
    json.dumps(metadata)


def test_timm_linear_probe_freezes_batchnorm_buffers():
    config = {
        "backend": "timm",
        "name": "resnet18.a1_in1k",
        "pretrained": False,
        "mode": "linear",
        "image_size": 32,
        "dropout": 0.0,
    }
    model = build_model(config, num_classes=7).train()
    before = model.backbone.bn1.running_mean.clone()
    F.cross_entropy(model(torch.randn(2, 3, 32, 32)), torch.tensor([1, 2])).backward()
    assert not model.backbone.training
    assert torch.equal(before, model.backbone.bn1.running_mean)
    assert all(param.grad is None for param in model.backbone.parameters())
    train_transform, eval_transform, metadata = build_transforms(config)
    image = Image.new("RGB", (53, 47), (75, 150, 30))
    assert train_transform(image).shape == eval_transform(image).shape == (3, 32, 32)
    json.dumps(metadata)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_timm_lora_updates_fused_qkv_adapters(monkeypatch, checkpointing):
    import timm
    from timm.models.vision_transformer import VisionTransformer

    # Use timm's actual attention implementation at a small scale.
    def tiny_create_model(*args, **kwargs):
        return VisionTransformer(
            img_size=32, patch_size=8, embed_dim=32, depth=2, num_heads=4, num_classes=0
        )

    monkeypatch.setattr(timm, "create_model", tiny_create_model)
    config = {
        "backend": "timm",
        "name": "vit_tiny_patch16_224",
        "pretrained": False,
        "mode": "lora",
        "image_size": 32,
        "dropout": 0.0,
        "lora_rank": 2,
        "lora_alpha": 4,
        "gradient_checkpointing": checkpointing,
    }
    model = build_model(config, num_classes=7).train()
    trainable = {n: p for n, p in model.backbone.named_parameters() if p.requires_grad}
    assert trainable and all("qkv.lora_" in name for name in trainable)
    before = {name: param.detach().clone() for name, param in trainable.items()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    F.cross_entropy(model(torch.randn(2, 3, 32, 32)), torch.tensor([1, 2])).backward()
    optimizer.step()
    assert all(param.grad is not None for param in trainable.values())
    assert any(not torch.equal(before[name], param) for name, param in trainable.items())


@pytest.mark.parametrize("backend", ["siglip", "timm"])
@pytest.mark.parametrize("mode", ["full", "linear", "lora"])
def test_naflex_fails_with_actionable_message(mode, backend):
    with pytest.raises(ValueError, match="NaFlex"):
        build_model(
            {"backend": backend, "name": "google/siglip2-base-patch16-naflex", "mode": mode}, 7
        )


def test_timm_revision_is_resolved_before_loading_and_applied_to_weights(monkeypatch):
    import huggingface_hub
    import timm
    from timm.models.vision_transformer import VisionTransformer

    immutable = "abc123" * 6 + "abcd"
    seen = {}

    def fake_download(**kwargs):
        seen["requested"] = kwargs
        return f"/cache/models--timm--vit_base/snapshots/{immutable}/config.json"

    def tiny_create(*args, **kwargs):
        seen["created"] = kwargs
        return VisionTransformer(
            img_size=32, patch_size=8, embed_dim=32, depth=1, num_heads=4, num_classes=0
        )

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(timm, "create_model", tiny_create)
    monkeypatch.setattr("firerisk.models.preflight_model_download", lambda *args, **kwargs: {})
    cfg = {
        "backend": "timm",
        "name": "vit_base_patch16_siglip_224.v2_webli",
        "revision": "requested-tag",
        "image_size": 32,
    }
    model = build_model(cfg, 7, "/task-cache")
    assert seen["requested"]["revision"] == "requested-tag"
    assert seen["requested"]["cache_dir"] == "/task-cache"
    assert seen["created"]["pretrained_cfg_overlay"]["hf_hub_id"].endswith("@" + immutable)
    assert model.metadata["resolved_revision"] == immutable


def test_timm_model_preflight_receives_the_full_config_storage_reserve(monkeypatch):
    import timm
    from timm.models.vision_transformer import VisionTransformer

    immutable = "f" * 40
    checked = []
    monkeypatch.setattr("firerisk.models._timm_revision", lambda *args: ({}, immutable))
    monkeypatch.setattr(
        "firerisk.models.preflight_model_download",
        lambda *args, **kwargs: checked.append((args, kwargs)),
    )
    monkeypatch.setattr(
        timm,
        "create_model",
        lambda *args, **kwargs: VisionTransformer(
            img_size=32, patch_size=8, embed_dim=32, depth=1, num_heads=4, num_classes=0
        ),
    )
    cfg = {
        "model": {
            "backend": "timm",
            "name": "vit_base_patch16_siglip_224.v2_webli",
            "image_size": 32,
        },
        "storage": {"min_free_gb": 7.5},
    }
    build_model(cfg, 7, "/task-cache")
    assert checked[0][1]["min_free_gb"] == 7.5
    assert checked[0][1]["backend"] == "timm"
    assert checked[0][0][1] == immutable


def test_siglip_model_preflight_receives_the_full_config_storage_reserve(tiny_siglip, monkeypatch):
    from transformers import SiglipVisionConfig, SiglipVisionModel

    vision_config = SiglipVisionConfig(**tiny_siglip["vision_config"])
    vision_config._commit_hash = "f" * 40
    checked = []
    monkeypatch.setattr("firerisk.models._siglip_config", lambda *args: vision_config)
    monkeypatch.setattr(
        SiglipVisionModel,
        "from_pretrained",
        lambda *args, **kwargs: SiglipVisionModel(vision_config),
    )
    monkeypatch.setattr(
        "firerisk.models.preflight_model_download",
        lambda *args, **kwargs: checked.append((args, kwargs)),
    )
    tiny_siglip["pretrained"] = True
    build_model({"model": tiny_siglip, "storage": {"min_free_gb": 9.0}}, 7, "/task-cache")
    assert checked[0][1]["min_free_gb"] == 9.0
    assert checked[0][1]["backend"] == "siglip"
    assert checked[0][0][1] == vision_config._commit_hash


def test_invalid_image_patch_size_is_rejected(tiny_siglip):
    tiny_siglip["image_size"] = 30
    with pytest.raises(ValueError, match="divisible"):
        build_model(tiny_siglip, 7)


def test_native_siglip_size_need_not_be_divisible_by_patch_size(tiny_siglip):
    # Official SigLIP2 SO400M uses 384px with 14px patches. Its pretrained
    # native grid intentionally floors the patch count; custom grids must
    # still meet our divisibility rule to make interpolation unambiguous.
    tiny_siglip["image_size"] = 30
    tiny_siglip["vision_config"]["image_size"] = 30
    model = build_model(tiny_siglip, num_classes=7).eval()
    _, eval_transform, metadata = build_transforms(tiny_siglip)
    pixels = eval_transform(Image.new("RGB", (39, 45), (75, 120, 50))).unsqueeze(0)
    assert pixels.shape == (1, 3, 30, 30)
    with torch.no_grad():
        logits = model(pixels)
    assert logits.shape == (1, 7) and torch.isfinite(logits).all()
    assert metadata["image_size"] == 30


def test_timm_convolutional_lora_is_rejected():
    with pytest.raises(ValueError, match="qkv"):
        build_model(
            {"backend": "timm", "name": "resnet18.a1_in1k", "pretrained": False, "mode": "lora"}, 7
        )
