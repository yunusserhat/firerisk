"""Format-specific model download checks use mocked Hub metadata and disk state."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from firerisk import model_cache

SHA = "0123456789abcdef" * 2 + "01234567"
REPO = "test/vision-checkpoint"


@pytest.fixture
def hub(monkeypatch, tmp_path):
    cached = {}
    metadata_calls = []
    disk_calls = []
    downloads = []
    state = SimpleNamespace(
        cached=cached,
        metadata_calls=metadata_calls,
        disk_calls=disk_calls,
        downloads=downloads,
        sizes={"model.safetensors": 100},
        sha=SHA,
        cache_dir=tmp_path,
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(model_cache.hub_constants, "HF_HUB_OFFLINE", False)

    def cache_lookup(repo_id, filename, revision, cache_dir):
        assert repo_id == REPO and revision == SHA and Path(cache_dir) == tmp_path
        return cached.get(filename)

    def info(repo_id, revision, files_metadata):
        assert repo_id == REPO and revision == SHA and files_metadata
        metadata_calls.append(repo_id)
        return SimpleNamespace(
            sha=state.sha,
            siblings=[
                SimpleNamespace(rfilename=name, size=size) for name, size in state.sizes.items()
            ],
        )

    def check(path, required_bytes, min_free_gb):
        disk_calls.append((Path(path), required_bytes, min_free_gb))

    def unexpected_download(*args, **kwargs):
        downloads.append((args, kwargs))
        raise AssertionError("The preflight may download a small index, never model weights")

    monkeypatch.setattr(model_cache, "try_to_load_from_cache", cache_lookup)
    monkeypatch.setattr(model_cache, "HfApi", lambda: SimpleNamespace(model_info=info))
    monkeypatch.setattr(model_cache, "check_free_space", check)
    monkeypatch.setattr(model_cache, "hf_hub_download", unexpected_download)
    return state


def _run(hub, **kwargs):
    return model_cache.preflight_model_download(REPO, SHA, hub.cache_dir, **kwargs)


def _cache(hub, name, content):
    path = hub.cache_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    hub.cached[name] = str(path)
    return path


def _index(hub, names):
    content = json.dumps(
        {"weight_map": {f"layer.{i}": name for i, name in enumerate(names)}}
    ).encode()
    return _cache(hub, "model.safetensors.index.json", content)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"pretrained": False}, "random_initialization"),
        ({"local_files_only": True}, "offline"),
    ],
)
def test_random_and_offline_workflows_make_no_metadata_requests(hub, kwargs, reason):
    result = _run(hub, **kwargs)
    assert result["skipped"] == reason
    assert hub.metadata_calls == hub.disk_calls == hub.downloads == []


def test_offline_environment_skips_metadata(hub, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert _run(hub)["skipped"] == "offline"
    assert not hub.metadata_calls


def test_local_checkpoint_skips_metadata(hub):
    result = model_cache.preflight_model_download(str(hub.cache_dir), None)
    assert result["skipped"] == "local_checkpoint"
    assert not hub.metadata_calls


def test_fully_cached_canonical_weights_skip_metadata_and_download_checks(hub):
    _cache(hub, "model.safetensors", b"cached")
    result = _run(hub, min_free_gb=7.5)
    assert result["fully_cached"] and result["required_bytes"] == 0
    assert result["files"] == ["model.safetensors"]
    assert hub.metadata_calls == hub.disk_calls == hub.downloads == []


@pytest.mark.parametrize("backend", ["siglip", "timm"])
def test_exact_canonical_missing_bytes_ignore_alternate_formats_and_quantizations(hub, backend):
    hub.sizes = {
        "model.safetensors": 120,
        "pytorch_model.bin": 240,
        "model.quantized.safetensors": 40,
        "optimizer.pt": 1000,
    }
    result = _run(hub, backend=backend, min_free_gb=7.5)
    assert result["required_bytes"] == 120 and result["files"] == ["model.safetensors"]
    assert hub.disk_calls == [(hub.cache_dir, 120, 7.5)]
    assert hub.downloads == []


def test_strict_disk_guard_raises_before_any_weights_download(hub, monkeypatch):
    from firerisk.storage import check_free_space

    monkeypatch.setattr(model_cache, "check_free_space", check_free_space)
    monkeypatch.setattr(
        "firerisk.storage.inspect_storage",
        lambda path: {
            "root": str(path),
            "free_bytes": 99,
            "free_gib": 99 / 2**30,
        },
    )
    with pytest.raises(OSError, match="Insufficient storage"):
        _run(hub, min_free_gb=0)
    assert hub.downloads == []


def test_reserve_is_enforced_in_addition_to_download_size(hub, monkeypatch):
    from firerisk.storage import check_free_space

    monkeypatch.setattr(model_cache, "check_free_space", check_free_space)
    monkeypatch.setattr(
        "firerisk.storage.inspect_storage",
        lambda path: {
            "root": str(path),
            "free_bytes": 2**30 + 99,
            "free_gib": 1.0,
        },
    )
    with pytest.raises(OSError, match="1.00 GiB reserved"):
        _run(hub, min_free_gb=1)
    assert hub.downloads == []


@pytest.mark.parametrize("size", [None, 0, -1])
def test_unknown_or_invalid_selected_size_refuses_download(hub, size):
    hub.sizes = {"model.safetensors": size}
    with pytest.raises(ValueError, match="valid size"):
        _run(hub)
    assert hub.disk_calls == hub.downloads == []


def test_metadata_must_match_the_immutable_revision(hub):
    hub.sha = "f" * 40
    with pytest.raises(ValueError, match="immutable revision"):
        _run(hub)
    assert hub.disk_calls == hub.downloads == []


def test_moving_revision_is_rejected_before_metadata_request(hub):
    with pytest.raises(ValueError, match="immutable Hub revision"):
        model_cache.preflight_model_download(REPO, "main", hub.cache_dir)
    assert hub.metadata_calls == []


def test_partially_cached_shards_count_only_uncached_referenced_files(hub):
    first, second = "model-00001.safetensors", "model-00002.safetensors"
    index = _index(hub, [first, second, first])
    _cache(hub, first, b"1234")
    hub.cached["model.safetensors"] = model_cache._CACHED_NO_EXIST
    hub.sizes = {
        "model.safetensors.index.json": index.stat().st_size,
        first: 4,
        second: 6,
        "other-format.safetensors": 500,
        "pytorch_model.bin": 1000,
    }
    result = _run(hub, min_free_gb=3.5)
    assert result["files"] == [first, second] and result["required_bytes"] == 6
    assert hub.disk_calls == [(hub.cache_dir, 6, 3.5)]
    assert hub.downloads == []


def test_fully_cached_index_and_shards_need_no_metadata(hub):
    first, second = "model-00001.safetensors", "model-00002.safetensors"
    _index(hub, [first, second])
    _cache(hub, first, b"1234")
    _cache(hub, second, b"123456")
    hub.cached["model.safetensors"] = model_cache._CACHED_NO_EXIST
    assert _run(hub)["fully_cached"]
    assert hub.metadata_calls == hub.disk_calls == hub.downloads == []


def test_uncached_index_is_preflighted_then_only_its_referenced_shards(hub, monkeypatch):
    first, second = "model-00001.safetensors", "model-00002.safetensors"
    encoded = json.dumps({"weight_map": {"a": first, "b": second}}).encode()
    hub.sizes = {
        "model.safetensors.index.json": len(encoded),
        first: 4,
        second: 6,
        "model.4bit.safetensors": 500,
    }

    def download_index(repo_id, filename, revision, cache_dir):
        assert filename == "model.safetensors.index.json" and repo_id == REPO and revision == SHA
        assert hub.disk_calls == [(hub.cache_dir, len(encoded), 5.0)]
        hub.downloads.append(filename)
        return str(_cache(hub, filename, encoded))

    monkeypatch.setattr(model_cache, "hf_hub_download", download_index)
    result = _run(hub)
    assert result["required_bytes"] == 10
    assert hub.disk_calls == [(hub.cache_dir, len(encoded), 5.0), (hub.cache_dir, 10, 5.0)]
    assert hub.downloads == ["model.safetensors.index.json"]


def test_declared_timm_filename_respects_its_safetensors_equivalent(hub):
    hub.sizes = {"vision.bin": 140, "vision.safetensors": 70, "model.safetensors": 999}
    result = _run(hub, backend="timm", filename="vision.bin")
    assert result["files"] == ["vision.safetensors"] and result["required_bytes"] == 70


def test_timm_bin_fallback_is_guarded_when_safetensors_are_absent(hub):
    hub.sizes = {"pytorch_model.bin": 140}
    result = _run(hub, backend="timm")
    assert result["files"] == ["pytorch_model.bin"] and result["required_bytes"] == 140


def test_cached_timm_fallback_skips_metadata_when_preferred_format_known_absent(hub):
    _cache(hub, "pytorch_model.bin", b"1234")
    hub.cached["model.safetensors"] = model_cache._CACHED_NO_EXIST
    assert _run(hub, backend="timm")["fully_cached"]
    assert hub.metadata_calls == hub.disk_calls == hub.downloads == []


def test_cached_bin_cannot_hide_an_uncached_preferred_safetensors_file(hub):
    _cache(hub, "pytorch_model.bin", b"1234")
    hub.sizes = {"model.safetensors": 80, "pytorch_model.bin": 4}
    result = _run(hub, backend="timm")
    assert result["files"] == ["model.safetensors"] and result["required_bytes"] == 80


@pytest.mark.parametrize(
    "index", [{}, {"weight_map": {}}, {"weight_map": {"a": "../bad.safetensors"}}]
)
def test_invalid_checkpoint_indices_fail_before_download(hub, index):
    _cache(hub, "model.safetensors.index.json", json.dumps(index).encode())
    hub.cached["model.safetensors"] = model_cache._CACHED_NO_EXIST
    with pytest.raises(ValueError, match="index|shard"):
        _run(hub)
    assert hub.downloads == []


def test_missing_shard_size_refuses_download(hub):
    index = _index(hub, ["model-00001.safetensors"])
    hub.sizes = {"model.safetensors.index.json": index.stat().st_size}
    with pytest.raises(ValueError, match="valid size"):
        _run(hub)
    assert hub.disk_calls == hub.downloads == []


def test_cached_shard_size_mismatch_is_reported_without_cleanup(hub):
    first, second = "model-00001.safetensors", "model-00002.safetensors"
    index = _index(hub, [first, second])
    corrupted = _cache(hub, first, b"12")
    hub.sizes = {"model.safetensors.index.json": index.stat().st_size, first: 4, second: 6}
    with pytest.raises(ValueError, match="size mismatch"):
        _run(hub)
    assert corrupted.read_bytes() == b"12"
    assert hub.downloads == []
