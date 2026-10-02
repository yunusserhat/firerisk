"""Strict, format-specific space checks before uncached model weight downloads.

The Hub's built-in disk check warns and continues. This guard preserves the
experiment's storage reserve while counting only the checkpoint selected by the
loader. Complete caches and offline/random/local workflows need no metadata API.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import _CACHED_NO_EXIST, HfApi, hf_hub_download, try_to_load_from_cache
from huggingface_hub import constants as hub_constants

from .storage import check_free_space


def _timm_candidates(filename: str) -> list[str]:
    """Match timm's safetensors preference without counting alternative weights."""
    if filename == "pytorch_model.bin":
        return ["model.safetensors", filename]
    if filename == "open_clip_pytorch_model.bin":
        return ["open_clip_model.safetensors", filename]
    if filename.endswith(".bin"):
        return [filename[:-4] + ".safetensors", filename]
    return [filename]


def _cache_status(repo_id: str, filename: str, revision: str, cache_dir: Path) -> Any:
    value = try_to_load_from_cache(repo_id, filename, revision=revision, cache_dir=cache_dir)
    if isinstance(value, str) and Path(value).is_file() and Path(value).stat().st_size > 0:
        return Path(value)
    return _CACHED_NO_EXIST if value is _CACHED_NO_EXIST else None


def _index_files(index: Path) -> list[str]:
    try:
        weight_map = json.loads(index.read_text())["weight_map"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid safetensors checkpoint index: {index}") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Checkpoint index has no weight_map: {index}")
    names = list(weight_map.values())
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid checkpoint shard name in {index}")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not name.endswith(".safetensors"):
            raise ValueError(f"Invalid safetensors shard name {name!r} in {index}")
    return sorted(set(names))


def preflight_model_download(
    repo_id: str,
    revision: str | None,
    cache_dir: str | Path | None = None,
    *,
    backend: str = "siglip",
    filename: str = "pytorch_model.bin",
    min_free_gb: float = 5.0,
    pretrained: bool = True,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Check missing bytes of one pinned checkpoint before its weight download.

    Transformers is configured with ``use_safetensors=True``; timm prefers the
    safetensors equivalent of its declared filename, then its declared format.
    Sharded Transformers checkpoints use only the index's referenced shards.
    Metadata/index downloads are skipped for fully cached checkpoints. Cache
    fallback selection respects known absent preferred files, so a cached .bin
    cannot mask an uncached safetensors file that timm would otherwise download.
    """
    if min_free_gb < 0:
        raise ValueError("min_free_gb must be nonnegative")
    reason = None
    if not pretrained:
        reason = "random_initialization"
    elif Path(repo_id).is_dir():
        reason = "local_checkpoint"
    elif (
        local_files_only
        or hub_constants.HF_HUB_OFFLINE
        or os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}
    ):
        reason = "offline"
    if reason:
        return {"required_bytes": 0, "skipped": reason}
    if backend not in {"siglip", "timm"}:
        raise ValueError(f"Unsupported checkpoint backend: {backend}")
    if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("Model download preflight requires an immutable Hub revision")
    cache_dir = Path(cache_dir or hub_constants.HF_HUB_CACHE).expanduser().resolve()
    candidates = ["model.safetensors"] if backend == "siglip" else _timm_candidates(filename)
    status = {name: _cache_status(repo_id, name, revision, cache_dir) for name in candidates}
    index_name = "model.safetensors.index.json"
    cached_index = (
        _cache_status(repo_id, index_name, revision, cache_dir) if backend == "siglip" else None
    )

    def result(files: list[str], missing: int, **extra: Any) -> dict[str, Any]:
        return {
            "repo_id": repo_id,
            "revision": revision,
            "files": files,
            "required_bytes": missing,
            "cache_dir": str(cache_dir),
            **extra,
        }

    # The first existing preferred format determines what the loader consumes.
    for index, name in enumerate(candidates):
        if isinstance(status[name], Path) and all(
            status[preferred] is _CACHED_NO_EXIST for preferred in candidates[:index]
        ):
            return result([name], 0, fully_cached=True)
    if isinstance(cached_index, Path) and status[candidates[0]] is _CACHED_NO_EXIST:
        shards = _index_files(cached_index)
        if all(
            isinstance(_cache_status(repo_id, name, revision, cache_dir), Path) for name in shards
        ):
            return result(shards, 0, fully_cached=True)

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if getattr(info, "sha", None) != revision:
        raise ValueError("Hub model metadata did not match the requested immutable revision")
    files = {item.rfilename: item for item in info.siblings}

    def size(name: str) -> int:
        item = files.get(name)
        value = getattr(item, "size", None)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"Hub did not report a valid size for {repo_id}@{revision}/{name}; download refused"
            )
        return value

    selected = next((name for name in candidates if name in files), None)
    if selected is not None:
        selected_files = [selected]
    elif backend == "siglip" and index_name in files:
        if not isinstance(cached_index, Path):
            # The small index is itself checked before downloading it, then its
            # exact shard set determines the subsequent weight-space check.
            check_free_space(cache_dir, size(index_name), min_free_gb)
            cached_index = Path(
                hf_hub_download(repo_id, index_name, revision=revision, cache_dir=cache_dir)
            )
        selected_files = _index_files(cached_index)
    else:
        raise ValueError(
            f"No supported checkpoint format declared for {repo_id}@{revision}; download refused"
        )

    missing_bytes = 0
    for name in selected_files:
        expected = size(name)
        cached = _cache_status(repo_id, name, revision, cache_dir)
        if isinstance(cached, Path):
            if cached.stat().st_size != expected:
                raise ValueError(
                    f"Cached checkpoint size mismatch for {cached}; expected {expected} bytes"
                )
        else:
            missing_bytes += expected
    if missing_bytes:
        check_free_space(cache_dir, missing_bytes, min_free_gb)
    return result(selected_files, missing_bytes, fully_cached=missing_bytes == 0)
