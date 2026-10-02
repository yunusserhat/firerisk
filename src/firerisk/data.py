"""Pinned FireRisk parquet input, duplicate audits and reproducible held-out splits.

Images remain encoded in the original Hub parquet shards. For efficient shuffled
training an optional flat archive stores the same encoded bytes without decoding
or reencoding. No extracted image tree or Datasets Arrow cache is created.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps

from .storage import check_free_space, configure_cache_environment, resolve_storage_root

DEFAULT_REPO = "blanchon/FireRisk"
DEFAULT_REVISION = "234b2e7fe6be2da773472e83bd4d42cc9815a630"
DEFAULT_CLASSES = ["high", "low", "moderate", "non-burnable", "very_high", "very_low", "water"]
FORMAT_VERSION = 1


def _valid_packed_offsets(offsets: np.ndarray, rows: int, byte_size: int) -> bool:
    return bool(
        offsets.ndim == 1
        and np.issubdtype(offsets.dtype, np.integer)
        and len(offsets) == rows + 1
        and int(offsets[0]) == 0
        and int(offsets[-1]) == byte_size
        and np.all(np.diff(offsets) > 0)
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as handle:
        tmp = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _save_splits(path: Path, splits: dict[str, np.ndarray], labels: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".npz") as handle:
        tmp = Path(handle.name)
        np.savez_compressed(handle, labels=labels, **splits)
    os.replace(tmp, path)


def split_content_hash(splits: dict[str, np.ndarray], labels: np.ndarray) -> str:
    """Hash values rather than the NPZ container (whose ZIP timestamps can vary)."""
    digest = hashlib.sha256()
    for name, values in [("labels", labels), *sorted(splits.items())]:
        values = np.asarray(values, dtype="<i8")
        digest.update(name.encode())
        digest.update(len(values).to_bytes(8, "little"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict) and value.get("bytes") is not None:
        return value["bytes"]
    raise ValueError("Expected embedded image bytes in parquet; path-only images are unsupported")


def image_fingerprint(value: Any, mode: str = "encoded") -> str:
    encoded = _image_bytes(value)
    if mode == "encoded":
        return hashlib.sha256(encoded).hexdigest()
    if mode == "pixels":
        with Image.open(io.BytesIO(encoded)) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            digest = hashlib.sha256()
            digest.update(f"RGB:{rgb.width}:{rgb.height}:".encode())
            digest.update(rgb.tobytes())
            return digest.hexdigest()
    raise ValueError("audit_hash must be 'encoded', 'pixels' or 'none'")


def make_splits(
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    deduplicate: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Allocate image groups by class, balancing row counts without leakage.

    Exact duplicate groups with contradictory labels are rejected for explicit
    review. A deterministic largest-deficit allocation supports arbitrary split
    fractions and keeps each observed class represented in all three splits.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive and sum to less than one")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be a non-empty one-dimensional array")
    groups = np.arange(len(labels)) if groups is None else np.asarray(groups)
    if groups.shape != labels.shape:
        raise ValueError("groups must contain one identifier per row")
    members: dict[Any, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        members[group.item() if hasattr(group, "item") else group].append(index)
    by_class: dict[int, list[list[int]]] = defaultdict(list)
    duplicate_groups = 0
    duplicate_rows = 0
    for indices in members.values():
        class_labels = np.unique(labels[indices])
        if len(class_labels) != 1:
            raise ValueError(
                f"Duplicate image group has conflicting labels at rows {indices[:10]}; resolve the annotation conflict before training"
            )
        duplicate_groups += int(len(indices) > 1)
        duplicate_rows += len(indices) - 1
        by_class[int(class_labels[0])].append(indices[:1] if deduplicate else indices)
    rng = np.random.default_rng(seed)
    names = ["train", "validation", "test"]
    output: dict[str, list[int]] = {name: [] for name in names}
    fractions = np.array([1 - val_fraction - test_fraction, val_fraction, test_fraction])
    for label in sorted(by_class):
        rows = by_class[label]
        if len(rows) < 3:
            raise ValueError(
                f"Class {label} has only {len(rows)} unique images; at least three are required for train/validation/test"
            )
        rows = [rows[i] for i in rng.permutation(len(rows))]
        rows.sort(key=len, reverse=True)  # shuffled tie order remains stable
        targets = sum(map(len, rows)) * fractions
        counts = np.zeros(3, dtype=np.int64)
        for position, indices in enumerate(rows):
            missing = np.flatnonzero(counts == 0)
            remaining = len(rows) - position
            candidates = missing if len(missing) == remaining else np.arange(3)
            chosen = int(candidates[np.argmax((targets - counts)[candidates])])
            output[names[chosen]].extend(indices)
            counts[chosen] += len(indices)
    splits = {
        name: np.sort(np.asarray(indices, dtype=np.int64)) for name, indices in output.items()
    }
    group_sets = [{str(groups[index]) for index in indices} for indices in splits.values()]
    for left in range(3):
        for right in range(left + 1, 3):
            if group_sets[left] & group_sets[right]:
                raise AssertionError("Duplicate groups leaked across dataset splits")
    return splits, {
        "method": "class-stratified-largest-deficit-group-allocation-v1",
        "duplicate_groups": duplicate_groups,
        "duplicate_rows": duplicate_rows,
        "removed_duplicate_rows": duplicate_rows if deduplicate else 0,
        "unique_image_groups": len(members),
        "cross_split_group_overlap": 0,
        "requested_fractions": dict(zip(names, fractions.tolist())),
        "actual_fractions": {
            name: len(indices) / sum(map(len, splits.values())) for name, indices in splits.items()
        },
    }


class ParquetImageDataset(torch.utils.data.Dataset):
    """Map-style reader with a bounded, per-worker cache of encoded row groups."""

    def __init__(
        self,
        manifest: dict[str, Any],
        indices: np.ndarray,
        transform: Callable | None = None,
        cache_row_groups: int = 2,
    ):
        if cache_row_groups < 1:
            raise ValueError("cache_row_groups must be at least one")
        self.shards = manifest["shards"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.cache_row_groups = cache_row_groups
        self.offsets = np.array(
            [shard["row_offset"] for shard in self.shards] + [manifest["num_rows"]]
        )
        self._pid: int | None = None
        self._cache: OrderedDict = OrderedDict()
        self._files: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.update(_pid=None, _cache=OrderedDict(), _files=OrderedDict())
        return state

    def _get_row_group(self, shard_index: int, group_index: int):
        if self._pid != os.getpid():
            self._cache.clear()
            self._files.clear()
            self._pid = os.getpid()
        key = (shard_index, group_index)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if shard_index not in self._files:
            self._files[shard_index] = pq.ParquetFile(
                self.shards[shard_index]["local_path"], memory_map=True
            )
            while len(self._files) > self.cache_row_groups:
                self._files.popitem(last=False)
        self._files.move_to_end(shard_index)
        table = self._files[shard_index].read_row_group(group_index, columns=["image", "label"])
        self._cache[key] = table
        while len(self._cache) > self.cache_row_groups:
            self._cache.popitem(last=False)
        return table

    def __getitem__(self, position: int) -> dict[str, Any]:
        global_index = int(self.indices[position])
        shard_index = int(np.searchsorted(self.offsets, global_index, side="right") - 1)
        shard = self.shards[shard_index]
        local_index = global_index - shard["row_offset"]
        group_offsets = np.asarray(shard["row_group_offsets"])
        group_index = int(np.searchsorted(group_offsets, local_index, side="right") - 1)
        in_group = local_index - int(group_offsets[group_index])
        table = self._get_row_group(shard_index, group_index)
        value = table.column("image")[in_group].as_py()
        label = int(table.column("label")[in_group].as_py())
        with Image.open(io.BytesIO(_image_bytes(value))) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixels = self.transform(image) if self.transform is not None else image.copy()
        return {"pixel_values": pixels, "labels": label, "index": global_index}


class PackedImageDataset(torch.utils.data.Dataset):
    """Random image access using one memory-mapped archive of original bytes.

    Memory mapping lets the operating system share its page cache across workers;
    each sample reads only its own encoded image rather than a whole row group.
    """

    def __init__(
        self,
        manifest: dict[str, Any],
        indices: np.ndarray,
        labels: np.ndarray,
        transform: Callable | None = None,
    ):
        self.packed_path = manifest["packed_images"]["path"]
        self.offsets_path = manifest["packed_images"]["offsets_path"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        self._pid: int | None = None
        self._packed: np.memmap | None = None
        self._offsets: np.memmap | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.update(_pid=None, _packed=None, _offsets=None)
        return state

    def __getitem__(self, position: int) -> dict[str, Any]:
        if self._pid != os.getpid():
            self._packed = np.memmap(self.packed_path, dtype=np.uint8, mode="r")
            self._offsets = np.load(self.offsets_path, mmap_mode="r", allow_pickle=False)
            self._pid = os.getpid()
        index = int(self.indices[position])
        start, stop = int(self._offsets[index]), int(self._offsets[index + 1])
        encoded = self._packed[start:stop].tobytes()
        with Image.open(io.BytesIO(encoded)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixels = self.transform(image) if self.transform is not None else image.copy()
        return {"pixel_values": pixels, "labels": int(self.labels[index]), "index": index}


@dataclass
class PreparedData:
    root: Path
    manifest: dict[str, Any]
    labels: np.ndarray
    splits: dict[str, np.ndarray]

    @property
    def class_names(self) -> list[str]:
        return self.manifest["class_names"]

    @property
    def train_class_counts(self) -> np.ndarray:
        return np.bincount(self.labels[self.splits["train"]], minlength=len(self.class_names))

    def build_dataset(
        self, split: str, transform: Callable | None = None
    ) -> torch.utils.data.Dataset:
        if split == "val":
            split = "validation"
        if split not in self.splits:
            raise ValueError(f"Unknown split {split!r}; choose one of {list(self.splits)}")
        if self.manifest.get("access_mode") == "packed":
            return PackedImageDataset(self.manifest, self.splits[split], self.labels, transform)
        return ParquetImageDataset(
            self.manifest, self.splits[split], transform, self.manifest.get("cache_row_groups", 2)
        )


def _class_names(card: Any) -> list[str]:
    card = card.to_dict() if hasattr(card, "to_dict") else (card or {})
    info = card.get("dataset_info", {})
    if isinstance(info, list):
        info = info[0]
    for feature in info.get("features", []):
        if feature.get("name") == "label":
            dtype = feature.get("dtype", {})
            names = dtype.get("class_label", {}).get("names") if isinstance(dtype, dict) else None
            if isinstance(names, dict):
                return [names[str(i)] for i in range(len(names))]
            if names:
                return list(names)
    return DEFAULT_CLASSES.copy()


def _validate_cached(
    manifest: dict[str, Any], split_path: Path, identity: dict[str, Any]
) -> PreparedData | None:
    if manifest.get("preparation_hash") != _canonical_hash(identity) or not split_path.exists():
        return None
    for shard in manifest["shards"]:
        path = Path(shard["local_path"])
        if not path.is_file() or path.stat().st_size != shard["size_bytes"]:
            return None
    packed = manifest.get("packed_images")
    if packed:
        archive = Path(packed["path"])
        offsets_path = Path(packed["offsets_path"])
        if (
            not archive.is_file()
            or archive.stat().st_size != packed["size_bytes"]
            or not offsets_path.is_file()
        ):
            return None
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        if not _valid_packed_offsets(offsets, manifest["num_rows"], packed["size_bytes"]):
            raise ValueError(f"Packed archive index integrity check failed: {offsets_path}")
    with np.load(split_path, allow_pickle=False) as arrays:
        labels = arrays["labels"].copy()
        splits = {name: arrays[name].copy() for name in ["train", "validation", "test"]}
    if split_content_hash(splits, labels) != manifest["split_hash"]:
        raise ValueError(
            f"Split artifact integrity check failed: {split_path}; restore or regenerate preparation artifacts"
        )
    return PreparedData(Path(manifest["storage_root"]), manifest, labels, splits)


def prepare_data(config: dict[str, Any]) -> PreparedData:
    """Serialize preparation to protect immutable archive and split artifacts."""
    from filelock import FileLock

    root = resolve_storage_root(config.get("storage", {}))
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    effective = {**config, "storage": {**config.get("storage", {}), "root": str(root)}}
    with FileLock(data_dir / "prepare.lock", timeout=3600):
        return _prepare_data_unlocked(effective)


def _prepare_data_unlocked(config: dict[str, Any]) -> PreparedData:
    """Download original parquet once, audit duplicates and persist held-out indices."""
    settings = config.get("data", {})
    root = resolve_storage_root(config.get("storage", {}))
    configure_cache_environment(root)
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, split_path = data_dir / "manifest.json", data_dir / "splits.npz"
    repo = settings.get("repo_id", DEFAULT_REPO)
    requested_revision = settings.get("revision", DEFAULT_REVISION)
    if not isinstance(requested_revision, str) or not requested_revision:
        raise ValueError("A dataset revision must be provided")
    max_shards = settings.get("max_shards")
    if max_shards is not None and (not isinstance(max_shards, int) or max_shards < 1):
        raise ValueError("max_shards must be a positive integer and is reserved for smoke subsets")
    audit_hash = settings.get("audit_hash", "encoded")
    if audit_hash not in {"encoded", "pixels", "none"}:
        raise ValueError("audit_hash must be 'encoded', 'pixels' or 'none'")
    deduplicate = bool(settings.get("deduplicate", True))
    if audit_hash == "none" and deduplicate:
        raise ValueError("deduplicate requires audit_hash='encoded' or 'pixels'")
    access_mode = settings.get("access_mode", "packed")
    if access_mode not in {"packed", "parquet"}:
        raise ValueError("access_mode must be 'packed' or 'parquet'")
    identity = {
        "format_version": FORMAT_VERSION,
        "repo_id": repo,
        "requested_revision": requested_revision,
        "seed": int(settings.get("split_seed", 42)),
        "val_fraction": float(settings.get("val_fraction", 0.15)),
        "test_fraction": float(settings.get("test_fraction", 0.15)),
        "audit_hash": audit_hash,
        "deduplicate": deduplicate,
        "max_shards": max_shards,
        "access_mode": access_mode,
    }
    # Fully pinned preparations are reusable without network access.
    if re.fullmatch(r"[a-fA-F0-9]{40}", requested_revision) and manifest_path.exists():
        cached = _validate_cached(json.loads(manifest_path.read_text()), split_path, identity)
        if cached is not None:
            cached.manifest["cache_row_groups"] = int(settings.get("cache_row_groups", 2))
            return cached
    api = HfApi()
    info = api.dataset_info(repo, revision=requested_revision, files_metadata=True)
    revision = info.sha
    files = sorted(
        [s for s in info.siblings if s.rfilename.endswith(".parquet")],
        key=lambda item: item.rfilename,
    )
    if not files:
        raise ValueError(f"No original parquet files found in {repo}@{revision}")
    source_shards = len(files)
    if max_shards:
        files = files[:max_shards]
    sizes = {item.rfilename: item.size for item in files}
    if any(size is None for size in sizes.values()):
        raise ValueError(
            "Hub did not report shard sizes; cannot safely preflight storage requirements"
        )
    missing_bytes = 0
    for item in files:
        cached_path = try_to_load_from_cache(
            repo, item.rfilename, cache_dir=root / "hub", revision=revision, repo_type="dataset"
        )
        if (
            not isinstance(cached_path, str)
            or not Path(cached_path).is_file()
            or Path(cached_path).stat().st_size != sizes[item.rfilename]
        ):
            missing_bytes += sizes[item.rfilename]
    reserve_gb = float(config.get("storage", {}).get("min_free_gb", 10))
    # Filesystem reserve accommodates checkpoints and temporary downloads.
    check_free_space(root, required_bytes=missing_bytes, min_free_gb=reserve_gb)
    shards, all_labels, fingerprints = [], [], []
    row_offset = 0
    for number, item in enumerate(files, 1):
        print(f"Preparing shard {number}/{len(files)}: {item.rfilename}", flush=True)
        local = Path(
            hf_hub_download(
                repo, item.rfilename, repo_type="dataset", revision=revision, cache_dir=root / "hub"
            )
        )
        if local.stat().st_size != sizes[item.rfilename]:
            raise ValueError(f"Shard size mismatch: {local}")
        parquet = pq.ParquetFile(local, memory_map=True)
        labels = np.asarray(
            parquet.read(columns=["label"]).column("label").to_numpy(), dtype=np.int64
        )
        all_labels.append(labels)
        group_offsets = [0]
        uncompressed_bytes = 0
        for group in range(parquet.num_row_groups):
            group_offsets.append(group_offsets[-1] + parquet.metadata.row_group(group).num_rows)
            uncompressed_bytes += parquet.metadata.row_group(group).total_byte_size
        lfs = getattr(item, "lfs", None)
        source_sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        if isinstance(lfs, dict):
            source_sha256 = lfs.get("sha256")
        shards.append(
            {
                "filename": item.rfilename,
                "local_path": str(local),
                "size_bytes": local.stat().st_size,
                "source_sha256": source_sha256,
                "num_rows": len(labels),
                "row_offset": row_offset,
                "row_group_offsets": group_offsets,
                "uncompressed_parquet_bytes": uncompressed_bytes,
            }
        )
        row_offset += len(labels)
    labels = np.concatenate(all_labels)
    class_names = _class_names(info.card_data)
    if labels.min() < 0 or labels.max() >= len(class_names):
        raise ValueError("Parquet label values do not match the dataset class metadata")
    source_key = _canonical_hash(
        {"repo": repo, "revision": revision, "files": [shard["filename"] for shard in shards]}
    )[:20]
    packed_images = None
    archive_path = data_dir / f"images-{source_key}.bin"
    offsets_path = data_dir / f"images-{source_key}.offsets.npy"
    fingerprint_path = data_dir / f"images-{source_key}.{audit_hash}-sha256.npy"
    need_pack = access_mode == "packed" and not (archive_path.exists() and offsets_path.exists())
    if access_mode == "packed" and not need_pack:
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        if not _valid_packed_offsets(offsets, len(labels), archive_path.stat().st_size):
            raise ValueError(
                "Existing packed archive/index is incomplete; move those project-generated artifacts aside and rerun"
            )
    if need_pack:
        # Uncompressed parquet size includes image payload plus representation
        # overhead, giving a conservative bound after original downloads finish.
        upper_bound = (
            sum(shard["uncompressed_parquet_bytes"] for shard in shards)
            + (len(labels) + 1) * 8
            + 1024
        )
        check_free_space(root, required_bytes=upper_bound, min_free_gb=reserve_gb)
    need_audit = audit_hash != "none" and not fingerprint_path.exists()
    if audit_hash != "none" and not need_audit:
        cached_fingerprints = np.load(fingerprint_path, allow_pickle=False)
        if len(cached_fingerprints) != len(labels):
            raise ValueError(f"Image fingerprint cache row count mismatch: {fingerprint_path}")
        fingerprints = cached_fingerprints.tolist()
    if need_pack or need_audit:
        packed_temp = archive_path.with_suffix(".bin.partial")
        archive_offsets = [0]
        try:
            context = packed_temp.open("wb") if need_pack else nullcontext(None)
            with context as output:
                for number, shard in enumerate(shards, 1):
                    print(f"Auditing/packing images {number}/{len(shards)}", flush=True)
                    parquet = pq.ParquetFile(shard["local_path"], memory_map=True)
                    for group in range(parquet.num_row_groups):
                        images = parquet.read_row_group(group, columns=["image"]).column("image")
                        for value in images:
                            encoded = _image_bytes(value.as_py())
                            if need_pack:
                                output.write(encoded)
                                archive_offsets.append(archive_offsets[-1] + len(encoded))
                            if need_audit:
                                fingerprints.append(image_fingerprint(encoded, audit_hash))
            if need_pack:
                with tempfile.NamedTemporaryFile(
                    "wb", dir=data_dir, delete=False, suffix=".npy"
                ) as handle:
                    offsets_temp = Path(handle.name)
                    np.save(handle, np.asarray(archive_offsets, dtype=np.int64), allow_pickle=False)
                os.replace(packed_temp, archive_path)
                os.replace(offsets_temp, offsets_path)
            if need_audit:
                with tempfile.NamedTemporaryFile(
                    "wb", dir=data_dir, delete=False, suffix=".npy"
                ) as handle:
                    fingerprint_temp = Path(handle.name)
                    np.save(handle, np.asarray(fingerprints, dtype="U64"), allow_pickle=False)
                os.replace(fingerprint_temp, fingerprint_path)
        except BaseException:
            # Only this invocation's incomplete output is eligible for cleanup.
            if need_pack:
                packed_temp.unlink(missing_ok=True)
            raise
    if access_mode == "packed":
        packed_images = {
            "path": str(archive_path),
            "offsets_path": str(offsets_path),
            "size_bytes": archive_path.stat().st_size,
            "format": "original-encoded-image-bytes-v1",
            "row_order": "original-parquet-order",
            "reencoded": False,
        }
    groups = np.asarray(fingerprints) if audit_hash != "none" else None
    splits, audit = make_splits(
        labels,
        groups,
        seed=identity["seed"],
        val_fraction=identity["val_fraction"],
        test_fraction=identity["test_fraction"],
        deduplicate=deduplicate,
    )
    manifest = {
        **identity,
        "resolved_revision": revision,
        "storage_root": str(root),
        "class_names": class_names,
        "num_rows": len(labels),
        "shards": shards,
        "source_num_shards": source_shards,
        "subset_is_smoke_only": max_shards is not None,
        "audit": audit,
        "duplicate_audit_performed": audit_hash != "none",
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "split_class_counts": {
            name: np.bincount(labels[indices], minlength=len(class_names)).tolist()
            for name, indices in splits.items()
        },
        "split_hash": split_content_hash(splits, labels),
        "preparation_hash": _canonical_hash(identity),
        "cache_row_groups": int(settings.get("cache_row_groups", 2)),
        "packed_images": packed_images,
    }
    _save_splits(split_path, splits, labels)
    _atomic_json(manifest_path, manifest)
    return PreparedData(root, manifest, labels, splits)
