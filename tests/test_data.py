import io
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from firerisk.data import (
    PackedImageDataset,
    ParquetImageDataset,
    PreparedData,
    image_fingerprint,
    make_splits,
    prepare_data,
    split_content_hash,
)


def image_bytes(color=(0, 0, 0), **options):
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(stream, format="PNG", **options)
    return stream.getvalue()


def test_split_reproducibility_stratification_and_holdout_disjointness():
    labels = np.repeat(np.arange(7), 100)
    a, audit = make_splits(labels, seed=19, val_fraction=0.15, test_fraction=0.15)
    b, _ = make_splits(labels, seed=19, val_fraction=0.15, test_fraction=0.15)
    c, _ = make_splits(labels, seed=20, val_fraction=0.15, test_fraction=0.15)
    assert split_content_hash(a, labels) == split_content_hash(b, labels)
    assert split_content_hash(a, labels) != split_content_hash(c, labels)
    assert sorted(np.concatenate(list(a.values())).tolist()) == list(range(len(labels)))
    assert np.bincount(labels[a["validation"]]).tolist() == [15] * 7
    assert np.bincount(labels[a["test"]]).tolist() == [15] * 7
    assert audit["cross_split_group_overlap"] == 0


@pytest.mark.parametrize("deduplicate", [False, True])
def test_exact_duplicate_groups_cannot_cross_splits(deduplicate):
    labels = np.repeat(np.arange(2), 40)
    groups = np.array([f"image-{i // 2}" for i in range(80)])
    splits, audit = make_splits(labels, groups, seed=42, deduplicate=deduplicate)
    assigned = [{groups[i] for i in indices} for indices in splits.values()]
    assert not (assigned[0] & assigned[1] or assigned[0] & assigned[2] or assigned[1] & assigned[2])
    assert sum(map(len, splits.values())) == (40 if deduplicate else 80)
    assert audit["duplicate_rows"] == 40
    assert audit["removed_duplicate_rows"] == (40 if deduplicate else 0)


def test_conflicting_duplicate_annotations_require_explicit_resolution():
    with pytest.raises(ValueError, match="conflicting labels"):
        make_splits(np.array([0, 1, 0, 1, 0, 1]), np.array(["same", "same", "a", "b", "c", "d"]))


def test_split_refuses_unrepresentable_class_holdout():
    with pytest.raises(ValueError, match="at least three"):
        make_splits(np.array([0, 0, 1, 1, 1]))
    with pytest.raises(ValueError, match="sum to less than one"):
        make_splits(np.repeat(np.arange(2), 10), val_fraction=0.6, test_fraction=0.5)


def test_pixel_audit_catches_different_encodings_of_same_image():
    first = image_bytes((12, 45, 78), compress_level=0)
    second = image_bytes((12, 45, 78), compress_level=9)
    assert first != second
    assert image_fingerprint(first, "encoded") != image_fingerprint(second, "encoded")
    assert image_fingerprint(first, "pixels") == image_fingerprint(second, "pixels")


def parquet_fixture(tmp_path):
    shards = []
    offset = 0
    for shard_number in range(2):
        path = tmp_path / f"shard-{shard_number}.parquet"
        images = [{"bytes": image_bytes((i + offset, 0, 0)), "path": None} for i in range(6)]
        labels = [shard_number] * 6
        table = pa.table({"image": images, "label": labels})
        pq.write_table(table, path, row_group_size=2)
        shards.append(
            {
                "local_path": str(path),
                "row_offset": offset,
                "row_group_offsets": [0, 2, 4, 6],
                "num_rows": 6,
            }
        )
        offset += 6
    return {"shards": shards, "num_rows": offset, "class_names": ["a", "b"]}


def test_parquet_reader_correctly_maps_global_rows_and_bounds_cache(tmp_path):
    manifest = parquet_fixture(tmp_path)
    dataset = ParquetImageDataset(manifest, np.array([11, 0, 7, 4, 3]), cache_row_groups=2)
    for position, expected in enumerate([11, 0, 7, 4, 3]):
        sample = dataset[position]
        assert sample["index"] == expected
        assert sample["labels"] == expected // 6
        assert sample["pixel_values"].getpixel((0, 0)) == (expected, 0, 0)
        assert len(dataset._cache) <= 2
        assert len(dataset._files) <= 2
    # Spawned workers should reopen readers, never serialize native parquet handles.
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored._cache and not restored._files
    assert restored[0]["index"] == 11


def test_prepared_data_uses_only_training_counts_and_validation_alias(tmp_path):
    manifest = parquet_fixture(tmp_path)
    bundle = PreparedData(
        tmp_path,
        manifest,
        np.array([0] * 6 + [1] * 6),
        {
            "train": np.array([0, 1, 2, 6]),
            "validation": np.array([3, 7]),
            "test": np.array([4, 5, 8, 9, 10, 11]),
        },
    )
    assert bundle.train_class_counts.tolist() == [3, 1]
    assert len(bundle.build_dataset("val")) == 2
    with pytest.raises(ValueError, match="Unknown split"):
        bundle.build_dataset("everything")


def test_prepare_streams_packed_archive_and_reuses_pinned_artifacts_offline(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    source = tmp_path / "originals"
    source.mkdir()
    original_manifest = parquet_fixture(source)
    files = {
        f"data/train-{i}.parquet": shard["local_path"]
        for i, shard in enumerate(original_manifest["shards"])
    }
    revision = "1" * 40
    info = SimpleNamespace(
        sha=revision,
        siblings=[
            SimpleNamespace(
                rfilename=name, size=__import__("pathlib").Path(path).stat().st_size, lfs=None
            )
            for name, path in files.items()
        ],
        card_data={
            "dataset_info": {
                "features": [{"name": "label", "dtype": {"class_label": {"names": ["a", "b"]}}}]
            },
        },
    )
    calls = {"api": 0, "download": 0}

    class FakeAPI:
        def dataset_info(self, *args, **kwargs):
            calls["api"] += 1
            return info

    def download(repo, filename, **kwargs):
        calls["download"] += 1
        assert kwargs["revision"] == revision
        return files[filename]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeAPI)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *args, **kwargs: None)
    for variable in [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HF_XET_CACHE",
        "HF_ASSETS_CACHE",
        "TORCH_HOME",
        "HF_XET_CHUNK_CACHE_SIZE_BYTES",
    ]:
        monkeypatch.delenv(variable, raising=False)
    config = {
        "storage": {"root": str(tmp_path / "artifacts"), "min_free_gb": 0},
        "data": {
            "repo_id": "test/fire",
            "revision": revision,
            "access_mode": "packed",
            "audit_hash": "encoded",
            "deduplicate": True,
        },
    }
    prepared = prepare_data(config)
    assert prepared.manifest["resolved_revision"] == revision
    assert prepared.manifest["packed_images"]["reencoded"] is False
    assert prepared.manifest["subset_is_smoke_only"] is False
    dataset = prepared.build_dataset("train")
    assert isinstance(dataset, PackedImageDataset)
    for position, expected in enumerate(prepared.splits["train"]):
        sample = dataset[position]
        assert sample["pixel_values"].getpixel((0, 0)) == (int(expected), 0, 0)
        assert sample["labels"] == int(expected) // 6
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._packed is None and restored._offsets is None
    assert restored[0]["index"] == dataset[0]["index"]
    first_calls = calls.copy()
    reused = prepare_data(config)
    assert calls == first_calls
    assert reused.manifest["split_hash"] == prepared.manifest["split_hash"]
    # Save original parquet files even when packing; cleanup never deletes them.
    assert all(__import__("pathlib").Path(path).exists() for path in files.values())
