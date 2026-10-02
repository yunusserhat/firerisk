from pathlib import Path
from unittest.mock import patch

import pytest

from firerisk.storage import (
    GIB,
    check_free_space,
    configure_cache_environment,
    resolve_storage_root,
)


def test_explicit_storage_root_wins_over_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRERISK_HOME", str(tmp_path / "from_env"))
    root = resolve_storage_root({"root": str(tmp_path / "chosen")})
    assert root == tmp_path / "chosen"
    assert root.is_dir()


def test_default_root_uses_home_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("FIRERISK_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = resolve_storage_root()
    assert root == tmp_path / ".cache" / "firerisk"
    assert root.is_dir()


def test_default_root_respects_xdg_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("FIRERISK_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert resolve_storage_root() == tmp_path / "cache" / "firerisk"


def test_relative_xdg_cache_uses_home_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("FIRERISK_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "relative-cache")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert resolve_storage_root() == tmp_path / ".cache" / "firerisk"


def test_unwritable_explicit_root_raises_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRERISK_HOME", str(tmp_path / "chosen"))
    with patch("firerisk.storage.os.access", return_value=False):
        with pytest.raises(PermissionError, match="Storage root is not writable"):
            resolve_storage_root()


def test_environment_root_places_all_caches_on_same_volume(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRERISK_HOME", str(tmp_path))
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
    root = resolve_storage_root()
    configure_cache_environment(root)
    import os

    assert Path(os.environ["HF_HUB_CACHE"]) == root / "hub"
    assert Path(os.environ["HF_DATASETS_CACHE"]) == root / "datasets"
    assert Path(os.environ["HF_XET_CACHE"]) == root / "xet"
    assert os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] == "0"


def test_storage_preflight_reserves_space_without_removing_existing_files(tmp_path):
    sentinel = tmp_path / "existing-checkpoint.pt"
    sentinel.write_text("keep")
    with patch(
        "firerisk.storage.shutil.disk_usage", return_value=(20 * GIB, 15 * GIB, 5 * GIB)
    ) as usage:
        # disk_usage is a namedtuple in production.
        from collections import namedtuple

        usage.return_value = namedtuple("usage", "total used free")(20 * GIB, 15 * GIB, 5 * GIB)
        with pytest.raises(OSError, match="Insufficient storage"):
            check_free_space(tmp_path, required_bytes=4 * GIB, min_free_gb=2)
        assert (
            check_free_space(tmp_path, required_bytes=2 * GIB, min_free_gb=2)["free_bytes"]
            == 5 * GIB
        )
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("required,reserve", [(-1, 1), (1, -1)])
def test_storage_rejects_negative_requirements(tmp_path, required, reserve):
    with pytest.raises(ValueError):
        check_free_space(tmp_path, required, reserve)
