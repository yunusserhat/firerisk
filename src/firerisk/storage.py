"""Portable artifact placement and conservative disk-space checks."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

GIB = 1024**3


def resolve_storage_root(config: dict[str, Any] | None = None) -> Path:
    """Use a configured root, FIRERISK_HOME, or the user's cache directory."""
    config = config or {}
    explicit = config.get("root") or os.environ.get("FIRERISK_HOME")
    if explicit:
        root = Path(str(explicit)).expanduser().resolve()
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", "")).expanduser()
        # The XDG specification requires an absolute cache path.
        if not cache.is_absolute():
            cache = Path.home() / ".cache"
        root = (cache / "firerisk").resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK):
        raise PermissionError(f"Storage root is not writable: {root}")
    return root


def configure_cache_environment(root: Path) -> None:
    """Place HF downloads and torch artifacts in the selected storage volume."""
    root = Path(root).resolve()
    locations = {
        "HF_HOME": root / "hf",
        "HF_HUB_CACHE": root / "hub",
        "HF_DATASETS_CACHE": root / "datasets",
        "HF_XET_CACHE": root / "xet",
        "HF_ASSETS_CACHE": root / "assets",
        "TORCH_HOME": root / "torch",
    }
    for variable, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        # The project's explicit storage choice wins over a default home cache.
        os.environ[variable] = str(path)
    # Xet's chunk cache can otherwise duplicate the parquet download on disk.
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")


def inspect_storage(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    return {
        "root": str(root),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": round(usage.free / GIB, 2),
        "note": "Filesystem free space does not include user or project quota limits.",
    }


def check_free_space(
    root: Path, required_bytes: int = 0, min_free_gb: float = 10
) -> dict[str, Any]:
    """Check that downloads plus an operational reserve fit without deleting files."""
    if required_bytes < 0 or min_free_gb < 0:
        raise ValueError("required_bytes and min_free_gb must be non-negative")
    state = inspect_storage(root)
    reserve = int(min_free_gb * GIB)
    if state["free_bytes"] < required_bytes + reserve:
        raise OSError(
            f"Insufficient storage at {state['root']}: {state['free_gib']:.2f} GiB free, "
            f"{required_bytes / GIB:.2f} GiB download required and "
            f"{min_free_gb:.2f} GiB reserved. Set FIRERISK_HOME to a larger writable "
            "volume; existing files will not be deleted. Check your quota too."
        )
    return state
