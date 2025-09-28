"""Runtime settings helpers for Mímir."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PROJECT_ROOT = Path(os.environ.get("MIMIR_PROJECT_ROOT", Path.cwd()))
_DEFAULT_CACHE_DIR = Path(os.environ.get("MIMIR_CACHE_DIR", ".cache/mimir"))


def get_project_root(override: Optional[str] = None) -> Path:
    """Return the project root used for indexing and caching."""
    if override:
        return Path(override).expanduser().resolve()
    env_root = os.environ.get("MIMIR_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if _DEFAULT_PROJECT_ROOT:
        return Path(_DEFAULT_PROJECT_ROOT).expanduser().resolve()
    # Fallback to package root when installed globally
    return _PACKAGE_ROOT


def get_cache_dir() -> Path:
    """Return the directory for cache storage."""
    cache = os.environ.get("MIMIR_CACHE_DIR")
    base = get_project_root()
    cache_path = Path(cache).expanduser() if cache else _DEFAULT_CACHE_DIR
    if cache_path.is_absolute():
        return cache_path
    return base / cache_path
