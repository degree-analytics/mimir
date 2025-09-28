"""Shared pytest fixtures for Mímir tests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import yaml


@pytest.fixture()
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated project root for tests."""
    monkeypatch.setenv("MIMIR_PROJECT_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture()
def sample_docs(project_root: Path) -> Path:
    """Create a small documentation tree for indexing tests."""
    docs_dir = project_root / "docs"
    docs_dir.mkdir()

    (docs_dir / "intro.md").write_text(
        """# Welcome\n\nMímir helps you search small documentation sets.\n""",
        encoding="utf-8",
    )

    (docs_dir / "setup.md").write_text(
        """# Setup\n\nInstall dependencies and run `mimir index`.\n""",
        encoding="utf-8",
    )

    return docs_dir


@pytest.fixture()
def minimal_config(project_root: Path, sample_docs: Path) -> Path:
    """Write a configuration file pointing to the temporary docs directory."""
    cache_dir = project_root / "cache"
    config: Dict[str, object] = {
        "docs_paths": [str(sample_docs)],
        "file_extensions": [".md"],
        "cache_dir": str(cache_dir),
        "encoding": "utf-8",
        "max_file_size": 1_048_576,
        "vector_search": {"enabled": False},
        "telemetry": {"enabled": False},
    }

    config_path = project_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
