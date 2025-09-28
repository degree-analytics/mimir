"""Tests for the document indexer."""

from __future__ import annotations

from pathlib import Path

from mimir.core.indexer import DocumentIndexer


def test_build_index_creates_metadata(sample_docs: Path, minimal_config: Path) -> None:
    indexer = DocumentIndexer(str(minimal_config))
    result = indexer.build_index([str(sample_docs)], force=True)

    assert result["metadata"]["total_documents"] == 2
    index_path = indexer.index_path
    assert index_path is not None
    assert (index_path / "index.yaml").exists()
