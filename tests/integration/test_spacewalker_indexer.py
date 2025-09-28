"""Integration coverage for the document indexer using the Spacewalker snapshot."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter

import pytest

from mimir.core.error_handling import SecurityError
from mimir.core.indexer import DocumentIndexer

pytestmark = pytest.mark.spacewalker


def _expected_document_count(dataset_root: Path) -> int:
    """Count markdown/text files in the frozen dataset."""
    valid_suffixes = {".md", ".txt"}
    return sum(1 for path in dataset_root.rglob("*") if path.suffix.lower() in valid_suffixes)


def test_spacewalker_index_builds_snapshot(
    spacewalker_config_factory,
    spacewalker_dataset,
    spacewalker_metrics,
):
    config_path = spacewalker_config_factory()

    indexer = DocumentIndexer(str(config_path))

    start = perf_counter()
    result = indexer.build_index(force=True)
    elapsed = perf_counter() - start

    metadata = result["metadata"]
    expected_docs = _expected_document_count(spacewalker_dataset)

    assert metadata["total_documents"] == expected_docs
    assert metadata["errors"] == 0
    assert metadata["successful_loads"] == expected_docs
    assert metadata["build_time"] <= elapsed + 1  # Build time is measured internally

    # Spot-check critical documents are present with readable titles
    doc_by_path = {doc.path: doc for doc in indexer.documents}
    assert "docs/mobile/build-pipeline.md" in doc_by_path
    assert doc_by_path["docs/mobile/build-pipeline.md"].title.startswith("Mobile Build Pipeline")

    assert "docs/workflows/deployment-guide.md" in doc_by_path
    assert "Deployment Guide" in doc_by_path["docs/workflows/deployment-guide.md"].title

    spacewalker_metrics.record_index(
        "indexer-sync",
        {
            "total_documents": metadata["total_documents"],
            "build_time": round(metadata["build_time"], 3),
            "success_rate_percent": metadata["success_rate_percent"],
            "total_bytes_processed": metadata["total_bytes_processed"],
        },
    )

    # Second run without force should hit the cache and avoid reprocessing
    indexer_cached = DocumentIndexer(str(config_path))
    cached_result = indexer_cached.build_index(force=False)
    assert cached_result["metadata"]["total_documents"] == expected_docs
    cache_stats = indexer_cached.cache_manager.get_cache_stats()
    assert cache_stats["cache_sizes"]["index"]["files"] >= 1
    assert cache_stats["total_hits"] >= 1


def test_spacewalker_index_async(spacewalker_config_factory, spacewalker_dataset):
    config_path = spacewalker_config_factory()
    indexer = DocumentIndexer(str(config_path))

    result = asyncio.run(indexer.build_index_async(force=True))
    expected_docs = _expected_document_count(spacewalker_dataset)
    assert result["metadata"]["total_documents"] == expected_docs
    assert result["metadata"]["errors"] == 0


def test_spacewalker_index_security_guard(spacewalker_config_factory, spacewalker_project):
    config_path = spacewalker_config_factory()
    indexer = DocumentIndexer(str(config_path))

    outside_path = spacewalker_project.parent
    with pytest.raises(SecurityError):
        indexer.scan_documents([str(outside_path)])
