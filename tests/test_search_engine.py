"""Tests for the search engine."""

from __future__ import annotations

from pathlib import Path

from mimir.core.indexer import DocumentIndexer
from mimir.core.search import DocumentSearchEngine


def test_search_returns_relevant_document(sample_docs: Path, minimal_config: Path) -> None:
    indexer = DocumentIndexer(str(minimal_config))
    indexer.build_index([str(sample_docs)], force=True)

    engine = DocumentSearchEngine(config={"vector_search": {"enabled": False}}, verbose=False)
    assert engine.load_documents_from_index(indexer.index_path)
    assert engine.build_search_index()

    results = engine.search("install dependencies", top_k=3)
    assert results, "Expected at least one search hit"
    top_hit = results[0]
    assert top_hit.document_path.endswith("setup.md")
