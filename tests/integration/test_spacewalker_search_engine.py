"""Search engine integration tests against the Spacewalker dataset."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from time import perf_counter
from typing import List

import pytest
import yaml

from mimir.core.cache import ThreadSafeCacheManager
from mimir.core.indexer import DocumentIndexer
from mimir.core.search import DocumentSearchEngine

pytestmark = pytest.mark.spacewalker


def _top_paths(results: List) -> List[str]:
    return [result.document_path for result in results]


def _cache_stats(cache_manager: ThreadSafeCacheManager) -> dict:
    return cache_manager.get_cache_stats()


def test_spacewalker_search_engine_relevance_and_cache(
    spacewalker_config_factory,
    spacewalker_metrics,
):
    config_path = spacewalker_config_factory()
    indexer = DocumentIndexer(str(config_path))
    indexer.build_index(force=True)

    engine = DocumentSearchEngine(config={"telemetry": {"enabled": False}}, verbose=False)
    assert engine.load_documents_from_index(indexer.index_path)
    assert engine.build_search_index()

    # Golden query: mobile EAS build pipeline
    start = perf_counter()
    results = engine.search("EAS build pipeline", top_k=5)
    elapsed = perf_counter() - start
    assert results, "Expected search hits for EAS build pipeline"
    assert any(path.endswith("docs/mobile/build-pipeline.md") for path in _top_paths(results))

    # Repeat the query to ensure the cache is exercised
    cached_results = engine.search("EAS build pipeline", top_k=5)
    assert cached_results
    stats = _cache_stats(engine.cache_manager)
    assert stats["total_hits"] >= 1

    spacewalker_metrics.record_search(
        "search-engine-eas-build",
        elapsed_seconds=elapsed,
        hit_paths=_top_paths(results)[:3],
        cache_stats=stats,
    )

    # Additional golden query covering deployment workflows
    deploy_results = engine.search("post-deploy e2e testing", top_k=5)
    assert deploy_results
    assert any(path.endswith("docs/INDEX.md") for path in _top_paths(deploy_results))


def test_spacewalker_search_engine_telemetry_and_llm(
    spacewalker_config_factory,
    spacewalker_metrics,
    fake_llm_client,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = spacewalker_config_factory(
        {
            "telemetry": {
                "enabled": True,
                "store_queries": True,
                "database_path": None,  # placeholder updated below
            },
            "llm": {
                "enabled": True,
                "api_key": "unit-test-key",
                "model": "meta-llama/test",
                "provider": "openrouter",
                "max_tokens": 256,
                "temperature": 0.0,
            },
        }
    )

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)

    cache_dir = Path(config_data["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    telemetry_db = cache_dir / "telemetry.db"
    config_data["telemetry"]["database_path"] = str(telemetry_db)

    # Rewrite config with concrete telemetry path so the search engine picks it up
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    indexer = DocumentIndexer(str(config_path))
    indexer.build_index(force=True)

    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")

    engine = DocumentSearchEngine(config=config_data, verbose=False)
    assert engine.load_documents_from_index(indexer.index_path)
    assert engine.build_search_index()

    # Replace the real client with the deterministic fake to avoid network calls
    engine.llm_enhancer.client = fake_llm_client
    fake_llm_client.enhanced_query = "post deploy e2e testing workflow"
    engine.embedding_generator = None

    enhanced_results = engine.llm_enhanced_search("post-deploy e2e testing", top_k=3)
    assert enhanced_results
    assert any("llm_analysis" in result.metadata for result in enhanced_results)
    assert fake_llm_client.recorded_calls

    answer = engine.ask_question("How do we reset demo data after deployment?", context_limit=3)
    assert answer == fake_llm_client.answer_text

    if engine.telemetry_manager:
        engine.telemetry_manager.wait_for_processing()

    # Inspect telemetry output to confirm the calls were persisted
    with sqlite3.connect(str(telemetry_db)) as conn:
        query_count = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
        stage_count = conn.execute("SELECT COUNT(*) FROM pipeline_stages").fetchone()[0]

    assert query_count >= 1
    spacewalker_metrics.record_telemetry(
        "llm-enabled",
        {
            "query_logs": query_count,
            "pipeline_stages": stage_count,
            "llm_calls": len(fake_llm_client.recorded_calls),
        },
    )
