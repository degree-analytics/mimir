"""Real-world CLI coverage against the Spacewalker documentation corpus."""

from __future__ import annotations

import json
from time import perf_counter

import pytest
from click.testing import CliRunner

from mimir.cli.main import cli

pytestmark = pytest.mark.spacewalker


def test_spacewalker_cli_end_to_end(spacewalker_project, spacewalker_config_factory, spacewalker_metrics):
    config_path = spacewalker_config_factory()
    docs_dir = spacewalker_project / "docs"

    runner = CliRunner()

    index_result = runner.invoke(
        cli,
        [
            "--config",
            str(config_path),
            "--project-root",
            str(spacewalker_project),
            "index",
            "--docs-path",
            str(docs_dir),
            "--force",
        ],
    )
    assert index_result.exit_code == 0, index_result.output
    assert "Indexed" in index_result.output

    # Baseline text search
    query = "post-deploy e2e testing"
    start = perf_counter()
    search_result = runner.invoke(
        cli,
        [
            "--config",
            str(config_path),
            "--project-root",
            str(spacewalker_project),
            "search",
            query,
            "--limit",
            "5",
        ],
    )
    elapsed = perf_counter() - start
    assert search_result.exit_code == 0, search_result.output
    assert "docs/INDEX.md" in search_result.output
    assert "Post-Deploy" in search_result.output

    # JSON formatting for golden query coverage
    json_result = runner.invoke(
        cli,
        [
            "--config",
            str(config_path),
            "--project-root",
            str(spacewalker_project),
            "search",
            "EAS build pipeline",
            "--limit",
            "3",
            "--format",
            "json",
        ],
    )
    assert json_result.exit_code == 0, json_result.output
    payload = json.loads(json_result.output)
    assert payload, "Expected JSON payload"
    assert any(
        item["document_path"].endswith("docs/mobile/build-pipeline.md") for item in payload
    )

    # Paths format for automation hooks
    paths_result = runner.invoke(
        cli,
        [
            "--config",
            str(config_path),
            "--project-root",
            str(spacewalker_project),
            "search",
            "CLAUDE rules",
            "--limit",
            "3",
            "--format",
            "paths",
        ],
    )
    assert paths_result.exit_code == 0, paths_result.output
    assert any("claude" in line.lower() for line in paths_result.output.splitlines())

    # Status reflects index metadata
    status_result = runner.invoke(
        cli,
        [
            "--config",
            str(config_path),
            "--project-root",
            str(spacewalker_project),
            "status",
        ],
    )
    assert status_result.exit_code == 0, status_result.output
    assert "Documents indexed:" in status_result.output
    assert "Successful loads:" in status_result.output

    spacewalker_metrics.record_search(
        "cli-search",
        elapsed_seconds=elapsed,
        hit_paths=[line.strip() for line in paths_result.output.splitlines() if line.startswith("docs/")],
        cache_stats=None,
    )


def test_spacewalker_cli_ask_reports_hits(spacewalker_project, spacewalker_config_factory):
    config_path = spacewalker_config_factory()
    runner = CliRunner()

    index_result = runner.invoke(
        cli,
        [
            "--config",
            str(config_path),
            "--project-root",
            str(spacewalker_project),
            "index",
            "--docs-path",
            str(spacewalker_project / "docs"),
            "--force",
        ],
    )
    assert index_result.exit_code == 0, index_result.output

    ask_result = runner.invoke(
        cli,
        [
            "--config",
            str(config_path),
            "--project-root",
            str(spacewalker_project),
            "ask",
            "How do we reset demo data after deploying?",
            "--context-limit",
            "3",
        ],
    )
    assert ask_result.exit_code == 0, ask_result.output
    assert "Suggested answer" in ask_result.output
    assert "docs/workflows/demo-data-management.md" in ask_result.output
