"""Integration tests for the CLI entry points."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from mimir.cli.main import cli


def test_cli_index_search_and_status(sample_docs: Path, minimal_config: Path) -> None:
    runner = CliRunner()

    project_root = sample_docs.parent

    index_result = runner.invoke(
        cli,
        [
            "--config",
            str(minimal_config),
            "--project-root",
            str(project_root),
            "index",
            "--docs-path",
            str(sample_docs),
        ],
    )
    assert index_result.exit_code == 0, index_result.output
    assert "Indexed 2 documents" in index_result.output

    search_result = runner.invoke(
        cli,
        [
            "--config",
            str(minimal_config),
            "--project-root",
            str(project_root),
            "search",
            "Mímir",
            "--limit",
            "2",
        ],
    )
    assert search_result.exit_code == 0, search_result.output
    assert "Welcome" in search_result.output

    status_result = runner.invoke(
        cli,
        ["--config", str(minimal_config), "--project-root", str(project_root), "status"],
    )
    assert status_result.exit_code == 0, status_result.output
    assert "Documents indexed: 2" in status_result.output
