"""Mímir command-line interface.

This module exposes a small, well-tested CLI that wraps the indexing and search helpers
provided by the core package. The original prototype shipped with a very large, AI-generated
command file that was difficult to maintain and exceeded the needs of the current release.

The redesigned CLI keeps the surface area intentionally small:

* ``mimir index`` builds an index from a documentation directory.
* ``mimir search`` looks up queries using the TF-IDF engine.
* ``mimir ask`` provides a friendly answer stitched together from the top hits.
* ``mimir status`` reports metadata about the cached index.

All commands honour an optional ``--config`` flag that points at a YAML file so teams can
customise cache locations or disable expensive features (for example vector search).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import click
import yaml

from mimir.core.indexer import DocumentIndexer
from mimir.core.search import DocumentSearchEngine, SearchResult
from mimir.core.settings import get_project_root

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Return configuration data from ``config_path`` or the packaged default."""
    path = config_path or _DEFAULT_CONFIG
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _resolve_cache_dir(config: Dict[str, Any], project_root: Path) -> Path:
    """Resolve the cache directory defined in the configuration."""
    cache_dir = Path(config.get("cache_dir", ".cache/mimir"))
    return cache_dir if cache_dir.is_absolute() else project_root / cache_dir


def _ensure_docs_path(docs_path: str) -> Path:
    """Validate ``docs_path`` and return it as a ``Path`` instance."""
    path = Path(docs_path).expanduser()
    if not path.exists():
        raise click.BadParameter(f"Path '{docs_path}' does not exist")
    if not path.is_dir():
        raise click.BadParameter(f"Path '{docs_path}' must be a directory")
    return path


def _render_text_results(results: Iterable[SearchResult]) -> str:
    """Human readable rendering for search results."""
    lines: List[str] = []
    for idx, result in enumerate(results, start=1):
        lines.append(f"{idx}. {result.title} (score {result.relevance_score:.2f})")
        lines.append(f"   {result.document_path}")
        snippet = result.content_snippet.strip()
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines) if lines else "No results found."


def _render_json_results(results: Iterable[SearchResult]) -> str:
    """JSON rendering for search results."""
    payload = [result.to_dict() for result in results]
    return json.dumps(payload, indent=2)


def _render_paths(results: Iterable[SearchResult]) -> str:
    """Return newline separated document paths."""
    return "\n".join(result.document_path for result in results)


def _format_results(results: List[SearchResult], output: str) -> str:
    """Return formatted search output according to ``output`` format."""
    if output == "json":
        return _render_json_results(results)
    if output == "paths":
        return _render_paths(results)
    return _render_text_results(results)


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to a YAML configuration file (defaults to the packaged config).",
)
@click.option(
    "--project-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Override the project root used for cache and relative paths.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: Optional[Path], project_root: Optional[Path]) -> None:
    """Mímir – quick documentation search from the command line."""

    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["config"] = _load_config(config_path)
    ctx.obj["project_root"] = get_project_root(
        str(project_root) if project_root is not None else None
    )
    ctx.obj["index_dir"] = _resolve_cache_dir(ctx.obj["config"], ctx.obj["project_root"])


@cli.command()
@click.option(
    "--docs-path",
    "docs_path",
    default="docs",
    show_default=True,
    help="Directory containing source documentation (markdown, text, reStructuredText).",
)
@click.option("--force", is_flag=True, help="Force a rebuild even if cached data exists.")
@click.pass_context
def index(ctx: click.Context, docs_path: str, force: bool) -> None:
    """Build or refresh the on-disk search index."""

    docs_dir = _ensure_docs_path(docs_path)
    config_path = ctx.obj.get("config_path")
    indexer = DocumentIndexer(str(config_path) if config_path else None)

    ctx.obj["index_dir"].mkdir(parents=True, exist_ok=True)
    result = indexer.build_index([str(docs_dir)], force=force)

    click.echo(
        f"Indexed {result['metadata']['total_documents']} documents into {ctx.obj['index_dir']}"
    )


def _load_engine(ctx: click.Context, verbose: bool = False) -> DocumentSearchEngine:
    engine = DocumentSearchEngine(config=ctx.obj.get("config"), verbose=verbose)
    index_dir: Path = ctx.obj["index_dir"]
    if not engine.load_documents_from_index(index_dir):
        raise click.UsageError(
            f"No index found in {index_dir}. Run 'mimir index' before searching."
        )
    if not engine.build_search_index():
        raise click.ClickException("Unable to build search index from cached documents")
    return engine


@cli.command()
@click.argument("query")
@click.option("--limit", default=5, show_default=True, type=click.IntRange(1, 50))
@click.option(
    "--format",
    "output",
    type=click.Choice(["text", "json", "paths"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format for search results.",
)
@click.option("--verbose", is_flag=True, help="Show additional diagnostics.")
@click.pass_context
def search(ctx: click.Context, query: str, limit: int, output: str, verbose: bool) -> None:
    """Search the indexed documentation for ``QUERY``."""

    engine = _load_engine(ctx, verbose=verbose)
    results = engine.search(query, top_k=limit)
    click.echo(_format_results(results, output.lower()))


@cli.command()
@click.argument("question")
@click.option("--context-limit", default=3, show_default=True, type=click.IntRange(1, 10))
@click.pass_context
def ask(ctx: click.Context, question: str, context_limit: int) -> None:
    """Provide a lightweight answer by stitching together top search hits."""

    engine = _load_engine(ctx, verbose=False)
    results = engine.search(question, top_k=context_limit)
    if not results:
        click.echo("No relevant documents found.")
        return

    summary_lines = ["Suggested answer:"]
    for result in results:
        summary_lines.append(f"• {result.title} — {result.document_path}")
        snippet = result.content_snippet.strip()
        if snippet:
            summary_lines.append(f"  {snippet}")
    click.echo("\n".join(summary_lines))


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Print information about the cached index."""

    index_dir: Path = ctx.obj["index_dir"]
    index_file = index_dir / "index.yaml"
    if not index_file.exists():
        raise click.UsageError(
            f"No index metadata found in {index_dir}. Run 'mimir index' first."
        )

    with index_file.open("r", encoding="utf-8") as handle:
        index_data = yaml.safe_load(handle) or {}

    metadata = index_data.get("metadata", {})
    click.echo(f"Index location: {index_dir}")
    click.echo(f"Documents indexed: {metadata.get('total_documents', 0)}")
    click.echo(f"Last build duration: {metadata.get('build_time', 0):.2f}s")
    click.echo(f"Successful loads: {metadata.get('successful_loads', 0)}")
    click.echo(f"Failed loads: {metadata.get('failed_loads', 0)}")


__all__ = ["cli"]
