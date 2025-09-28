"""Shared pytest fixtures for Mímir tests."""

from __future__ import annotations

import json
import os
import tarfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest
import yaml


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``target``."""
    for key, value in updates.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


@dataclass
class SpacewalkerMetricsRecorder:
    """Collect metrics emitted by the Spacewalker integration tests."""

    output_path: Path
    index_runs: List[Dict[str, Any]] = field(default_factory=list)
    search_runs: List[Dict[str, Any]] = field(default_factory=list)
    telemetry_runs: List[Dict[str, Any]] = field(default_factory=list)

    def record_index(self, label: str, metadata: Dict[str, Any]) -> None:
        self.index_runs.append({"label": label, "metadata": metadata})

    def record_search(
        self,
        label: str,
        elapsed_seconds: float,
        hit_paths: List[str],
        cache_stats: Dict[str, Any] | None = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "label": label,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "hit_paths": hit_paths,
        }
        if cache_stats:
            payload["cache"] = cache_stats
        self.search_runs.append(payload)

    def record_telemetry(self, label: str, summary: Dict[str, Any]) -> None:
        self.telemetry_runs.append({"label": label, "summary": summary})

    def flush(self) -> None:
        if not (self.index_runs or self.search_runs or self.telemetry_runs):
            return
        output = {
            "index_runs": self.index_runs,
            "search_runs": self.search_runs,
            "telemetry_runs": self.telemetry_runs,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


@dataclass
class FakeLLMClient:
    """Deterministic stand-in for the OpenRouter client used in tests."""

    available: bool = True
    summary_text: str = "Test summary"
    analysis_text: str = "Test analysis"
    answer_text: str = "Test answer"
    enhanced_query: str | None = None
    recorded_calls: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=lambda: {"total_requests": 0})

    def is_available(self) -> bool:
        return self.available

    def summarize_document(self, title: str, content: str, max_length: int = 150) -> str:
        self.recorded_calls.append(
            {
                "method": "summarize_document",
                "title": title,
                "max_length": max_length,
            }
        )
        self.stats["total_requests"] = self.stats.get("total_requests", 0) + 1
        return self.summary_text

    def analyze_document_content(self, title: str, content: str) -> str:
        self.recorded_calls.append(
            {
                "method": "analyze_document_content",
                "title": title,
                "content_length": len(content),
            }
        )
        self.stats["total_requests"] = self.stats.get("total_requests", 0) + 1
        return self.analysis_text

    def answer_question_with_context(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        self.recorded_calls.append(
            {
                "method": "answer_question_with_context",
                "question": question,
                "context_count": len(context_docs),
            }
        )
        self.stats["total_requests"] = self.stats.get("total_requests", 0) + 1
        return self.answer_text

    def enhance_search_query(self, original_query: str) -> str:
        self.recorded_calls.append(
            {"method": "enhance_search_query", "query": original_query}
        )
        self.stats["total_requests"] = self.stats.get("total_requests", 0) + 1
        return self.enhanced_query or original_query

    def get_stats(self) -> Dict[str, Any]:
        return dict(self.stats)


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
        "llm": {"enabled": False},
    }

    config_path = project_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


@pytest.fixture(scope="session")
def spacewalker_archive_path() -> Path:
    """Location of the frozen Spacewalker docs snapshot."""
    archive = Path(__file__).resolve().parent / "data" / "spacewalker_docs.tar.gz"
    if not archive.exists():
        pytest.fail("Spacewalker docs archive missing. Run scripts/freeze_spacewalker_docs.py")
    return archive


@pytest.fixture(scope="session")
def spacewalker_dataset(spacewalker_archive_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Extract the Spacewalker docs snapshot into a temporary directory."""
    extract_dir = tmp_path_factory.mktemp("spacewalker_dataset")
    with tarfile.open(spacewalker_archive_path, "r:gz") as archive:
        archive.extractall(path=extract_dir)

    docs_dir = extract_dir / "docs"
    if not docs_dir.exists():
        subdirs = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(subdirs) != 1:
            pytest.fail("Unexpected layout in Spacewalker archive")
        docs_dir = subdirs[0]
    return docs_dir


@pytest.fixture()
def spacewalker_project(tmp_path: Path, spacewalker_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Materialise the Spacewalker docs inside an isolated project root."""
    docs_target = tmp_path / "docs"
    shutil.copytree(spacewalker_dataset, docs_target)
    monkeypatch.setenv("MIMIR_PROJECT_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture()
def spacewalker_docs(spacewalker_project: Path) -> Path:
    """Return the docs directory for the materialised Spacewalker project."""
    return spacewalker_project / "docs"


@pytest.fixture()
def spacewalker_config_factory(spacewalker_project: Path) -> Callable[[Dict[str, Any]], Path]:
    """Factory fixture to create configs targeting the Spacewalker dataset."""

    def _factory(overrides: Dict[str, Any] | None = None) -> Path:
        base: Dict[str, Any] = {
            "docs_paths": [str(spacewalker_project / "docs")],
            "file_extensions": [".md", ".txt"],
            "cache_dir": str(spacewalker_project / ".cache"),
            "encoding": "utf-8",
            "max_file_size": 15_000_000,
            "vector_search": {"enabled": False},
            "telemetry": {"enabled": False},
            "llm": {"enabled": False},
        }

        if overrides:
            _deep_update(base, overrides)

        config_path = spacewalker_project / "spacewalker-config.yaml"
        config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
        return config_path

    return _factory


@pytest.fixture(scope="session")
def spacewalker_metrics() -> SpacewalkerMetricsRecorder:
    """Session-scoped metrics recorder used by integration tests."""

    metrics_path_env = os.getenv("MIMIR_SPACEWALKER_METRICS")
    if metrics_path_env:
        output_path = Path(metrics_path_env).expanduser().resolve()
    else:
        output_path = Path.cwd() / "spacewalker-metrics.json"

    recorder = SpacewalkerMetricsRecorder(output_path)
    yield recorder
    recorder.flush()


@pytest.fixture()
def fake_llm_client() -> FakeLLMClient:
    """Provide a reusable fake LLM client for deterministic tests."""

    return FakeLLMClient()
