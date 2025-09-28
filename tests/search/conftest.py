from __future__ import annotations

import json
import os
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

try:
    from mimir.core.indexer import DocumentIndexer
    from mimir.core.search import DocumentSearchEngine
    from mimir.ai.llm_integration import OpenRouterClient
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    mimir_src = repo_root / "../mimir/src"
    if mimir_src.exists():
        sys.path.insert(0, str(mimir_src.resolve()))
        from mimir.core.indexer import DocumentIndexer
        from mimir.core.search import DocumentSearchEngine
        from mimir.ai.llm_integration import OpenRouterClient
    else:
        raise

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
CACHE_ROOT = Path(".cache/mimir")
SNAPSHOT_CACHE = CACHE_ROOT / "docs_snapshot"
SNAPSHOT_STAMP = SNAPSHOT_CACHE / ".snapshot_sha"
DEFAULT_CONFIG = Path(__file__).parent / "configs/default_minilm.yaml"


@dataclass
class SnapshotMeta:
    archive: str
    sha256: str
    doc_count: int
    commit: str
    created_at: str

    @classmethod
    def load(cls) -> "SnapshotMeta":
        meta_path = FIXTURES_DIR / "snapshot_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError("snapshot_meta.json missing. Run scripts/search_snapshot.py --update.")
        data = json.loads(meta_path.read_text())
        return cls(
            archive=data["archive"],
            sha256=data["sha256"],
            doc_count=data["doc_count"],
            commit=data["commit"],
            created_at=data["created_at"],
        )


@pytest.hookimpl
def pytest_addoption(parser):
    parser.addoption(
        "--embedding-config",
        action="store",
        default=str(DEFAULT_CONFIG),
        help="Path to YAML config used to initialise the docs-search engine.",
    )


@pytest.fixture(scope="session")
def embedding_config_path(pytestconfig) -> Path:
    path = Path(pytestconfig.getoption("--embedding-config")).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Embedding config not found: {path}")
    return path


@pytest.fixture(scope="session")
def snapshot_meta() -> SnapshotMeta:
    return SnapshotMeta.load()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def snapshot_archive(snapshot_meta: SnapshotMeta) -> Path:
    archive = FIXTURES_DIR / snapshot_meta.archive
    if not archive.exists():
        raise FileNotFoundError(
            f"Snapshot archive {archive} missing. Run scripts/search_snapshot.py --update."
        )
    actual_hash = _sha256(archive)
    if actual_hash != snapshot_meta.sha256:
        raise RuntimeError(
            f"Snapshot hash mismatch for {archive}. Expected {snapshot_meta.sha256}, got {actual_hash}."
        )
    return archive


def _extract_snapshot(archive: Path, snapshot_meta: SnapshotMeta) -> Path:
    SNAPSHOT_CACHE.mkdir(parents=True, exist_ok=True)
    if SNAPSHOT_STAMP.exists() and SNAPSHOT_STAMP.read_text().strip() == snapshot_meta.sha256:
        return SNAPSHOT_CACHE

    for item in SNAPSHOT_CACHE.iterdir():
        if item.is_dir():
            import shutil

            shutil.rmtree(item)
        else:
            item.unlink()

    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(SNAPSHOT_CACHE)

    SNAPSHOT_STAMP.write_text(snapshot_meta.sha256)
    return SNAPSHOT_CACHE


@pytest.fixture(scope="session")
def docs_snapshot_dir(snapshot_archive: Path, snapshot_meta: SnapshotMeta) -> Path:
    path = _extract_snapshot(snapshot_archive, snapshot_meta)
    if not path.exists():
        raise RuntimeError("Failed to extract documentation snapshot.")
    return path


@pytest.fixture(scope="session")
def embedding_config(embedding_config_path: Path, docs_snapshot_dir: Path) -> Dict[str, any]:
    config = yaml.safe_load(embedding_config_path.read_text()) or {}
    docs_paths = [str(docs_snapshot_dir)]
    config["docs_paths"] = docs_paths

    cache_dir = config.get("cache_dir", ".cache/mimir/index_default_minilm")
    cache_path = Path(cache_dir)
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path
    cache_path.mkdir(parents=True, exist_ok=True)
    config["cache_dir"] = str(cache_path)
    return config


@pytest.fixture(scope="session")
def document_index(embedding_config: Dict[str, any]) -> Path:
    indexer = DocumentIndexer()
    indexer.config.update(embedding_config)

    from mimir.ai.embeddings import DocumentEmbeddingGenerator  # type: ignore

    indexer.embedding_generator = DocumentEmbeddingGenerator(
        indexer.config, indexer.cache_manager
    )

    docs_paths = embedding_config.get("docs_paths", [])
    if not docs_paths:
        raise RuntimeError("docs_paths not configured for index build")

    start = time.perf_counter()
    indexer.build_index(docs_paths, force=False)
    duration = time.perf_counter() - start
    print(f"📚 Document index prepared in {duration:.2f}s ({len(indexer.documents)} docs)")
    return indexer.index_path


@pytest.fixture(scope="session")
def search_engine(embedding_config: Dict[str, any], document_index: Path):
    engine = DocumentSearchEngine(config=embedding_config, verbose=False)
    if not engine.load_documents_from_index(document_index):
        raise RuntimeError("Could not load documents from index")
    if not engine.build_search_index():
        raise RuntimeError("Failed to build TF-IDF search index")
    return engine


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content
        self.usage = {"total_tokens": 0}
        self.model = "stubbed"
        self.response_time = 0.01
        self.metadata = {}


def _load_conversation_answers() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    convo_path = DATA_DIR / "conversations.jsonl"
    if not convo_path.exists():
        return mapping
    for line in convo_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for turn in record.get("turns", []):
            question = turn.get("question", "").strip()
            answer = turn.get("answer", "").strip()
            if question and answer:
                mapping[question.lower()] = answer
    return mapping


@pytest.fixture(scope="session", autouse=True)
def stub_llm():
    answers = _load_conversation_answers()

    original_request = getattr(OpenRouterClient, "_make_request", None)
    original_is_available = getattr(OpenRouterClient, "is_available", None)

    def always_available(self):
        return True

    def fake_request(self, prompt: str, system_prompt: str | None = None):
        prompt_lower = prompt.lower()
        for question, answer in answers.items():
            if question in prompt_lower:
                return _DummyResponse(answer)
        return _DummyResponse("Documentation context unavailable.")

    OpenRouterClient._make_request = fake_request  # type: ignore[attr-defined]
    OpenRouterClient.is_available = always_available  # type: ignore[attr-defined]
    try:
        yield
    finally:
        if original_request is not None:
            OpenRouterClient._make_request = original_request  # type: ignore[attr-defined]
        if original_is_available is not None:
            OpenRouterClient.is_available = original_is_available  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def queries_dataset() -> List[Dict[str, any]]:
    path = DATA_DIR / "queries.jsonl"
    if not path.exists():
        raise FileNotFoundError("queries.jsonl missing in tests/search/data")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="session")
def conversations_dataset() -> List[Dict[str, any]]:
    path = DATA_DIR / "conversations.jsonl"
    if not path.exists():
        raise FileNotFoundError("conversations.jsonl missing in tests/search/data")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="session", autouse=True)
def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ".gitignore").write_text("*\n!.gitignore\n")
