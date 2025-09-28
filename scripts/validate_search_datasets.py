#!/usr/bin/env python3
"""Validate docs-search datasets reference files present in the snapshot archive."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests/search/fixtures"
DATA_DIR = Path(__file__).resolve().parent.parent / "tests/search/data"


def load_snapshot_members(archive: Path) -> set[str]:
    with tarfile.open(archive, "r:gz") as tf:
        return {member.name for member in tf.getmembers() if member.isfile()}


def validate_paths(paths: Iterable[str], available: set[str], errors: List[str], context: str) -> None:
    for path in paths:
        snapshot_path = f"docs/{path}" if not path.startswith("docs/") else path
        if snapshot_path not in available:
            errors.append(f"Missing path '{path}' referenced in {context}")


def validate_queries(available: set[str], errors: List[str]) -> None:
    queries_path = DATA_DIR / "queries.jsonl"
    if not queries_path.exists():
        errors.append("queries.jsonl not found")
        return
    for line_no, line in enumerate(queries_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        validate_paths(record.get("relevant_paths", []), available, errors, f"queries.jsonl line {line_no}")


def validate_conversations(available: set[str], errors: List[str]) -> None:
    convo_path = DATA_DIR / "conversations.jsonl"
    if not convo_path.exists():
        errors.append("conversations.jsonl not found")
        return
    for line_no, line in enumerate(convo_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        for turn in record.get("turns", []):
            validate_paths(turn.get("expected_paths", []), available, errors, f"conversations.jsonl line {line_no}")


def main() -> int:
    meta_path = FIXTURES_DIR / "snapshot_meta.json"
    if not meta_path.exists():
        print("❌ snapshot_meta.json missing. Generate snapshot first.")
        return 1
    meta = json.loads(meta_path.read_text())
    archive = FIXTURES_DIR / meta["archive"]
    if not archive.exists():
        print(f"❌ Snapshot archive not found: {archive}")
        return 1

    available = load_snapshot_members(archive)
    errors: List[str] = []
    validate_queries(available, errors)
    validate_conversations(available, errors)

    if errors:
        print("❌ Dataset validation failed:")
        for error in errors:
            print(f"   - {error}")
        return 1
    print("✅ Dataset paths validated against snapshot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
