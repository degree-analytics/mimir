"""Freeze the current Spacewalker documentation tree into a reproducible archive.

Usage::

    python scripts/freeze_spacewalker_docs.py [--source ../spacewalker/docs \
                                              --destination tests/data/spacewalker_docs.tar.gz]

The script normalises metadata (mtime/uid/gid) to keep diffs stable and emits the
resulting archive's size and SHA256 so maintainers can track changes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
from pathlib import Path
from typing import Iterable


def iter_files(root: Path) -> Iterable[Path]:
    """Yield all files under ``root`` to help detect unexpected changes."""
    for path in sorted(root.rglob("*")):
        yield path


def normalise_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    """Strip non-deterministic metadata so archives compare cleanly."""
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.mtime = 0
    # Permissions: keep executable bit off for regular files
    if tarinfo.isfile():
        tarinfo.mode = 0o644
    elif tarinfo.isdir():
        tarinfo.mode = 0o755
    return tarinfo


def build_archive(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source must be a directory: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    archive_root = source.name if source.name != "" else "spacewalker_docs"

    with tarfile.open(destination, "w:gz") as archive:
        archive.add(source, arcname=archive_root, filter=normalise_tarinfo)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root.parent / "spacewalker" / "docs",
        help="Path to the Spacewalker docs directory to snapshot.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=repo_root / "tests" / "data" / "spacewalker_docs.tar.gz",
        help="Where to write the frozen archive.",
    )
    args = parser.parse_args()

    build_archive(args.source.resolve(), args.destination.resolve())

    size_bytes = args.destination.stat().st_size
    digest = sha256sum(args.destination)

    print(f"Archive written: {args.destination}")
    print(f"  Size: {size_bytes / (1024 * 1024):.2f} MiB")
    print(f"  SHA256: {digest}")

    # Provide a quick listing so maintainers can eyeball the latest changes
    rel_root = os.path.relpath(args.source, repo_root)
    print(f"Captured files under: {rel_root}")
    for path in iter_files(args.source):
        if path.is_file():
            rel_path = os.path.relpath(path, args.source)
            print(f"  - {rel_path}")


if __name__ == "__main__":
    main()
