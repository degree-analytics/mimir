"""
Document Indexer - Core indexing functionality for mimir tool

Handles document loading, parsing, and index building for efficient search.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import yaml
from dataclasses import dataclass, asdict
import time
import asyncio

from .settings import get_project_root
from .cache import ThreadSafeCacheManager
from ..ai.embeddings import DocumentEmbeddingGenerator
from .error_handling import (
    SecurityError,
    IndexingError,
    log_info,
    log_warning,
    log_error,
    log_success,
    log_progress,
    handle_error,
)


@dataclass
class Document:
    """Represents a parsed document with metadata."""

    path: str
    content: str
    checksum: str
    title: str
    size: int
    last_modified: float
    metadata: Dict[str, Any]


class DocumentIndexer:
    """Core document indexing functionality."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the indexer with configuration."""
        self.config = self._load_config(config_path)
        self.documents: List[Document] = []

        # Determine project root (overridable via env)
        self.project_root = get_project_root()

        # Resolve cache directory relative to project root unless absolute
        cache_dir = Path(self.config.get("cache_dir", ".cache/mimir"))
        if cache_dir.is_absolute():
            self.index_path = cache_dir
        else:
            self.index_path = self.project_root / cache_dir
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Initialize cache manager
        self.cache_manager = ThreadSafeCacheManager(self.index_path)

        # Initialize embedding generator
        self.embedding_generator = DocumentEmbeddingGenerator(
            self.config, self.cache_manager
        )

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = (
                Path(__file__).resolve().parent.parent / "config" / "default.yaml"
            )

        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "docs_paths": ["docs/"],
            "file_extensions": [".md", ".txt", ".rst"],
            "cache_dir": ".cache/mimir",
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "encoding": "utf-8",
        }

    def _validate_and_resolve_path(self, path: Path) -> Path:
        """Validate and resolve a path with reasonable security boundaries."""
        # Convert to absolute path if relative
        project_root = self.project_root

        if not path.is_absolute():
            path = project_root / path

        # Resolve to canonical path (resolves symlinks and .. references)
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Cannot resolve path: {e}")

        # Define reasonable project boundary (allow paths within project root)
        # For the Spacewalker project, docs are at the same level as scripts
        project_boundary = project_root.resolve()

        # Check if resolved path is within reasonable bounds
        try:
            resolved_path.relative_to(project_boundary)
            # Path is within acceptable project structure
        except ValueError:
            # Path goes outside reasonable project bounds
            raise SecurityError(
                f"Path '{resolved_path}' is outside reasonable project boundaries. "
                f"Paths must be within: {project_boundary}"
            )

        # Additional check: prevent access to sensitive system directories
        sensitive_paths = ["/etc", "/usr", "/var", "/bin", "/sbin", "/sys", "/proc"]
        path_str = str(resolved_path).lower()
        for sensitive in sensitive_paths:
            if path_str.startswith(sensitive.lower()):
                raise SecurityError(
                    f"Access to system directory '{sensitive}' is not allowed"
                )

        return resolved_path

    def scan_documents(self, docs_paths: Optional[List[str]] = None) -> List[Path]:
        """Scan for documents in specified paths with comprehensive error handling."""
        if docs_paths is None:
            docs_paths = self.config["docs_paths"]

        found_files = []
        extensions = self.config["file_extensions"]

        for docs_path in docs_paths:
            path = Path(docs_path)

            # Validate and resolve path securely
            try:
                path = self._validate_and_resolve_path(path)
            except SecurityError as e:
                if not handle_error(e, f"Path validation for '{docs_path}'"):
                    raise  # Re-raise critical security errors
                continue
            except ValueError as e:
                handle_error(
                    IndexingError(f"Invalid path '{docs_path}': {e}"), "Path resolution"
                )
                continue

            try:
                if not path.exists():
                    log_warning(f"Path does not exist: {path}")
                    continue

                if path.is_file() and path.suffix.lower() in extensions:
                    # Check file permissions
                    try:
                        path.stat()  # Test if we can access file stats
                        found_files.append(path)
                    except (OSError, PermissionError) as e:
                        handle_error(e, f"File access check for {path}")
                        continue

                elif path.is_dir():
                    # Recursively scan directory with error handling
                    try:
                        for ext in extensions:
                            # Use generator for memory efficiency
                            for file_path in path.rglob(f"*{ext}"):
                                try:
                                    # Verify file is readable
                                    if file_path.is_file():
                                        file_path.stat()  # Test if we can access file stats
                                        found_files.append(file_path)
                                except (OSError, PermissionError) as e:
                                    log_warning(f"Cannot access file {file_path}: {e}")
                                    continue
                    except (OSError, PermissionError) as e:
                        log_warning(f"Cannot access directory {path}: {e}")
                        continue

            except (OSError, PermissionError) as e:
                log_warning(f"Cannot access path {path}: {e}")
                continue

        return sorted(found_files)

    def _compute_checksum(self, content: str) -> str:
        """Compute SHA256 checksum of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _extract_front_matter(self, content: str) -> Dict[str, Any]:
        """Extract YAML front matter from document if present."""
        front_matter = {}

        # Check if content starts with front matter delimiter
        if content.startswith("---\n"):
            try:
                # Find the end of front matter
                end_delimiter = content.find("\n---\n", 4)
                if end_delimiter != -1:
                    front_matter_text = content[4:end_delimiter]
                    front_matter = yaml.safe_load(front_matter_text) or {}
            except yaml.YAMLError:
                pass  # Ignore invalid YAML

        return front_matter

    def _extract_title(self, content: str, file_path: Path) -> str:
        """Extract title from document content with multiple strategies."""
        # First, try front matter
        front_matter = self._extract_front_matter(content)
        if front_matter.get("title"):
            return str(front_matter["title"]).strip()

        lines = content.strip().split("\n")

        # Skip front matter if present
        start_line = 0
        if content.startswith("---\n"):
            end_delimiter = content.find("\n---\n", 4)
            if end_delimiter != -1:
                # Count lines in front matter
                front_matter_lines = content[: end_delimiter + 5].count("\n")
                start_line = front_matter_lines

        # Look for markdown headers
        for i in range(start_line, min(len(lines), start_line + 15)):
            line = lines[i].strip()
            if line.startswith("# "):
                return line[2:].strip()
            elif line.startswith("## ") and not lines[i - 1].strip() if i > 0 else True:
                return line[3:].strip()

        # Fall back to filename
        return file_path.stem.replace("-", " ").replace("_", " ").title()

    def load_document(self, file_path: Path) -> Optional[Document]:
        """Load and parse a single document with streaming support."""
        try:
            # Get file stats first
            stat = file_path.stat()

            # Check file size
            if stat.st_size > self.config["max_file_size"]:
                log_warning(
                    f"Skipping large file: {file_path} ({stat.st_size / 1024 / 1024:.1f}MB)"
                )
                return None

            # Try multiple encodings if the default fails
            encodings_to_try = [self.config["encoding"], "utf-8", "latin1", "cp1252"]
            content = None
            used_encoding = None

            for encoding in encodings_to_try:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        # For large files, read in chunks to avoid memory issues
                        if stat.st_size > 1024 * 1024:  # 1MB
                            chunks = []
                            while True:
                                chunk = f.read(8192)  # 8KB chunks
                                if not chunk:
                                    break
                                chunks.append(chunk)
                            content = "".join(chunks)
                        else:
                            content = f.read()

                    used_encoding = encoding
                    break

                except UnicodeDecodeError:
                    continue

            if content is None:
                log_error(f"Could not decode file {file_path} with any encoding")
                return None

            # Extract metadata
            checksum = self._compute_checksum(content)
            title = self._extract_title(content, file_path)

            # Extract front matter if present
            front_matter = self._extract_front_matter(content)

            return Document(
                path=str(file_path.relative_to(self.project_root)),
                content=content,
                checksum=checksum,
                title=title,
                size=stat.st_size,
                last_modified=stat.st_mtime,
                metadata={
                    "encoding": used_encoding,
                    "front_matter": front_matter,
                    "relative_path": str(file_path.relative_to(self.project_root)),
                    "absolute_path": str(file_path),
                },
            )

        except (OSError, PermissionError) as e:
            log_error(f"Error accessing {file_path}: {e}")
            return None
        except Exception as e:
            log_error(f"Unexpected error loading {file_path}: {e}")
            return None

    def build_index(
        self, docs_paths: Optional[List[str]] = None, force: bool = False
    ) -> Dict[str, Any]:
        """Build the document index with comprehensive real-time progress reporting and caching."""
        log_info(
            "Building document index...", "📚", operation="index_build", force=force
        )
        start_time = time.time()

        # Calculate content checksum for cache validation
        log_info("Scanning for documents...", "🔍", operation="document_scan")
        file_paths = self.scan_documents(docs_paths)
        content_checksum = self._calculate_content_checksum(file_paths)

        # Try to load from cache if not forcing rebuild
        if not force:
            cached_index = self.cache_manager.get_cached_index(content_checksum)
            if cached_index:
                log_success(
                    "Loaded index from cache",
                    "⚡",
                    documents_count=cached_index["metadata"]["total_documents"],
                    cache_hit=True,
                )
                self.documents = [Document(**doc) for doc in cached_index["documents"]]
                return cached_index

        # Clear existing documents
        self.documents = []

        # Initialize progress tracking
        total_files = len(file_paths)
        loaded_count = 0
        error_count = 0
        skipped_count = 0
        total_bytes_processed = 0

        log_info(
            f"Found {total_files} documents to process",
            "📄",
            total_files=total_files,
            paths=docs_paths or self.config["docs_paths"],
        )

        # Load documents with enhanced real-time progress reporting
        for i, file_path in enumerate(file_paths):
            current_file_num = i + 1

            # Real-time progress feedback with file counts and status
            log_progress(
                f"Processing: {file_path.name}",
                current=current_file_num,
                total=total_files,
                file_path=str(file_path),
                loaded_count=loaded_count,
                error_count=error_count,
                skipped_count=skipped_count,
                operation="document_processing",
            )

            # Load the document
            doc = self.load_document(file_path)
            if doc:
                self.documents.append(doc)
                loaded_count += 1
                total_bytes_processed += doc.size

                # Show success with document details
                log_success(
                    f"✓ Loaded: {doc.title}",
                    file_path=str(file_path),
                    document_size=doc.size,
                    document_title=doc.title,
                    total_loaded=loaded_count,
                    operation="document_loaded",
                )
            else:
                error_count += 1
                log_warning(
                    f"✗ Failed: {file_path.name}",
                    file_path=str(file_path),
                    total_errors=error_count,
                    operation="document_failed",
                )

            # Show intermediate progress every 10 files or at milestones
            if current_file_num % 10 == 0 or current_file_num in [1, 5, total_files]:
                progress_pct = (
                    (current_file_num / total_files * 100) if total_files > 0 else 0
                )
                log_info(
                    f"Progress: {current_file_num}/{total_files} files processed ({progress_pct:.1f}%)",
                    "📊",
                    progress_percent=progress_pct,
                    files_processed=current_file_num,
                    files_loaded=loaded_count,
                    files_failed=error_count,
                    bytes_processed=total_bytes_processed,
                    operation="progress_milestone",
                )

        elapsed = time.time() - start_time

        # Calculate comprehensive statistics
        total_size = sum(doc.size for doc in self.documents)
        avg_size = total_size / len(self.documents) if self.documents else 0
        success_rate = (loaded_count / total_files * 100) if total_files > 0 else 0
        processing_rate = total_files / elapsed if elapsed > 0 else 0

        # Build index data with enhanced metadata
        index_data = {
            "documents": [asdict(doc) for doc in self.documents],
            "metadata": {
                "total_documents": len(self.documents),
                "total_files_found": total_files,
                "successful_loads": loaded_count,
                "errors": error_count,
                "skipped": skipped_count,
                "success_rate_percent": round(success_rate, 1),
                "total_size_bytes": total_size,
                "average_size_bytes": round(avg_size, 0),
                "total_bytes_processed": total_bytes_processed,
                "build_time": elapsed,
                "processing_rate_files_per_sec": round(processing_rate, 2),
                "timestamp": time.time(),
                "docs_paths": docs_paths or self.config["docs_paths"],
                "content_checksum": content_checksum,
            },
        }

        # Save to file cache with error handling
        try:
            index_file = self.index_path / "index.yaml"
            with open(index_file, "w") as f:
                yaml.dump(index_data, f, default_flow_style=False)
            log_success("Index saved to file cache", "💾", file_path=str(index_file))
        except Exception as e:
            log_error(
                "Failed to save index to file", file_path=str(index_file), error=str(e)
            )

        # Save to memory cache
        try:
            self.cache_manager.cache_index(index_data, content_checksum)
            log_success("Index saved to memory cache", "⚡")
        except Exception as e:
            log_error("Failed to save index to memory cache", error=str(e))

        # Generate embeddings if enabled with progress tracking
        embeddings_generated = False
        if self.config.get("vector_search", {}).get("enabled", False):
            log_info(
                "Generating document embeddings...",
                "🧠",
                document_count=len(self.documents),
                operation="embeddings_generation",
            )
            try:
                embeddings = self.embedding_generator.generate_embeddings(
                    [asdict(doc) for doc in self.documents]
                )
                if embeddings:
                    index_data["embeddings_count"] = len(embeddings)
                    index_data["metadata"]["embeddings_generated"] = True
                    embeddings_generated = True
                    log_success(
                        f"Generated {len(embeddings)} embeddings",
                        "🧠",
                        embeddings_count=len(embeddings),
                    )
                else:
                    index_data["metadata"]["embeddings_generated"] = False
                    log_warning("No embeddings were generated", "🧠")
            except Exception as e:
                log_error("Failed to generate embeddings", error=str(e))
                index_data["metadata"]["embeddings_generated"] = False

        # Comprehensive final summary with structured context
        log_success(
            f"Index build complete: {loaded_count}/{total_files} documents processed in {elapsed:.2f}s",
            "🎉",
            operation="index_build_complete",
            total_files=total_files,
            loaded_count=loaded_count,
            error_count=error_count,
            success_rate=success_rate,
            processing_time=elapsed,
            processing_rate=processing_rate,
            total_size_mb=round(total_size / (1024 * 1024), 2),
            embeddings_generated=embeddings_generated,
        )

        if error_count > 0:
            suggestions = {
                "review": "Check file permissions and formats for failed documents",
                "command": "Use --verbose flag to see detailed error information",
                "retry": "Re-run indexing with --force flag if needed",
            }
            log_warning(
                f"{error_count} documents could not be processed",
                error_count=error_count,
                suggestions=suggestions,
            )

        return index_data

    async def build_index_async(
        self, docs_paths: Optional[List[str]] = None, force: bool = False
    ) -> Dict[str, Any]:
        """Build the document index asynchronously with comprehensive real-time progress reporting."""
        log_info(
            "Building document index (async)...",
            "📚",
            operation="async_index_build",
            force=force,
        )
        start_time = time.time()

        # Calculate content checksum for cache validation
        log_info(
            "Scanning for documents (async)...", "🔍", operation="async_document_scan"
        )
        file_paths = self.scan_documents(docs_paths)
        content_checksum = self._calculate_content_checksum(file_paths)

        # Try to load from cache if not forcing rebuild
        if not force:
            cached_index = self.cache_manager.get_cached_index(content_checksum)
            if cached_index:
                log_success(
                    "Loaded index from cache (async)",
                    "⚡",
                    documents_count=cached_index["metadata"]["total_documents"],
                    cache_hit=True,
                )
                self.documents = [Document(**doc) for doc in cached_index["documents"]]
                return cached_index

        # Clear existing documents
        self.documents = []

        # Initialize progress tracking
        total_files = len(file_paths)
        loaded_count = 0
        error_count = 0
        skipped_count = 0
        total_bytes_processed = 0

        log_info(
            f"Found {total_files} documents to process (async)",
            "📄",
            total_files=total_files,
            paths=docs_paths or self.config["docs_paths"],
        )

        # Load documents with enhanced async progress reporting
        for i, file_path in enumerate(file_paths):
            current_file_num = i + 1

            # Real-time progress feedback with file counts and status
            log_progress(
                f"Processing (async): {file_path.name}",
                current=current_file_num,
                total=total_files,
                file_path=str(file_path),
                loaded_count=loaded_count,
                error_count=error_count,
                skipped_count=skipped_count,
                operation="async_document_processing",
            )

            # Yield control to event loop for non-blocking operation
            await asyncio.sleep(0)

            # Load the document
            doc = self.load_document(file_path)
            if doc:
                self.documents.append(doc)
                loaded_count += 1
                total_bytes_processed += doc.size

                # Show success with document details
                log_success(
                    f"✓ Loaded (async): {doc.title}",
                    file_path=str(file_path),
                    document_size=doc.size,
                    document_title=doc.title,
                    total_loaded=loaded_count,
                    operation="async_document_loaded",
                )
            else:
                error_count += 1
                log_warning(
                    f"✗ Failed (async): {file_path.name}",
                    file_path=str(file_path),
                    total_errors=error_count,
                    operation="async_document_failed",
                )

            # Yield control periodically for responsiveness and show progress milestones
            if current_file_num % 5 == 0:
                await asyncio.sleep(0)

            # Show intermediate progress every 10 files or at milestones
            if current_file_num % 10 == 0 or current_file_num in [1, 5, total_files]:
                progress_pct = (
                    (current_file_num / total_files * 100) if total_files > 0 else 0
                )
                log_info(
                    f"Async Progress: {current_file_num}/{total_files} files processed ({progress_pct:.1f}%)",
                    "📊",
                    progress_percent=progress_pct,
                    files_processed=current_file_num,
                    files_loaded=loaded_count,
                    files_failed=error_count,
                    bytes_processed=total_bytes_processed,
                    operation="async_progress_milestone",
                )

        elapsed = time.time() - start_time

        # Calculate comprehensive statistics
        total_size = sum(doc.size for doc in self.documents)
        avg_size = total_size / len(self.documents) if self.documents else 0
        success_rate = (loaded_count / total_files * 100) if total_files > 0 else 0
        processing_rate = total_files / elapsed if elapsed > 0 else 0

        # Build index data with enhanced metadata
        index_data = {
            "documents": [asdict(doc) for doc in self.documents],
            "metadata": {
                "total_documents": len(self.documents),
                "total_files_found": total_files,
                "successful_loads": loaded_count,
                "errors": error_count,
                "skipped": skipped_count,
                "success_rate_percent": round(success_rate, 1),
                "total_size_bytes": total_size,
                "average_size_bytes": round(avg_size, 0),
                "total_bytes_processed": total_bytes_processed,
                "build_time": elapsed,
                "processing_rate_files_per_sec": round(processing_rate, 2),
                "timestamp": time.time(),
                "docs_paths": docs_paths or self.config["docs_paths"],
                "content_checksum": content_checksum,
                "async_build": True,
            },
        }

        # Save to file cache with error handling
        try:
            index_file = self.index_path / "index.yaml"
            with open(index_file, "w") as f:
                yaml.dump(index_data, f, default_flow_style=False)
            log_success(
                "Index saved to file cache (async)", "💾", file_path=str(index_file)
            )
        except Exception as e:
            log_error(
                "Failed to save index to file (async)",
                file_path=str(index_file),
                error=str(e),
            )

        # Save to memory cache
        try:
            self.cache_manager.cache_index(index_data, content_checksum)
            log_success("Index saved to memory cache (async)", "⚡")
        except Exception as e:
            log_error("Failed to save index to memory cache (async)", error=str(e))

        # Generate embeddings if enabled (async) with progress tracking
        embeddings_generated = False
        if self.config.get("vector_search", {}).get("enabled", False):
            log_info(
                "Generating document embeddings (async)...",
                "🧠",
                document_count=len(self.documents),
                operation="async_embeddings_generation",
            )
            await asyncio.sleep(0)  # Yield control
            try:
                embeddings = self.embedding_generator.generate_embeddings(
                    [asdict(doc) for doc in self.documents]
                )
                if embeddings:
                    index_data["embeddings_count"] = len(embeddings)
                    index_data["metadata"]["embeddings_generated"] = True
                    embeddings_generated = True
                    log_success(
                        f"Generated {len(embeddings)} embeddings (async)",
                        "🧠",
                        embeddings_count=len(embeddings),
                    )
                else:
                    index_data["metadata"]["embeddings_generated"] = False
                    log_warning("No embeddings were generated (async)", "🧠")
            except Exception as e:
                log_error("Failed to generate embeddings (async)", error=str(e))
                index_data["metadata"]["embeddings_generated"] = False

        # Comprehensive final summary with structured context
        log_success(
            f"Async index build complete: {loaded_count}/{total_files} documents processed in {elapsed:.2f}s",
            "🎉",
            operation="async_index_build_complete",
            total_files=total_files,
            loaded_count=loaded_count,
            error_count=error_count,
            success_rate=success_rate,
            processing_time=elapsed,
            processing_rate=processing_rate,
            total_size_mb=round(total_size / (1024 * 1024), 2),
            embeddings_generated=embeddings_generated,
            async_build=True,
        )

        if error_count > 0:
            suggestions = {
                "review": "Check file permissions and formats for failed documents",
                "command": "Use --verbose flag to see detailed error information",
                "retry": "Re-run indexing with --force flag if needed",
            }
            log_warning(
                f"{error_count} documents could not be processed (async)",
                error_count=error_count,
                suggestions=suggestions,
                async_build=True,
            )

        return index_data

    def _calculate_content_checksum(self, file_paths: List[Path]) -> str:
        """Calculate checksum of all document content for cache validation."""
        hasher = hashlib.sha256()

        # Sort paths for consistent checksum
        for file_path in sorted(file_paths):
            try:
                stat = file_path.stat()
                # Include path, size, and modification time in checksum
                content = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
                hasher.update(content.encode("utf-8"))
            except Exception:
                continue

        return hasher.hexdigest()

    def get_index_status(self) -> Dict[str, Any]:
        """Get current index status."""
        index_file = self.index_path / "index.yaml"

        if not index_file.exists():
            return {"exists": False, "total_documents": 0, "last_updated": None}

        try:
            with open(index_file, "r") as f:
                index_data = yaml.safe_load(f)

            return {
                "exists": True,
                "total_documents": index_data["metadata"]["total_documents"],
                "last_updated": index_data["metadata"]["timestamp"],
                "build_time": index_data["metadata"]["build_time"],
            }
        except Exception as e:
            return {
                "exists": False,
                "error": str(e),
                "total_documents": 0,
                "last_updated": None,
            }
