"""
Cache Manager - Multi-level caching system for mimir tool

Implements index persistence, query result caching, and embedding caching
with checksum-based invalidation and TTL management.
"""

import time
import json
import pickle
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    ttl: Optional[float] = None  # Time to live in seconds
    checksum: Optional[str] = None
    size_bytes: int = 0


@dataclass
class CacheStats:
    """Cache statistics and metrics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    last_cleanup: float = 0


class ThreadSafeCacheManager:
    """Thread-safe multi-level cache manager for mimir."""

    def __init__(self, cache_dir: Path, max_size_mb: int = 100):
        """Initialize cache manager."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache configuration
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.default_query_ttl = 24 * 60 * 60  # 24 hours in seconds
        self.cleanup_interval = 60 * 60  # 1 hour cleanup interval

        # Thread safety
        self._lock = threading.RLock()

        # Cache directories
        self.index_cache_dir = self.cache_dir / "index"
        self.query_cache_dir = self.cache_dir / "queries"
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self.stats_file = self.cache_dir / "cache_stats.json"

        # Create subdirectories
        self.index_cache_dir.mkdir(exist_ok=True)
        self.query_cache_dir.mkdir(exist_ok=True)
        self.embedding_cache_dir.mkdir(exist_ok=True)

        # Load statistics
        self.stats = self._load_stats()

        # Schedule periodic cleanup
        self._last_cleanup = time.time()

    def _load_stats(self) -> CacheStats:
        """Load cache statistics from disk."""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, "r") as f:
                    data = json.load(f)
                return CacheStats(**data)
        except Exception:
            pass
        return CacheStats()

    def _save_stats(self):
        """Save cache statistics to disk."""
        try:
            with open(self.stats_file, "w") as f:
                json.dump(asdict(self.stats), f, indent=2)
        except Exception:
            pass

    def _compute_checksum(self, data: Any) -> str:
        """Compute SHA256 checksum of data."""
        if isinstance(data, str):
            content = data.encode("utf-8")
        elif isinstance(data, bytes):
            content = data
        else:
            content = str(data).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _get_cache_key(
        self, query: str, mode: str = "smart", model: str = "default"
    ) -> str:
        """Generate cache key for query results."""
        key_data = f"{query}:{mode}:{model}"
        return hashlib.sha256(key_data.encode("utf-8")).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.ttl is None:
            return False
        return time.time() - entry.created_at > entry.ttl

    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        current_time = time.time()

        # Only cleanup if enough time has passed
        if current_time - self._last_cleanup < self.cleanup_interval:
            return

        with self._lock:
            # Cleanup query cache
            for cache_file in self.query_cache_dir.glob("*.pkl"):
                try:
                    with open(cache_file, "rb") as f:
                        entry = pickle.load(f)

                    if self._is_expired(entry):
                        cache_file.unlink()
                        self.stats.evictions += 1

                except Exception:
                    # Remove corrupted cache files
                    cache_file.unlink()

            # Update stats
            self.stats.last_cleanup = current_time
            self._last_cleanup = current_time
            self._save_stats()

    def _enforce_size_limits(self):
        """Enforce cache size limits using LRU eviction."""
        with self._lock:
            # Get all cache files with their access times
            cache_files = []
            total_size = 0

            for cache_dir in [self.query_cache_dir, self.embedding_cache_dir]:
                for cache_file in cache_dir.glob("*.pkl"):
                    try:
                        stat = cache_file.stat()
                        cache_files.append((cache_file, stat.st_atime, stat.st_size))
                        total_size += stat.st_size
                    except Exception:
                        continue

            # If over limit, remove oldest files
            if total_size > self.max_size_bytes:
                # Sort by access time (oldest first)
                cache_files.sort(key=lambda x: x[1])

                while total_size > self.max_size_bytes and cache_files:
                    file_path, _, file_size = cache_files.pop(0)
                    try:
                        file_path.unlink()
                        total_size -= file_size
                        self.stats.evictions += 1
                    except Exception:
                        continue

    # Index Cache Methods

    def cache_index(self, index_data: Dict[str, Any], checksum: str) -> bool:
        """Cache document index with checksum validation."""
        try:
            with self._lock:
                cache_file = self.index_cache_dir / "document_index.pkl"

                entry = CacheEntry(
                    key="document_index",
                    value=index_data,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    checksum=checksum,
                    size_bytes=len(str(index_data)),
                )

                with open(cache_file, "wb") as f:
                    pickle.dump(entry, f)

                return True

        except Exception:
            return False

    def get_cached_index(self, current_checksum: str) -> Optional[Dict[str, Any]]:
        """Get cached index if checksum matches."""
        try:
            with self._lock:
                cache_file = self.index_cache_dir / "document_index.pkl"

                if not cache_file.exists():
                    self.stats.misses += 1
                    return None

                with open(cache_file, "rb") as f:
                    entry = pickle.load(f)

                # Validate checksum
                if entry.checksum != current_checksum:
                    self.stats.misses += 1
                    return None

                # Update access time
                entry.accessed_at = time.time()
                with open(cache_file, "wb") as f:
                    pickle.dump(entry, f)

                self.stats.hits += 1
                return entry.value

        except Exception:
            self.stats.misses += 1
            return None

    # Query Cache Methods

    def cache_query_result(
        self, query: str, mode: str, results: List[Any], ttl: Optional[float] = None
    ) -> bool:
        """Cache query results with TTL."""
        try:
            with self._lock:
                cache_key = self._get_cache_key(query, mode)
                cache_file = self.query_cache_dir / f"{cache_key}.pkl"

                if ttl is None:
                    ttl = self.default_query_ttl

                entry = CacheEntry(
                    key=cache_key,
                    value=results,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    ttl=ttl,
                    size_bytes=len(str(results)),
                )

                with open(cache_file, "wb") as f:
                    pickle.dump(entry, f)

                # Periodic cleanup and size enforcement
                self._cleanup_expired_entries()
                self._enforce_size_limits()

                return True

        except Exception:
            return False

    def get_cached_query_result(self, query: str, mode: str) -> Optional[List[Any]]:
        """Get cached query results if not expired."""
        try:
            with self._lock:
                cache_key = self._get_cache_key(query, mode)
                cache_file = self.query_cache_dir / f"{cache_key}.pkl"

                if not cache_file.exists():
                    self.stats.misses += 1
                    return None

                with open(cache_file, "rb") as f:
                    entry = pickle.load(f)

                # Check if expired
                if self._is_expired(entry):
                    cache_file.unlink()
                    self.stats.misses += 1
                    self.stats.evictions += 1
                    return None

                # Update access time
                entry.accessed_at = time.time()
                with open(cache_file, "wb") as f:
                    pickle.dump(entry, f)

                self.stats.hits += 1
                return entry.value

        except Exception:
            self.stats.misses += 1
            return None

    # Embedding Cache Methods (for future use)

    def cache_embeddings(self, doc_path: str, model_name: str, embeddings: Any) -> bool:
        """Cache document embeddings with document path and model name."""
        try:
            with self._lock:
                # Create unique cache key from doc path and model
                cache_key = hashlib.sha256(
                    f"{doc_path}:{model_name}".encode("utf-8")
                ).hexdigest()
                cache_file = self.embedding_cache_dir / f"{cache_key}.pkl"

                entry = CacheEntry(
                    key=cache_key,
                    value=embeddings,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    checksum=cache_key,
                    size_bytes=len(str(embeddings)),
                )

                with open(cache_file, "wb") as f:
                    pickle.dump(entry, f)

                return True

        except Exception:
            return False

    def get_cached_embeddings(self, doc_path: str, model_name: str) -> Optional[Any]:
        """Get cached embeddings for document with specific model."""
        try:
            with self._lock:
                # Create same cache key as in cache_embeddings
                cache_key = hashlib.sha256(
                    f"{doc_path}:{model_name}".encode("utf-8")
                ).hexdigest()
                cache_file = self.embedding_cache_dir / f"{cache_key}.pkl"

                if not cache_file.exists():
                    self.stats.misses += 1
                    return None

                with open(cache_file, "rb") as f:
                    entry = pickle.load(f)

                # Update access time
                entry.accessed_at = time.time()
                with open(cache_file, "wb") as f:
                    pickle.dump(entry, f)

                self.stats.hits += 1
                return entry.value

        except Exception:
            self.stats.misses += 1
            return None

    # Cache Management Methods

    def clear_cache(self, cache_type: str = "all") -> Dict[str, int]:
        """Clear cache entries."""
        cleared = {"index": 0, "queries": 0, "embeddings": 0}

        with self._lock:
            if cache_type in ["all", "index"]:
                for cache_file in self.index_cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    cleared["index"] += 1

            if cache_type in ["all", "queries"]:
                for cache_file in self.query_cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    cleared["queries"] += 1

            if cache_type in ["all", "embeddings"]:
                for cache_file in self.embedding_cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    cleared["embeddings"] += 1

            # Reset stats for cleared caches
            if cache_type == "all":
                self.stats = CacheStats()

            self._save_stats()

        return cleared

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            # Calculate current cache sizes
            cache_sizes = {}
            total_files = 0
            total_size = 0

            for name, cache_dir in [
                ("index", self.index_cache_dir),
                ("queries", self.query_cache_dir),
                ("embeddings", self.embedding_cache_dir),
            ]:
                size = 0
                files = 0
                for cache_file in cache_dir.glob("*.pkl"):
                    try:
                        stat = cache_file.stat()
                        size += stat.st_size
                        files += 1
                    except Exception:
                        continue

                cache_sizes[name] = {"size_bytes": size, "files": files}
                total_size += size
                total_files += files

            # Calculate hit rate
            total_requests = self.stats.hits + self.stats.misses
            hit_rate = (
                (self.stats.hits / total_requests * 100) if total_requests > 0 else 0
            )

            return {
                "hit_rate_percent": round(hit_rate, 2),
                "total_hits": self.stats.hits,
                "total_misses": self.stats.misses,
                "total_evictions": self.stats.evictions,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "total_files": total_files,
                "max_size_mb": round(self.max_size_bytes / (1024 * 1024), 2),
                "cache_sizes": cache_sizes,
                "last_cleanup": self.stats.last_cleanup,
            }
