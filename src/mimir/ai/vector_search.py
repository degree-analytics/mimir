"""
Vector Similarity Search - FAISS-based efficient vector search for mimir tool

Implements high-performance vector similarity search using FAISS with support for
different index types and similarity metrics.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
from ..core.cache import ThreadSafeCacheManager
from .embeddings import EmbeddingResult


@dataclass
class VectorSearchResult:
    """Represents a vector search result with similarity score."""

    document_path: str
    title: str
    content_snippet: str
    similarity_score: float
    chunk_index: int
    embedding_metadata: Dict[str, Any]
    distance: float  # Raw FAISS distance


class FAISSVectorSearchEngine:
    """High-performance vector search engine using FAISS."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache_manager: Optional[ThreadSafeCacheManager] = None,
    ):
        """Initialize FAISS vector search engine."""
        self.config = config or {}
        self.cache_manager = cache_manager
        self.embedding_generator = None

        # FAISS configuration
        self.faiss_enabled = self.config.get("vector_search", {}).get(
            "faiss_enabled", True
        )
        self.index_type = self.config.get("vector_search", {}).get(
            "faiss_index_type", "IndexFlatIP"
        )  # Inner Product
        self.dimension = self.config.get("vector_search", {}).get("dimension", 384)
        self.similarity_threshold = self.config.get("vector_search", {}).get(
            "similarity_threshold", 0.5
        )

        # FAISS index and data
        self.faiss_index = None
        self.document_embeddings = []  # List of EmbeddingResult objects
        self.index_metadata = {}

        # Performance tracking
        self.last_search_time = 0
        self.index_build_time = 0

    def _lazy_load_faiss(self) -> bool:
        """Lazy load FAISS library."""
        if not self.faiss_enabled:
            return False

        try:
            import faiss

            self.faiss = faiss
            return True
        except ImportError:
            print("❌ FAISS not installed. Run: pip install faiss-cpu")
            return False

    def _create_faiss_index(self, dimension: int) -> Any:
        """Create FAISS index based on configuration."""
        if not self._lazy_load_faiss():
            return None

        index_type = self.index_type

        if index_type == "IndexFlatIP":
            # Inner Product (cosine similarity for normalized vectors)
            index = self.faiss.IndexFlatIP(dimension)
        elif index_type == "IndexFlatL2":
            # L2 distance (Euclidean)
            index = self.faiss.IndexFlatL2(dimension)
        elif index_type == "IndexIVFFlat":
            # Inverted File with Flat quantizer (faster for large datasets)
            nlist = min(
                100, max(1, len(self.document_embeddings) // 10)
            )  # Adaptive nlist
            quantizer = self.faiss.IndexFlatIP(dimension)
            index = self.faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif index_type == "IndexHNSWFlat":
            # Hierarchical Navigable Small World (very fast approximate search)
            index = self.faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
        else:
            # Default to flat inner product
            print(f"⚠️  Unknown index type {index_type}, using IndexFlatIP")
            index = self.faiss.IndexFlatIP(dimension)

        return index

    def build_faiss_index(
        self, embeddings: List[EmbeddingResult], force_rebuild: bool = False
    ) -> bool:
        """Build FAISS index from embeddings."""
        if not self._lazy_load_faiss():
            print("⚠️  FAISS not available, falling back to scikit-learn similarity")
            return False

        if not embeddings:
            print("❌ No embeddings provided for FAISS index")
            return False

        print(f"🔨 Building FAISS index with {len(embeddings)} embeddings...")
        start_time = time.time()

        # Check cache first
        cache_key = self._get_index_cache_key(embeddings)
        if not force_rebuild and self.cache_manager:
            cached_index = self._load_cached_faiss_index(cache_key)
            if cached_index:
                self.faiss_index, self.document_embeddings, self.index_metadata = (
                    cached_index
                )
                print(
                    f"⚡ Loaded FAISS index from cache ({len(self.document_embeddings)} embeddings)"
                )
                return True

        # Extract embedding vectors
        embedding_matrix = np.array([emb.embedding for emb in embeddings]).astype(
            "float32"
        )

        # Normalize vectors for cosine similarity (if using IndexFlatIP)
        if self.index_type == "IndexFlatIP":
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            embedding_matrix = embedding_matrix / (
                norms + 1e-8
            )  # Avoid division by zero

        # Create and train index
        self.faiss_index = self._create_faiss_index(embedding_matrix.shape[1])
        if self.faiss_index is None:
            print("❌ Failed to create FAISS index")
            return False

        # Train index if needed (for IVF indexes)
        if hasattr(self.faiss_index, "train"):
            print("🎯 Training FAISS index...")
            self.faiss_index.train(embedding_matrix)

        # Add vectors to index
        self.faiss_index.add(embedding_matrix)
        self.document_embeddings = embeddings

        # Store metadata
        self.index_metadata = {
            "index_type": self.index_type,
            "dimension": embedding_matrix.shape[1],
            "total_vectors": len(embeddings),
            "build_time": time.time() - start_time,
            "timestamp": time.time(),
        }

        self.index_build_time = self.index_metadata["build_time"]

        # Cache the index
        if self.cache_manager:
            self._cache_faiss_index(cache_key)

        print(f"✅ FAISS index built in {self.index_build_time:.2f}s")
        print(f"📊 Index type: {self.index_type}, Vectors: {len(embeddings)}")

        return True

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[VectorSearchResult]:
        """Search using FAISS index."""
        if self.faiss_index is None:
            print("❌ FAISS index not built")
            return []

        start_time = time.time()

        # Normalize query embedding for cosine similarity
        if self.index_type == "IndexFlatIP":
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm

        # Prepare query for FAISS (must be 2D)
        query_vector = query_embedding.astype("float32").reshape(1, -1)

        # Search
        similarities, indices = self.faiss_index.search(query_vector, top_k)

        # Convert results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for no more results
                break

            if similarity < self.similarity_threshold:
                continue

            embedding_result = self.document_embeddings[idx]

            # Create search result
            result = VectorSearchResult(
                document_path=embedding_result.document_path,
                title="",  # Will be filled by caller
                content_snippet=embedding_result.chunk_text,
                similarity_score=float(similarity),
                chunk_index=embedding_result.chunk_index,
                embedding_metadata={
                    "model_name": embedding_result.model_name,
                    "chunk_index": embedding_result.chunk_index,
                },
                distance=float(similarities[0][i]),
            )
            results.append(result)

        self.last_search_time = time.time() - start_time
        return results

    def _get_index_cache_key(self, embeddings: List[EmbeddingResult]) -> str:
        """Generate cache key for FAISS index."""
        import hashlib

        # Create key from embeddings metadata and config
        key_data = f"{len(embeddings)}:{self.index_type}:{self.dimension}"
        if embeddings:
            key_data += f":{embeddings[0].model_name}"

        return hashlib.sha256(key_data.encode("utf-8")).hexdigest()

    def _cache_faiss_index(self, cache_key: str) -> bool:
        """Cache FAISS index to disk."""
        if not self.cache_manager or not self.faiss_index:
            return False

        try:
            cache_dir = self.cache_manager.cache_dir / "faiss_indexes"
            cache_dir.mkdir(exist_ok=True)

            # Save FAISS index
            index_file = cache_dir / f"{cache_key}.faiss"
            self.faiss.write_index(self.faiss_index, str(index_file))

            # Save embeddings and metadata
            data_file = cache_dir / f"{cache_key}.pkl"
            with open(data_file, "wb") as f:
                pickle.dump(
                    {
                        "embeddings": self.document_embeddings,
                        "metadata": self.index_metadata,
                    },
                    f,
                )

            return True

        except Exception as e:
            print(f"⚠️  Failed to cache FAISS index: {e}")
            return False

    def _load_cached_faiss_index(self, cache_key: str) -> Optional[Tuple]:
        """Load cached FAISS index from disk."""
        if not self.cache_manager:
            return None

        try:
            cache_dir = self.cache_manager.cache_dir / "faiss_indexes"

            index_file = cache_dir / f"{cache_key}.faiss"
            data_file = cache_dir / f"{cache_key}.pkl"

            if not index_file.exists() or not data_file.exists():
                return None

            # Load FAISS index
            if not self._lazy_load_faiss():
                return None

            faiss_index = self.faiss.read_index(str(index_file))

            # Load embeddings and metadata
            with open(data_file, "rb") as f:
                data = pickle.load(f)

            return faiss_index, data["embeddings"], data["metadata"]

        except Exception as e:
            print(f"⚠️  Failed to load cached FAISS index: {e}")
            return None

    def get_index_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics."""
        if not self.faiss_index:
            return {"index_built": False, "faiss_available": self._lazy_load_faiss()}

        stats = {
            "index_built": True,
            "faiss_available": True,
            "index_type": self.index_type,
            "total_vectors": self.faiss_index.ntotal,
            "dimension": self.dimension,
            "build_time_s": round(self.index_build_time, 3),
            "last_search_time_ms": round(self.last_search_time * 1000, 2),
            "similarity_threshold": self.similarity_threshold,
        }

        # Add index-specific stats
        if hasattr(self.faiss_index, "nlist"):
            stats["nlist"] = self.faiss_index.nlist
        if hasattr(self.faiss_index, "hnsw"):
            stats["hnsw_efSearch"] = self.faiss_index.hnsw.efSearch

        return stats

    def benchmark_search_performance(
        self, query_embedding: np.ndarray, iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark search performance."""
        if not self.faiss_index:
            return {"error": "Index not built"}

        print(f"🏃 Benchmarking FAISS search performance ({iterations} iterations)...")

        times = []
        for _ in range(iterations):
            start = time.time()
            self.search(query_embedding, top_k=10)
            times.append(time.time() - start)

        return {
            "avg_time_ms": np.mean(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "total_vectors": self.faiss_index.ntotal,
        }
