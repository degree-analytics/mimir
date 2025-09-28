"""
Document Embeddings - Semantic vector generation for mimir tool

Handles document embedding generation using sentence transformers for semantic search.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..core.cache import ThreadSafeCacheManager
from ..core.error_handling import (
    log_info,
    log_warning,
    log_error,
    log_success,
)


@dataclass
class EmbeddingResult:
    """Represents an embedding result with metadata."""

    document_path: str
    embedding: np.ndarray
    model_name: str
    chunk_index: int = 0
    chunk_text: str = ""


class DocumentEmbeddingGenerator:
    """Generates document embeddings using sentence transformers."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache_manager: Optional[ThreadSafeCacheManager] = None,
        verbose: bool = True,
    ):
        """Initialize the embedding generator."""
        self.config = config or {}
        self.cache_manager = cache_manager
        self.verbose = verbose
        self.model = None
        self.model_name = self.config.get("vector_search", {}).get(
            "model", "all-MiniLM-L6-v2"
        )
        self.enabled = self.config.get("vector_search", {}).get("enabled", False)
        self.dimension = self.config.get("vector_search", {}).get("dimension", 384)

        # Performance tracking
        self.last_embedding_time = 0
        self.embeddings_cache = {}

    def _lazy_load_model(self) -> bool:
        """Lazy load the sentence transformer model."""
        if self.model is not None:
            return True

        if not self.enabled:
            if self.verbose:
                log_warning("Vector search disabled in config")
            return False

        try:
            # Import here to avoid dependency issues if not installed
            from sentence_transformers import SentenceTransformer

            if self.verbose:
                log_info(f"Loading sentence transformer model: {self.model_name}", "📥")
            start_time = time.time()

            self.model = SentenceTransformer(self.model_name)

            load_time = time.time() - start_time
            if self.verbose:
                log_success(f"Model loaded in {load_time:.2f}s")
            return True

        except ImportError:
            if self.verbose:
                log_error(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
            return False
        except Exception as e:
            if self.verbose:
                log_error(f"Error loading model: {e}")
            return False

    def _chunk_text(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """Split text into chunks for embedding."""
        # Simple sentence-based chunking
        sentences = text.replace("\n", " ").split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Add sentence to current chunk if it fits
            if len(current_chunk + sentence) < max_chunk_size:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Ensure minimum chunk quality
        quality_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) > 20:  # Minimum meaningful chunk length
                quality_chunks.append(chunk.strip())

        return quality_chunks if quality_chunks else [text[:max_chunk_size]]

    def generate_embeddings(
        self, documents: List[Dict[str, Any]], force: bool = False
    ) -> List[EmbeddingResult]:
        """Generate embeddings for a list of documents."""
        if not self._lazy_load_model():
            return []

        if self.verbose:
            log_info("Generating document embeddings...", "🧠")
        start_time = time.time()

        all_embeddings = []
        processed_docs = 0

        for doc in documents:
            doc_path = doc["path"]

            # Check cache first (if not forcing regeneration)
            if not force and self.cache_manager:
                cached_embeddings = self.cache_manager.get_cached_embeddings(
                    doc_path, self.model_name
                )
                if cached_embeddings:
                    # Convert cached data back to EmbeddingResult objects
                    for emb_data in cached_embeddings:
                        embedding_array = np.array(emb_data["embedding"])
                        result = EmbeddingResult(
                            document_path=emb_data["document_path"],
                            embedding=embedding_array,
                            model_name=emb_data["model_name"],
                            chunk_index=emb_data.get("chunk_index", 0),
                            chunk_text=emb_data.get("chunk_text", ""),
                        )
                        all_embeddings.append(result)
                    continue

            # Generate new embeddings
            try:
                # Combine title and content for better semantic representation
                full_text = f"{doc['title']} {doc['content']}"

                # Split into chunks for large documents
                chunks = self._chunk_text(full_text)
                doc_embeddings = []

                for chunk_idx, chunk in enumerate(chunks):
                    # Generate embedding
                    embedding = self.model.encode([chunk], convert_to_numpy=True)[0]

                    result = EmbeddingResult(
                        document_path=doc_path,
                        embedding=embedding,
                        model_name=self.model_name,
                        chunk_index=chunk_idx,
                        chunk_text=chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    )

                    all_embeddings.append(result)
                    doc_embeddings.append(result)

                # Cache the embeddings for this document
                if self.cache_manager and doc_embeddings:
                    cache_data = []
                    for emb_result in doc_embeddings:
                        cache_data.append(
                            {
                                "document_path": emb_result.document_path,
                                "embedding": emb_result.embedding.tolist(),
                                "model_name": emb_result.model_name,
                                "chunk_index": emb_result.chunk_index,
                                "chunk_text": emb_result.chunk_text,
                            }
                        )
                    self.cache_manager.cache_embeddings(
                        doc_path, self.model_name, cache_data
                    )

                processed_docs += 1

            except Exception as e:
                if self.verbose:
                    log_error(f"Error generating embedding for {doc_path}: {e}")
                continue

        elapsed = time.time() - start_time
        self.last_embedding_time = elapsed

        if self.verbose:
            log_success(
                f"Generated {len(all_embeddings)} embeddings for {processed_docs} documents in {elapsed:.2f}s"
            )

        return all_embeddings

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        """Encode a search query into an embedding vector."""
        if not self._lazy_load_model():
            return None

        try:
            embedding = self.model.encode([query], convert_to_numpy=True)[0]
            return embedding
        except Exception as e:
            if self.verbose:
                log_error(f"Error encoding query: {e}")
            return None

    def compute_similarity(
        self, query_embedding: np.ndarray, doc_embeddings: List[EmbeddingResult]
    ) -> List[Tuple[EmbeddingResult, float]]:
        """Compute cosine similarity between query and document embeddings."""
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            # Prepare document embedding matrix
            embedding_matrix = np.array([emb.embedding for emb in doc_embeddings])

            # Compute similarities
            similarities = cosine_similarity([query_embedding], embedding_matrix)[0]

            # Pair embeddings with similarity scores
            results = list(zip(doc_embeddings, similarities))

            # Sort by similarity (highest first)
            results.sort(key=lambda x: x[1], reverse=True)

            return results

        except Exception as e:
            if self.verbose:
                log_error(f"Error computing similarity: {e}")
            return []

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics."""
        return {
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "enabled": self.enabled,
            "dimension": self.dimension,
            "last_embedding_time_s": round(self.last_embedding_time, 2),
            "cached_embeddings": len(self.embeddings_cache),
        }
