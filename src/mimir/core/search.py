"""
Search Engine - TF-IDF based document search for mimir tool

Implements fast keyword-based search using TF-IDF (Term Frequency-Inverse Document Frequency)
with preprocessing pipeline optimized for technical documentation.
"""

import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import yaml
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .cache import ThreadSafeCacheManager
from ..ai.embeddings import DocumentEmbeddingGenerator
from ..ai.vector_search import FAISSVectorSearchEngine
from ..ai.llm_integration import DocumentLLMEnhancer
from .telemetry import TelemetryManager
from .cost_engine import LLMCostEngine
from .settings import get_project_root
from abc import ABC, abstractmethod
from .error_handling import log_info, log_warning, log_error, log_success


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List["SearchResult"]:
        """Perform search and return results."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this search strategy is available."""
        pass


class TFIDFSearchStrategy(SearchStrategy):
    """TF-IDF based search strategy."""

    def __init__(
        self,
        preprocessor: "TechnicalDocumentationPreprocessor",
        cache_manager: Optional["ThreadSafeCacheManager"] = None,
    ):
        self.preprocessor = preprocessor
        self.cache_manager = cache_manager
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.inverted_index = {}
        self.last_search_time = 0

    def is_available(self) -> bool:
        return self.vectorizer is not None and self.tfidf_matrix is not None

    def build_index(self, documents: List[Dict[str, Any]]) -> bool:
        """Build TF-IDF index for documents."""
        self.documents = documents

        # Prepare documents for TF-IDF
        processed_docs = []
        for doc in documents:
            full_text = f"{doc['title']} {doc['content']}"
            processed_text = self.preprocessor.preprocess_text(full_text)
            processed_docs.append(processed_text)

        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=1, max_df=0.8, stop_words=None
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        self._build_inverted_index(processed_docs)
        return True

    def _build_inverted_index(self, processed_docs: List[str]):
        """Build inverted index for exact keyword lookup."""
        self.inverted_index = {}
        for doc_idx, processed_text in enumerate(processed_docs):
            words = processed_text.split()
            for word in set(words):
                if word not in self.inverted_index:
                    self.inverted_index[word] = []
                self.inverted_index[word].append(doc_idx)

    def search(
        self, query: str, top_k: int = 10, mode: str = "smart"
    ) -> List["SearchResult"]:
        """Perform TF-IDF search with exact match boosting."""
        if not self.is_available():
            return []

        # Check cache first
        if self.cache_manager:
            cached_results = self.cache_manager.get_cached_query_result(query, mode)
            if cached_results:
                # Handle cached results that may not have new fields
                results = []
                for result_dict in cached_results:
                    # Ensure backwards compatibility with cached results
                    if "context" not in result_dict:
                        result_dict["context"] = None
                    if "actionable_info" not in result_dict:
                        result_dict["actionable_info"] = None
                    results.append(SearchResult.from_dict(result_dict))
                self.last_search_time = 0.001
                return results

        start_time = time.time()
        processed_query = self.preprocessor.preprocess_text(query)
        if not processed_query.strip():
            return []

        # Get TF-IDF and exact match scores
        query_vector = self.vectorizer.transform([processed_query])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        exact_match_scores, exact_match_counts = self._get_exact_match_scores(
            query.lower()
        )

        # Combine scores
        combined_scores = tfidf_scores + (exact_match_scores * 0.3)
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0:
                doc = self.documents[idx]
                snippet = self._generate_snippet(doc["content"], query)

                # Generate enhanced context and actionable info for LLM consumption
                context = self._extract_search_result_context(
                    doc, query, exact_match_counts[idx]
                )
                actionable_info = self._extract_actionable_info(
                    doc, query, float(combined_scores[idx])
                )

                result = SearchResult(
                    document_path=doc["path"],
                    title=doc["title"],
                    content_snippet=snippet,
                    relevance_score=float(combined_scores[idx]),
                    exact_matches=exact_match_counts[idx],
                    metadata=doc.get("metadata", {}),
                    context=context,
                    actionable_info=actionable_info,
                )
                results.append(result)

        self.last_search_time = time.time() - start_time

        # Cache results
        if self.cache_manager and results:
            result_dicts = [result.to_dict() for result in results]
            self.cache_manager.cache_query_result(query, mode, result_dicts)

        return results

    def _get_exact_match_scores(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate exact match scores and counts."""
        scores = np.zeros(len(self.documents))
        counts = np.zeros(len(self.documents), dtype=int)

        query_words = query.split()
        for word in query_words:
            if word in self.inverted_index:
                for doc_idx in self.inverted_index[word]:
                    scores[doc_idx] += 1.0
                    counts[doc_idx] += 1

        if len(query_words) > 0:
            scores = scores / len(query_words)

        return scores, counts

    def _generate_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """Generate a relevant snippet from document content with basic term highlighting."""
        query_words = query.lower().split()
        content_lower = content.lower()

        best_pos = 0
        best_score = 0

        # Find the best position that contains the most query words
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1:
                snippet_start = max(0, pos - 100)
                snippet_end = min(len(content), pos + 100)
                snippet = content_lower[snippet_start:snippet_end]

                score = sum(1 for w in query_words if w in snippet)
                if score > best_score:
                    best_score = score
                    best_pos = snippet_start

        # Extract snippet
        snippet_end = min(len(content), best_pos + max_length)
        snippet = content[best_pos:snippet_end].strip()

        # Add basic term highlighting with **bold** markup
        import re

        for word in query_words:
            if word:  # Skip empty words
                snippet = re.sub(
                    f"({re.escape(word)})", r"**\1**", snippet, flags=re.IGNORECASE
                )

        # Add ellipsis for truncated content
        if best_pos > 0:
            snippet = "..." + snippet
        if snippet_end < len(content):
            snippet = snippet + "..."

        return snippet

    def _extract_search_result_context(
        self, doc: Dict[str, Any], query: str, exact_matches: int
    ) -> Optional["SearchResultContext"]:
        """Extract rich context information for SearchResult.

        Args:
            doc: Document dictionary with content and metadata
            query: Search query for context extraction
            exact_matches: Number of exact matches found

        Returns:
            SearchResultContext object with structured context data
        """
        content = doc.get("content", "")
        if not content:
            return None

        lines = content.split("\n")
        query_words = [word.lower() for word in query.split()]

        # Extract core context components using helper methods
        line_numbers = self._extract_matched_line_numbers(lines, query_words)
        section_headers = self._extract_section_headers(lines, line_numbers)
        surrounding_context = self._extract_surrounding_context(
            lines, line_numbers, query_words, section_headers
        )
        document_metadata = self._build_document_metadata(doc, content, lines)
        matched_sections = self._build_matched_sections(section_headers, content)
        related_files = self._extract_related_files(content)

        return SearchResultContext(
            section_headers=section_headers,
            line_numbers=line_numbers[:10],  # Limit to first 10 matches
            surrounding_context=surrounding_context,
            document_metadata=document_metadata,
            matched_sections=matched_sections,
            related_files=related_files,
        )

    def _extract_matched_line_numbers(
        self, lines: List[str], query_words: List[str]
    ) -> List[int]:
        """Extract line numbers where query terms appear."""
        line_numbers = []
        for line_num, line in enumerate(lines, 1):
            if any(word in line.lower() for word in query_words):
                line_numbers.append(line_num)
        return line_numbers

    def _extract_section_headers(
        self, lines: List[str], line_numbers: List[int]
    ) -> List[str]:
        """Extract markdown section headers that contain matched lines."""
        section_headers = []
        current_headers = []
        header_line_ranges = {}  # Track which lines belong to which headers

        for line_num, line in enumerate(lines, 1):
            # Detect markdown headers
            if line.strip().startswith("#"):
                header_level = len(line) - len(line.lstrip("#"))
                header_text = line.strip("#").strip()

                # Maintain hierarchy - keep headers up to current level
                current_headers = current_headers[: header_level - 1]
                current_headers.append(header_text)

                # Track header ranges (simplified - header applies until next header of same/higher level)
                header_key = " > ".join(current_headers)
                header_line_ranges[header_key] = line_num

        # Find which section contains the matched lines
        if line_numbers and header_line_ranges:
            first_match_line = line_numbers[0]

            # Find the most recent header before the first match
            closest_header = None
            closest_header_line = 0

            for header_key, header_line in header_line_ranges.items():
                if (
                    header_line <= first_match_line
                    and header_line > closest_header_line
                ):
                    closest_header = header_key
                    closest_header_line = header_line

            if closest_header:
                section_headers = closest_header.split(" > ")

        return section_headers

    def _extract_surrounding_context(
        self,
        lines: List[str],
        line_numbers: List[int],
        query_words: List[str],
        section_headers: List[str],
    ) -> Dict[str, Any]:
        """Extract surrounding context around matches with term highlighting."""
        surrounding_context = {}

        if line_numbers:
            # Get context around multiple matches (up to 3 matches)
            contexts = []
            for i, match_line in enumerate(line_numbers[:3]):
                # Get context before match (2-3 lines before)
                before_start = max(
                    0, match_line - 4
                )  # -1 for 0-indexing, -3 for context
                before_lines = lines[before_start : match_line - 1]

                # Get context after match (2-3 lines after)
                after_end = min(len(lines), match_line + 3)
                after_lines = lines[match_line:after_end]

                # Highlight terms in the matched lines
                highlighted_lines = []
                for line in after_lines:
                    highlighted_line = line
                    for word in query_words:
                        if word in line.lower():
                            # Simple highlighting with **bold** markup
                            import re

                            highlighted_line = re.sub(
                                f"({re.escape(word)})",
                                r"**\1**",
                                highlighted_line,
                                flags=re.IGNORECASE,
                            )
                    highlighted_lines.append(highlighted_line)

                contexts.append(
                    {
                        "match_line": match_line,
                        "before": "\n".join(before_lines).strip(),
                        "after": "\n".join(highlighted_lines).strip(),
                    }
                )

            # For backwards compatibility, use first match for main context
            if contexts:
                surrounding_context["before"] = contexts[0]["before"]
                surrounding_context["after"] = contexts[0]["after"]

                # Add detailed contexts for multiple matches
                if len(contexts) > 1:
                    surrounding_context["all_matches"] = contexts

            # Get full section context if we have section headers
            if section_headers:
                surrounding_context["full_section"] = (
                    f"Section: {' > '.join(section_headers)}"
                )

        return surrounding_context

    def _build_document_metadata(
        self, doc: Dict[str, Any], content: str, lines: List[str]
    ) -> Dict[str, Any]:
        """Build comprehensive document metadata."""
        document_metadata = {
            "file_type": "markdown" if doc["path"].endswith(".md") else "text",
            "title": doc.get("title", ""),
            "path": doc.get("path", ""),
            "content_length": len(content),
            "total_lines": len(lines),
        }

        # Add any existing metadata
        if "metadata" in doc:
            document_metadata.update(doc["metadata"])

        return document_metadata

    def _build_matched_sections(
        self, section_headers: List[str], content: str
    ) -> List[Dict[str, Any]]:
        """Build matched sections with content previews."""
        matched_sections = []
        for i, header in enumerate(section_headers):
            matched_sections.append(
                {
                    "level": i + 1,
                    "title": header,
                    "content_preview": self._generate_snippet(
                        content, header, max_length=100
                    ),
                }
            )
        return matched_sections

    def _extract_related_files(self, content: str) -> List[str]:
        """Extract related files from markdown links."""
        related_files = []
        import re

        # Find markdown links [text](path)
        link_pattern = r"\[([^\]]+)\]\(([^)]+\.md[^)]*)\)"
        links = re.findall(link_pattern, content)
        for text, path in links:
            if path not in related_files:
                related_files.append(path)

        # Limit to most relevant related files
        return related_files[:5]

    def _extract_actionable_info(
        self, doc: Dict[str, Any], query: str, relevance_score: float
    ) -> Optional["ActionableInfo"]:
        """Extract actionable information for immediate use.

        Args:
            doc: Document dictionary with content and metadata
            query: Search query for context
            relevance_score: Relevance score for confidence calculation

        Returns:
            ActionableInfo object with commands, prerequisites, and guidance
        """
        content = doc.get("content", "")
        if not content:
            return None

        lines = content.split("\n")

        # Extract actionable components using helper methods
        related_commands = self._extract_commands_from_content(content)
        prerequisites = self._extract_prerequisites_from_content(lines)
        confidence_score = self._calculate_actionable_confidence(relevance_score)
        suggested_actions = self._generate_context_specific_actions(doc, content)
        workflow_steps = self._extract_workflow_steps(lines)
        example_usage = self._extract_example_usage(content)

        return ActionableInfo(
            related_commands=related_commands[:5],  # Limit to top 5 commands
            prerequisites=prerequisites,
            confidence_score=confidence_score,
            suggested_actions=suggested_actions[:4],  # Limit to top 4 actions
            workflow_steps=workflow_steps,
            example_usage=example_usage,
        )

    def _extract_commands_from_content(self, content: str) -> List[str]:
        """Extract executable commands from code blocks and inline patterns."""
        import re

        related_commands = []

        # Find bash/shell commands
        bash_pattern = r"```(?:bash|shell|sh)\n(.*?)```"
        bash_matches = re.findall(bash_pattern, content, re.DOTALL)
        for match in bash_matches:
            commands = [
                line.strip()
                for line in match.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            related_commands.extend(commands[:2])  # Max 2 commands per block

        # Find inline commands (backticks with command-like patterns)
        inline_pattern = r"`([^`]*(?:just|docker|npm|pip|git|curl|python|node)[^`]*)`"
        inline_matches = re.findall(inline_pattern, content, re.IGNORECASE)
        related_commands.extend(inline_matches[:3])

        return related_commands

    def _extract_prerequisites_from_content(self, lines: List[str]) -> List[str]:
        """Extract prerequisite information from content lines."""
        prerequisites = []
        prereq_indicators = [
            "prerequisite",
            "requirement",
            "need",
            "install",
            "setup",
            "before",
            "first",
            "ensure",
            "make sure",
        ]

        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in prereq_indicators):
                # Extract meaningful prerequisites
                if "install" in line_lower:
                    prerequisites.append(line.strip())
                elif "prerequisite" in line_lower or "requirement" in line_lower:
                    prerequisites.append(line.strip())
                elif "ensure" in line_lower or "make sure" in line_lower:
                    prerequisites.append(line.strip())

        return prerequisites[:3]  # Limit to 3 most relevant

    def _calculate_actionable_confidence(self, relevance_score: float) -> float:
        """Calculate confidence score for actionable information."""
        return min(0.95, relevance_score * 0.8 + 0.2)

    def _generate_context_specific_actions(
        self, doc: Dict[str, Any], content: str
    ) -> List[str]:
        """Generate suggested actions based on document type and content."""
        suggested_actions = []
        doc_path = doc.get("path", "").lower()

        if "api" in doc_path or "api" in content.lower():
            suggested_actions.extend(
                [
                    "Review API endpoint documentation",
                    "Test API calls with provided examples",
                    "Check authentication requirements",
                ]
            )
        elif "deploy" in doc_path or "deploy" in content.lower():
            suggested_actions.extend(
                [
                    "Review deployment prerequisites",
                    "Test deployment in staging environment",
                    "Verify environment configuration",
                ]
            )
        elif "setup" in doc_path or "install" in content.lower():
            suggested_actions.extend(
                [
                    "Follow installation steps carefully",
                    "Verify system requirements",
                    "Test setup with provided examples",
                ]
            )
        else:
            suggested_actions.extend(
                [
                    "Read through complete documentation",
                    "Follow examples and code samples",
                    "Review related documentation links",
                ]
            )

        return suggested_actions

    def _extract_workflow_steps(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract workflow steps from numbered lists or step-by-step content."""
        import re

        workflow_steps = []
        step_patterns = [
            r"^(\d+)\.\s*(.+)$",  # 1. Step description
            r"^Step\s*(\d+):\s*(.+)$",  # Step 1: Description
            r"^##\s*Step\s*(\d+)[:\s]*(.+)$",  # ## Step 1: Description
        ]

        for line in lines:
            line = line.strip()
            for pattern in step_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    step_num = int(match.group(1))
                    step_desc = match.group(2).strip()
                    if len(workflow_steps) < 5:  # Limit to 5 steps
                        workflow_steps.append(
                            {
                                "step": step_num,
                                "description": step_desc,
                                "type": "procedural",
                            }
                        )
                    break

        return workflow_steps

    def _extract_example_usage(self, content: str) -> List[str]:
        """Extract example usage patterns from content."""
        import re

        example_usage = []

        # Code examples in backticks
        code_examples = re.findall(r"`([^`]+)`", content)
        for example in code_examples:
            if any(
                keyword in example.lower()
                for keyword in ["curl", "http", "api", "command", "="]
            ):
                example_usage.append(example)
                if len(example_usage) >= 3:
                    break

        return example_usage


@dataclass
class SearchResultContext:
    """Rich context information for search results optimized for LLM consumption.

    This dataclass provides structured context that helps LLMs understand:
    - Document structure and hierarchy
    - Precise location information
    - Relationships to other content
    - Surrounding context for better comprehension

    Attributes:
        section_headers: Hierarchical list of section headers containing the match
        line_numbers: Specific line numbers where matches occur
        surrounding_context: Context before/after matches (keys: 'before', 'after', 'full_section')
        document_metadata: Document-level metadata (author, created_date, tags, etc.)
        matched_sections: Detailed information about matched sections with content
        related_files: List of related/referenced files mentioned in context
    """

    section_headers: List[str] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)
    surrounding_context: Dict[str, str] = field(default_factory=dict)
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    matched_sections: List[Dict[str, Any]] = field(default_factory=list)
    related_files: List[str] = field(default_factory=list)


@dataclass
class ActionableInfo:
    """Actionable information extracted from search results for immediate use.

    This dataclass structures information that enables direct action:
    - Commands that can be run immediately
    - Prerequisites needed before proceeding
    - Confidence levels for decision making
    - Step-by-step guidance
    - Real usage examples

    Attributes:
        related_commands: Shell commands, CLI calls, or code snippets ready to execute
        prerequisites: Dependencies, setup steps, or conditions required first
        confidence_score: Float 0.0-1.0 indicating reliability of the actionable info
        suggested_actions: Human-readable next steps or recommendations
        workflow_steps: Structured step-by-step processes with titles and descriptions
        example_usage: Working examples or code samples demonstrating usage
    """

    related_commands: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    suggested_actions: List[str] = field(default_factory=list)
    workflow_steps: List[Dict[str, Any]] = field(default_factory=list)
    example_usage: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Enhanced search result with rich context and actionable information for LLM consumption.

    This dataclass extends the original SearchResult with structured context and actionable
    information designed for optimal LLM consumption and Claude Code integration.

    Fields:
        document_path: Path to the source document
        title: Document title or section header
        content_snippet: Relevant content excerpt
        relevance_score: Numerical relevance score (0.0-1.0)
        exact_matches: Number of exact keyword matches
        metadata: Legacy metadata dictionary
        context: Rich context information including sections, line numbers, etc.
        actionable_info: Commands, prerequisites, and actionable guidance

    Example:
        >>> result = SearchResult(
        ...     document_path="/path/to/doc.md",
        ...     title="API Documentation",
        ...     content_snippet="Example content...",
        ...     relevance_score=0.95,
        ...     exact_matches=2,
        ...     metadata={},
        ...     context=SearchResultContext(
        ...         section_headers=["Getting Started", "Authentication"],
        ...         line_numbers=[10, 15, 20]
        ...     ),
        ...     actionable_info=ActionableInfo(
        ...         related_commands=["curl -X POST", "npm install"],
        ...         confidence_score=0.9
        ...     )
        ... )
        >>> data = result.to_dict()  # JSON serializable
        >>> restored = SearchResult.from_dict(data)  # Round-trip
    """

    document_path: str
    title: str
    content_snippet: str
    relevance_score: float
    exact_matches: int
    metadata: Dict[str, Any]
    context: Optional[SearchResultContext] = None
    actionable_info: Optional[ActionableInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
            All nested dataclasses are converted to dictionaries.
        """
        payload = asdict(self)
        payload["relevance_score"] = float(payload["relevance_score"])
        payload["exact_matches"] = int(payload.get("exact_matches", 0))
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create SearchResult from dictionary.

        Args:
            data: Dictionary containing SearchResult data

        Returns:
            SearchResult instance with properly typed nested objects
        """
        # Handle nested dataclasses
        if "context" in data and data["context"] is not None:
            data["context"] = SearchResultContext(**data["context"])
        if "actionable_info" in data and data["actionable_info"] is not None:
            data["actionable_info"] = ActionableInfo(**data["actionable_info"])
        return cls(**data)


class TechnicalDocumentationPreprocessor:
    """Preprocessor optimized for technical documentation."""

    def __init__(self):
        # Technical terms that should NOT be treated as stopwords
        self.technical_preserve = {
            "api",
            "cli",
            "gui",
            "ui",
            "ux",
            "db",
            "sql",
            "aws",
            "gcp",
            "cpu",
            "gpu",
            "http",
            "https",
            "tcp",
            "ip",
            "dns",
            "ssl",
            "tls",
            "jwt",
            "oauth",
            "rest",
            "json",
            "yaml",
            "xml",
            "html",
            "css",
            "js",
            "ts",
            "py",
            "go",
            "rust",
            "docker",
            "k8s",
            "kubernetes",
            "nginx",
            "redis",
            "postgres",
            "mysql",
            "react",
            "vue",
            "angular",
            "node",
            "npm",
            "pip",
            "git",
            "ci",
            "cd",
            "dev",
            "prod",
            "test",
            "debug",
            "log",
            "config",
            "env",
            "var",
        }

        # Common English stopwords minus technical terms
        self.stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "would",
            "you",
            "your",
            "this",
            "these",
            "those",
            "they",
            "them",
            "their",
            "there",
            "where",
            "when",
            "what",
            "who",
            "why",
            "how",
            "if",
            "or",
            "but",
            "not",
            "no",
            "yes",
            "can",
            "could",
            "should",
            "would",
            "may",
            "might",
            "must",
            "shall",
            "do",
            "does",
            "did",
            "have",
            "had",
            "been",
            "being",
            "been",
            "am",
            "i",
        }

        # Remove technical terms from stopwords
        self.stopwords = self.stopwords - self.technical_preserve

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for technical documentation search."""
        # Convert to lowercase
        text = text.lower()

        # Remove markdown syntax but preserve content
        text = re.sub(r"#+\s*", "", text)  # Remove markdown headers
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Remove bold
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Remove italic
        text = re.sub(r"`([^`]+)`", r"\1", text)  # Remove inline code backticks
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # Remove markdown links

        # Preserve technical terms and version numbers
        text = re.sub(
            r"\b(\w+[._-]\w+|\d+\.\d+\.\d+)\b",
            lambda m: m.group(1).replace(".", "_dot_").replace("-", "_dash_"),
            text,
        )

        # Tokenize preserving technical terms
        tokens = re.findall(r"\b\w+\b", text)

        # Filter stopwords but keep technical terms
        filtered_tokens = []
        for token in tokens:
            # Restore special characters
            token = token.replace("_dot_", ".").replace("_dash_", "-")

            if token not in self.stopwords or len(token) <= 2:
                if len(token) > 1:  # Keep tokens longer than 1 character
                    filtered_tokens.append(token)

        return " ".join(filtered_tokens)


class DocumentSearchEngine:
    """Coordinated search engine that delegates to specialized search strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, verbose: bool = True):
        """Initialize the search coordinator."""
        self.config = config or {}
        self.verbose = verbose
        self.documents = []
        self.index_path = None
        self.last_search_time = 0

        # Initialize strategies
        self.preprocessor = TechnicalDocumentationPreprocessor()
        self.cache_manager = None
        self.tfidf_strategy = None
        self.embedding_generator = None
        self.faiss_engine = None
        self.llm_enhancer = None
        self.pipeline = None
        self.telemetry_manager = None
        self.cost_engine = None

    def load_documents_from_index(self, index_path: Path) -> bool:
        """Load documents from the indexer's saved index."""
        try:
            with open(index_path / "index.yaml", "r") as f:
                index_data = yaml.safe_load(f)

            self.documents = index_data.get("documents", [])
            self.index_path = index_path

            # Initialize cache manager
            self.cache_manager = ThreadSafeCacheManager(index_path)

            # Load config from file and initialize embedding generator
            config_file = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
            config = {}
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f) or {}

            # Merge with existing config
            if self.config:
                config.update(self.config)

            self.config = config

            # Initialize telemetry manager and simplified cost tracking
            cache_dir = Path(config.get("cache_dir", ".cache/mimir"))
            if not cache_dir.is_absolute():
                cache_dir = get_project_root() / cache_dir
            self.telemetry_manager = TelemetryManager(config, cache_dir)
            self.cost_engine = LLMCostEngine(config)  # Now simplified tracker

            # Initialize search strategies
            self.tfidf_strategy = TFIDFSearchStrategy(
                self.preprocessor, self.cache_manager
            )
            self.embedding_generator = DocumentEmbeddingGenerator(
                config, self.cache_manager, self.verbose
            )
            self.faiss_engine = FAISSVectorSearchEngine(config, self.cache_manager)
            self.llm_enhancer = DocumentLLMEnhancer(
                config, self.telemetry_manager, self.cost_engine
            )

            # Initialize pipeline after all components are ready
            self._init_pipeline()

            if self.verbose:
                log_success(f"Loaded {len(self.documents)} documents from index", "📚")
            return True

        except Exception as e:
            if self.verbose:
                log_error(f"Error loading documents: {e}")
            return False

    def build_search_index(self) -> bool:
        """Build search indices using available strategies."""
        if not self.documents:
            if self.verbose:
                log_error("No documents loaded. Run indexer first.")
            return False

        if self.verbose:
            log_info("Building search index...", "🔍")
        start_time = time.time()

        # Build TF-IDF strategy index
        if self.tfidf_strategy:
            if not self.tfidf_strategy.build_index(self.documents):
                if self.verbose:
                    log_error("Failed to build TF-IDF index")
                return False

        # Save search index metadata
        self._save_search_index()

        elapsed = time.time() - start_time
        if self.verbose:
            log_success(f"Search index built in {elapsed:.2f}s")

        if self.tfidf_strategy and self.tfidf_strategy.vectorizer and self.verbose:
            log_info(
                f"Vocabulary size: {len(self.tfidf_strategy.vectorizer.vocabulary_)}",
                "📊",
            )

        return True

    def _save_search_index(self):
        """Save search index metadata to disk."""
        if not self.index_path:
            return

        search_index_file = self.index_path / "search_index.yaml"

        # Save metadata about the search index
        search_metadata = {
            "vocabulary_size": (
                len(self.tfidf_strategy.vectorizer.vocabulary_)
                if self.tfidf_strategy and self.tfidf_strategy.vectorizer
                else 0
            ),
            "document_count": len(self.documents),
            "last_built": time.time(),
        }

        with open(search_index_file, "w") as f:
            yaml.dump(search_metadata, f)

    def search(
        self, query: str, top_k: int = 10, mode: str = "smart"
    ) -> List[SearchResult]:
        """Search documents using the TF-IDF strategy."""
        if not self.tfidf_strategy or not self.tfidf_strategy.is_available():
            log_error("TF-IDF search not available. Run build_search_index() first.")
            return []

        # Use telemetry context manager if available
        if self.telemetry_manager and self.telemetry_manager.is_enabled():
            with self.telemetry_manager.track_query(query, mode, top_k) as query_id:  # noqa: F841
                # Delegate to TF-IDF strategy
                results = self.tfidf_strategy.search(query, top_k, mode)
                self.last_search_time = self.tfidf_strategy.last_search_time

                # Update telemetry with search metrics
                cache_hit_rate = getattr(
                    self.cache_manager, "get_hit_rate", lambda: 0.0
                )()
                self.telemetry_manager.update_search_metrics(
                    search_time_ms=self.last_search_time * 1000,
                    results=results,
                    cache_hit_rate=cache_hit_rate,
                )

                # Update system context
                self.telemetry_manager.update_system_context(
                    total_docs=len(self.documents),
                    index_version=getattr(self, "_index_version", None),
                )

                return results
        else:
            # Fallback to normal operation without telemetry
            results = self.tfidf_strategy.search(query, top_k, mode)
            self.last_search_time = self.tfidf_strategy.last_search_time
            return results

    def vector_search(
        self, query: str, top_k: int = 10, use_faiss: bool = True
    ) -> List[SearchResult]:
        """Search documents using vector similarity with optional FAISS acceleration."""
        if not self.embedding_generator:
            log_error("Embedding generator not initialized")
            return []

        # Use telemetry context manager if available
        if self.telemetry_manager and self.telemetry_manager.is_enabled():
            with self.telemetry_manager.track_query(query, "vector", top_k) as query_id:  # noqa: F841
                return self._do_vector_search(query, top_k, use_faiss)
        else:
            return self._do_vector_search(query, top_k, use_faiss)

    def _do_vector_search(
        self, query: str, top_k: int = 10, use_faiss: bool = True
    ) -> List[SearchResult]:
        """Internal vector search implementation."""
        start_time = time.time()

        # Encode query to embedding
        query_embedding = self.embedding_generator.encode_query(query)
        if query_embedding is None:
            log_error("Could not encode query")
            return []

        # Get all document embeddings from cache
        doc_embeddings = []
        for doc in self.documents:
            doc_path = doc["path"]
            model_name = self.embedding_generator.model_name
            cached_embeddings = self.cache_manager.get_cached_embeddings(
                doc_path, model_name
            )
            if cached_embeddings:
                for emb_data in cached_embeddings:
                    embedding_array = np.array(emb_data["embedding"])
                    from ..ai.embeddings import EmbeddingResult

                    result = EmbeddingResult(
                        document_path=emb_data["document_path"],
                        embedding=embedding_array,
                        model_name=emb_data["model_name"],
                        chunk_index=emb_data.get("chunk_index", 0),
                        chunk_text=emb_data.get("chunk_text", ""),
                    )
                    doc_embeddings.append(result)

        if not doc_embeddings:
            if self.verbose:
                log_error(
                    "No embeddings found. Run indexing with vector search enabled first."
                )
            return []

        # Try FAISS search first if enabled and available
        if (
            use_faiss
            and self.faiss_engine
            and self.config.get("vector_search", {}).get("faiss_enabled", True)
        ):
            # Build FAISS index if not already built
            if self.faiss_engine.faiss_index is None:
                if self.verbose:
                    log_info("Building FAISS index for fast vector search...", "🔨")
                if self.faiss_engine.build_faiss_index(doc_embeddings):
                    if self.verbose:
                        log_success("FAISS index ready")
                else:
                    if self.verbose:
                        log_warning(
                            "FAISS index build failed, falling back to scikit-learn"
                        )
                    use_faiss = False

            if use_faiss and self.faiss_engine.faiss_index is not None:
                # Use FAISS for fast search
                faiss_results = self.faiss_engine.search(query_embedding, top_k)

                # Convert FAISS results to SearchResult objects
                results = []
                seen_docs = set()

                for faiss_result in faiss_results:
                    # Avoid duplicate documents (take highest scoring chunk)
                    if faiss_result.document_path in seen_docs:
                        continue
                    seen_docs.add(faiss_result.document_path)

                    # Find the corresponding document
                    doc = None
                    for d in self.documents:
                        if d["path"] == faiss_result.document_path:
                            doc = d
                            break

                    if doc:
                        # Use chunk text as snippet if available
                        snippet = (
                            faiss_result.content_snippet
                            if faiss_result.content_snippet
                            else self._generate_snippet(doc["content"], query)
                        )

                        result = SearchResult(
                            document_path=doc["path"],
                            title=doc["title"],
                            content_snippet=snippet,
                            relevance_score=faiss_result.similarity_score,
                            exact_matches=0,  # Vector search doesn't use exact matches
                            metadata={
                                **doc.get("metadata", {}),
                                "search_type": "vector_faiss",
                                "chunk_index": faiss_result.chunk_index,
                                "faiss_distance": faiss_result.distance,
                            },
                        )
                        results.append(result)

                self.last_search_time = time.time() - start_time
                return results

        # Fallback to scikit-learn similarity search
        if self.verbose:
            log_info("Using scikit-learn for vector similarity search", "🔄")
        similarity_results = self.embedding_generator.compute_similarity(
            query_embedding, doc_embeddings
        )

        # Convert to SearchResult objects
        results = []
        seen_docs = set()

        for emb_result, similarity in similarity_results[:top_k]:
            # Avoid duplicate documents (take highest scoring chunk)
            if emb_result.document_path in seen_docs:
                continue
            seen_docs.add(emb_result.document_path)

            # Find the corresponding document
            doc = None
            for d in self.documents:
                if d["path"] == emb_result.document_path:
                    doc = d
                    break

            if doc:
                # Use chunk text as snippet if available
                snippet = (
                    emb_result.chunk_text
                    if emb_result.chunk_text
                    else self._generate_snippet(doc["content"], query)
                )

                result = SearchResult(
                    document_path=doc["path"],
                    title=doc["title"],
                    content_snippet=snippet,
                    relevance_score=float(similarity),
                    exact_matches=0,  # Vector search doesn't use exact matches
                    metadata={
                        **doc.get("metadata", {}),
                        "search_type": "vector_sklearn",
                        "chunk_index": emb_result.chunk_index,
                    },
                )
                results.append(result)

        self.last_search_time = time.time() - start_time

        # Update telemetry if enabled
        if self.telemetry_manager and self.telemetry_manager.is_enabled():
            cache_hit_rate = getattr(self.cache_manager, "get_hit_rate", lambda: 0.0)()
            self.telemetry_manager.update_search_metrics(
                search_time_ms=self.last_search_time * 1000,
                results=results,
                cache_hit_rate=cache_hit_rate,
            )

            # Update system context
            self.telemetry_manager.update_system_context(
                total_docs=len(self.documents),
                index_version=getattr(self, "_index_version", None),
            )

        return results

    def hybrid_search(
        self, query: str, top_k: int = 10, vector_weight: float = 0.6
    ) -> List[SearchResult]:
        """Combine TF-IDF and vector search for best results."""
        if not self.embedding_generator:
            # Fall back to TF-IDF only
            return self.search(query, top_k, mode="smart")

        # Use telemetry context manager if available
        if self.telemetry_manager and self.telemetry_manager.is_enabled():
            with self.telemetry_manager.track_query(query, "hybrid", top_k) as query_id:  # noqa: F841
                return self._do_hybrid_search(query, top_k, vector_weight)
        else:
            return self._do_hybrid_search(query, top_k, vector_weight)

    def _do_hybrid_search(
        self, query: str, top_k: int = 10, vector_weight: float = 0.6
    ) -> List[SearchResult]:
        """Internal hybrid search implementation."""
        start_time = time.time()

        # Get results from both methods (these will track their own telemetry if telemetry is disabled here)
        # We temporarily disable telemetry to avoid double-tracking
        original_enabled = None
        if self.telemetry_manager:
            original_enabled = self.telemetry_manager.enabled
            self.telemetry_manager.enabled = False

        try:
            tfidf_results = self.search(query, top_k, mode="smart")
            vector_results = self.vector_search(query, top_k)
        finally:
            # Restore telemetry state
            if self.telemetry_manager and original_enabled is not None:
                self.telemetry_manager.enabled = original_enabled

        # Combine and rerank results
        combined_scores = {}

        # Add TF-IDF scores
        for result in tfidf_results:
            path = result.document_path
            combined_scores[path] = {
                "tfidf_score": result.relevance_score,
                "vector_score": 0.0,
                "result": result,
            }

        # Add vector scores
        for result in vector_results:
            path = result.document_path
            if path in combined_scores:
                combined_scores[path]["vector_score"] = result.relevance_score
            else:
                combined_scores[path] = {
                    "tfidf_score": 0.0,
                    "vector_score": result.relevance_score,
                    "result": result,
                }

        # Compute hybrid scores and rank
        final_results = []
        for path, scores in combined_scores.items():
            # Weighted combination of scores
            hybrid_score = (1 - vector_weight) * scores[
                "tfidf_score"
            ] + vector_weight * scores["vector_score"]

            # Use the result object and update score
            result = scores["result"]
            result.relevance_score = hybrid_score
            result.metadata = {**result.metadata, "search_type": "hybrid"}

            final_results.append(result)

        # Sort by hybrid score
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        final_results = final_results[:top_k]

        # Update search timing
        self.last_search_time = time.time() - start_time

        # Update telemetry if enabled
        if self.telemetry_manager and self.telemetry_manager.is_enabled():
            cache_hit_rate = getattr(self.cache_manager, "get_hit_rate", lambda: 0.0)()
            self.telemetry_manager.update_search_metrics(
                search_time_ms=self.last_search_time * 1000,
                results=final_results,
                cache_hit_rate=cache_hit_rate,
            )

            # Update system context
            self.telemetry_manager.update_system_context(
                total_docs=len(self.documents),
                index_version=getattr(self, "_index_version", None),
            )

        return final_results

    def llm_enhanced_search(
        self,
        query: str,
        top_k: int = 10,
        enhance_query: bool = True,
        enhance_results: bool = True,
    ) -> List[SearchResult]:
        """Search with LLM enhancements for query expansion and result analysis."""
        if not self.llm_enhancer or not self.llm_enhancer.is_available():
            # Fall back to hybrid search if LLM not available
            return self.hybrid_search(query, top_k)

        original_query = query

        # Enhance query if requested
        if enhance_query:
            enhanced_query = self.llm_enhancer.client.enhance_search_query(query)
            if enhanced_query and enhanced_query != query:
                query = enhanced_query
                if self.verbose:
                    log_info(f"Enhanced query: {query}", "🔍")

        # Perform hybrid search with enhanced query
        results = self.hybrid_search(query, top_k)

        # Enhance results if requested and we have results
        if enhance_results and results:
            try:
                # Convert SearchResult objects to dicts for LLM processing
                result_dicts = []
                for result in results:
                    result_dict = {
                        "title": result.title,
                        "content": result.content_snippet,
                        "path": result.document_path,
                        "score": result.relevance_score,
                    }
                    result_dicts.append(result_dict)

                enhanced_results = self.llm_enhancer.enhance_search_results(
                    original_query, result_dicts
                )

                # Merge LLM enhancements back into SearchResult objects
                for i, result in enumerate(results):
                    if i < len(enhanced_results):
                        enhanced = enhanced_results[i]
                        if "llm_summary" in enhanced:
                            result.metadata["llm_summary"] = enhanced["llm_summary"]
                        if "llm_analysis" in enhanced:
                            result.metadata["llm_analysis"] = enhanced["llm_analysis"]
                        result.metadata["search_type"] = "llm_enhanced"

            except Exception as e:
                log_warning(f"LLM enhancement failed: {e}")

        return results

    def ask_question(self, question: str, context_limit: int = 5) -> Optional[str]:
        """Ask a question and get an answer based on document context."""
        if not self.llm_enhancer or not self.llm_enhancer.is_available():
            return None

        # First, search for relevant documents
        search_results = self.hybrid_search(question, top_k=context_limit)

        if not search_results:
            return "I couldn't find any relevant documents to answer your question."

        # Convert to format expected by LLM
        context_docs = []
        for result in search_results:
            # Find the full document content
            full_doc = None
            for doc in self.documents:
                if doc["path"] == result.document_path:
                    full_doc = doc
                    break

            if full_doc:
                context_docs.append(
                    {
                        "title": result.title,
                        "content": full_doc["content"][
                            :2000
                        ],  # Use more content for Q&A
                        "content_snippet": result.content_snippet,
                        "path": result.document_path,
                    }
                )

        # Get LLM answer
        answer = self.llm_enhancer.interactive_qa(question, context_docs)
        return answer

    def _init_pipeline(self):
        """Initialize the search pipeline orchestrator."""
        try:
            from ..ai.pipeline import SearchPipelineOrchestrator

            self.pipeline = SearchPipelineOrchestrator(self, self.config)
        except ImportError as e:
            log_warning(f"Pipeline not available: {e}")
            self.pipeline = None

    def pipeline_search(
        self, query: str, mode: str = "smart", limit: int = 10, cost_limit: float = 0.05
    ):
        """Search using the multi-stage pipeline."""
        if not self.pipeline:
            log_warning("Pipeline not available, falling back to hybrid search")
            return self.hybrid_search(query, limit)

        try:
            pipeline_result = self.pipeline.search(query, mode, limit, cost_limit)

            # Update search time tracking
            self.last_search_time = pipeline_result.execution_time

            return pipeline_result.results

        except Exception as e:
            log_warning(f"Pipeline search failed: {e}")
            # Fallback to hybrid search
            return self.hybrid_search(query, limit)

    def get_pipeline_info(self, mode: str = None):
        """Get information about available pipeline modes."""
        if not self.pipeline:
            return {"error": "Pipeline not available"}

        return self.pipeline.get_mode_info(mode)

    def estimate_search_cost(self, query: str, mode: str = "smart"):
        """Estimate cost for a search query."""
        if not self.pipeline:
            return {"error": "Pipeline not available", "cost": 0.0}

        return self.pipeline.estimate_cost(query, mode)

    def _generate_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """Generate a relevant snippet from document content."""
        query_words = query.lower().split()
        content_lower = content.lower()

        # Find the best position to start the snippet
        best_pos = 0
        best_score = 0

        # Look for query words in content
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1:
                # Count nearby query words
                snippet_start = max(0, pos - 100)
                snippet_end = min(len(content), pos + 100)
                snippet = content_lower[snippet_start:snippet_end]

                score = sum(1 for w in query_words if w in snippet)
                if score > best_score:
                    best_score = score
                    best_pos = snippet_start

        # Extract snippet
        snippet_end = min(len(content), best_pos + max_length)
        snippet = content[best_pos:snippet_end]

        # Clean up snippet
        snippet = snippet.strip()
        if best_pos > 0:
            snippet = "..." + snippet
        if snippet_end < len(content):
            snippet = snippet + "..."

        return snippet

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            "documents_indexed": len(self.documents),
            "vocabulary_size": (
                len(self.tfidf_strategy.vectorizer.vocabulary_)
                if self.tfidf_strategy and self.tfidf_strategy.vectorizer
                else 0
            ),
            "inverted_index_terms": (
                len(self.tfidf_strategy.inverted_index) if self.tfidf_strategy else 0
            ),
            "last_search_time_ms": round(self.last_search_time * 1000, 2),
            "index_built": self.tfidf_strategy
            and self.tfidf_strategy.vectorizer is not None,
        }
