#!/usr/bin/env python3
"""
LLM-Optimized Output Formatter for mimir

Generates structured JSON output optimized for Claude Code and other LLM consumption,
including rich context, actionable info, search timing, suggestions, and related docs.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import jsonschema
import numpy as np

# Handle imports for both direct execution and package imports
try:
    from ..core.search import SearchResult
except ImportError:
    # For direct execution or testing
    import sys
    from pathlib import Path

    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))
    from core.search import SearchResult


class LLMOutputFormatter:
    """
    Formats search results into structured JSON optimized for LLM consumption.

    Features:
    - Rich context with enhanced SearchResult fields
    - Search timing and performance metrics
    - Suggestions and related documents
    - Schema validation using jsonschema v4.20+
    - Backwards compatibility with existing formats
    """

    # JSON Schema for LLM output format
    LLM_OUTPUT_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "search_metadata": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "search_time_ms": {"type": "number", "minimum": 0},
                    "total_results": {"type": "integer", "minimum": 0},
                    "search_mode": {"type": "string"},
                    "format_version": {"type": "string"},
                },
                "required": [
                    "query",
                    "timestamp",
                    "search_time_ms",
                    "total_results",
                    "search_mode",
                    "format_version",
                ],
            },
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "document_path": {"type": "string"},
                        "title": {"type": "string"},
                        "content_snippet": {"type": "string"},
                        "relevance_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "exact_matches": {"type": "integer", "minimum": 0},
                        "metadata": {"type": "object"},
                        "context": {
                            "type": ["object", "null"],
                            "properties": {
                                "section_headers": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "line_numbers": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                "surrounding_context": {"type": "object"},
                                "document_metadata": {"type": "object"},
                                "matched_sections": {"type": "array"},
                                "related_files": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        "actionable_info": {
                            "type": ["object", "null"],
                            "properties": {
                                "related_commands": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "prerequisites": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "confidence_score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "suggested_actions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "workflow_steps": {"type": "array"},
                                "example_usage": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": [
                        "document_path",
                        "title",
                        "content_snippet",
                        "relevance_score",
                        "exact_matches",
                        "metadata",
                    ],
                },
            },
            "suggestions": {
                "type": "object",
                "properties": {
                    "related_queries": {"type": "array", "items": {"type": "string"}},
                    "improvement_tips": {"type": "array", "items": {"type": "string"}},
                    "context_hints": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["related_queries", "improvement_tips", "context_hints"],
            },
            "related_documents": {
                "type": "object",
                "properties": {
                    "frequently_accessed": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "by_topic": {"type": "object"},
                    "recently_updated": {"type": "array", "items": {"type": "string"}},
                    "cross_referenced": {"type": "array", "items": {"type": "string"}},
                    "by_confidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "frequently_accessed",
                    "by_topic",
                    "recently_updated",
                    "cross_referenced",
                    "by_confidence",
                ],
            },
        },
        "required": ["search_metadata", "results", "suggestions", "related_documents"],
    }

    def __init__(self, validate_schema: bool = True):
        """
        Initialize the LLM formatter.

        Args:
            validate_schema: Whether to validate output against JSON schema
        """
        self.validate_schema = validate_schema
        self.format_version = "1.1.0"

    def _convert_numpy_types(self, obj: Any) -> Any:
        """
        Convert numpy types to native Python types for JSON serialization.

        Args:
            obj: Object that may contain numpy types

        Returns:
            Object with numpy types converted to native Python types
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj

    def format_search_results(
        self,
        results: List[SearchResult],
        query: str,
        search_time_seconds: float,
        search_mode: str = "smart",
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format search results into LLM-optimized JSON.

        Args:
            results: List of SearchResult objects
            query: Original search query
            search_time_seconds: Time taken for search in seconds
            search_mode: Search algorithm used
            additional_context: Optional additional context for suggestions

        Returns:
            JSON string formatted for LLM consumption

        Raises:
            jsonschema.ValidationError: If schema validation fails
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        search_time_ms = search_time_seconds * 1000

        # Build the output structure
        output = {
            "search_metadata": {
                "query": query,
                "timestamp": timestamp,
                "search_time_ms": search_time_ms,
                "total_results": len(results),
                "search_mode": search_mode,
                "format_version": self.format_version,
            },
            "results": self._format_results(results),
            "suggestions": self._generate_suggestions(
                query, results, additional_context
            ),
            "related_documents": self._generate_related_documents(
                results, additional_context
            ),
        }

        # Convert numpy types to native Python types
        output = self._convert_numpy_types(output)

        # Validate schema if enabled
        if self.validate_schema:
            try:
                jsonschema.validate(output, self.LLM_OUTPUT_SCHEMA)
            except jsonschema.ValidationError as e:
                raise jsonschema.ValidationError(
                    f"LLM output schema validation failed: {e.message}"
                )

        return json.dumps(output, indent=2, ensure_ascii=False)

    def _format_results(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """
        Format SearchResult objects into JSON-serializable dictionaries.

        Args:
            results: List of SearchResult objects

        Returns:
            List of dictionaries with enhanced SearchResult data
        """
        formatted_results = []

        for result in results:
            # Convert SearchResult to dict using its built-in method
            result_dict = result.to_dict()

            # Ensure all required fields are present with defaults
            if "context" not in result_dict:
                result_dict["context"] = None
            if "actionable_info" not in result_dict:
                result_dict["actionable_info"] = None

            # Add LLM-specific enhancements
            if "search_type" not in result_dict.get("metadata", {}):
                result_dict["metadata"]["search_type"] = "enhanced"

            formatted_results.append(result_dict)

        return formatted_results

    def _generate_suggestions(
        self,
        query: str,
        results: List[SearchResult],
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """
        Generate search suggestions based on query and results.

        Args:
            query: Original search query
            results: Search results for context
            additional_context: Optional additional context

        Returns:
            Dictionary with suggestion categories
        """
        suggestions = {
            "related_queries": [],
            "improvement_tips": [],
            "context_hints": [],
        }

        # Generate related queries based on common patterns
        query_words = query.lower().split()

        # Technical documentation patterns
        if any(
            word in query_words
            for word in ["api", "endpoint", "authentication", "auth"]
        ):
            suggestions["related_queries"].extend(
                [
                    "API authentication examples",
                    "endpoint configuration",
                    "authentication troubleshooting",
                ]
            )

        if any(
            word in query_words for word in ["deploy", "deployment", "install", "setup"]
        ):
            suggestions["related_queries"].extend(
                [
                    "deployment troubleshooting",
                    "installation requirements",
                    "setup verification",
                ]
            )

        if any(
            word in query_words for word in ["test", "testing", "unit", "integration"]
        ):
            suggestions["related_queries"].extend(
                [
                    "testing best practices",
                    "test configuration",
                    "debugging test failures",
                ]
            )

        # Generate improvement tips based on results
        if len(results) == 0:
            suggestions["improvement_tips"].extend(
                [
                    "Try broader search terms",
                    "Check for typos in your query",
                    "Use alternative keywords or synonyms",
                ]
            )
        elif len(results) > 20:
            suggestions["improvement_tips"].extend(
                [
                    "Use more specific search terms",
                    "Add technical keywords to narrow results",
                    "Use quotes for exact phrase matching",
                ]
            )
        else:
            suggestions["improvement_tips"].extend(
                [
                    "Results look relevant - consider related queries above",
                    "Check document context for additional details",
                    "Use actionable info for next steps",
                ]
            )

        # Generate enhanced context hints using SearchResult enhanced fields
        doc_types = set()
        actionable_docs = 0
        command_count = 0
        workflow_steps_count = 0

        for result in results[:5]:  # Analyze top 5 results
            # Use enhanced metadata if available
            if result.context and result.context.document_metadata:
                doc_metadata = result.context.document_metadata
                if "doc_type" in doc_metadata:
                    doc_types.add(doc_metadata["doc_type"])

                # Enhanced type detection from context
                file_type = doc_metadata.get("file_type", "unknown")
                if file_type == "markdown":
                    doc_types.add("Structured documentation")

            # Count actionable information
            if result.actionable_info:
                actionable_docs += 1
                command_count += len(result.actionable_info.related_commands)
                workflow_steps_count += len(result.actionable_info.workflow_steps)

            # Legacy metadata fallback
            metadata = result.metadata
            if "doc_type" in metadata:
                doc_types.add(metadata["doc_type"])

            # Infer document type from path
            path = result.document_path.lower()
            if "api" in path:
                doc_types.add("API documentation")
            elif "guide" in path or "tutorial" in path:
                doc_types.add("Tutorial/Guide")
            elif "setup" in path or "install" in path:
                doc_types.add("Setup instructions")
            elif "troubleshoot" in path or "debug" in path:
                doc_types.add("Troubleshooting guide")
            elif "workflow" in path:
                doc_types.add("Workflow documentation")

        # Enhanced context hints with actionable information
        if doc_types:
            suggestions["context_hints"].append(f"Found {', '.join(doc_types)} content")

        if actionable_docs > 0:
            suggestions["context_hints"].append(
                f"{actionable_docs} documents have actionable information"
            )

        if command_count > 0:
            suggestions["context_hints"].append(
                f"{command_count} executable commands found"
            )

        if workflow_steps_count > 0:
            suggestions["context_hints"].append(
                f"{workflow_steps_count} workflow steps identified"
            )

        if any("workflow" in result.document_path.lower() for result in results):
            suggestions["context_hints"].append("Contains workflow documentation")

        if any(result.exact_matches > 0 for result in results):
            suggestions["context_hints"].append("Exact keyword matches found")

        # Add confidence-based hints
        high_confidence_docs = sum(
            1
            for r in results
            if r.actionable_info and r.actionable_info.confidence_score > 0.7
        )
        if high_confidence_docs > 0:
            suggestions["context_hints"].append(
                f"{high_confidence_docs} high-confidence actionable documents"
            )

        return suggestions

    def _generate_related_documents(
        self,
        results: List[SearchResult],
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate enhanced related document suggestions using SearchResult context.

        Args:
            results: Search results with enhanced context and actionable info
            additional_context: Optional additional context

        Returns:
            Dictionary with enhanced related document categories
        """
        related_docs = {
            "frequently_accessed": [],
            "by_topic": {},
            "recently_updated": [],
            "cross_referenced": [],  # New: docs that reference each other
            "by_confidence": [],  # New: docs ordered by actionable info confidence
        }

        # Extract paths and categorize by directory
        paths_by_dir = {}
        all_paths = []
        cross_references = set()
        confidence_docs = []

        for result in results:
            path = result.document_path
            all_paths.append(path)

            # Group by directory for topic organization
            dir_path = "/".join(path.split("/")[:-1]) if "/" in path else "root"
            if dir_path not in paths_by_dir:
                paths_by_dir[dir_path] = []
            paths_by_dir[dir_path].append(path)

            # Extract cross-referenced documents from context
            if result.context and result.context.related_files:
                for related_file in result.context.related_files:
                    # Convert relative paths to full paths if needed
                    if not related_file.startswith("/") and not related_file.startswith(
                        "docs/"
                    ):
                        related_file = f"docs/{related_file}"
                    cross_references.add(related_file)

            # Collect documents with actionable info for confidence ranking
            if result.actionable_info and result.actionable_info.confidence_score > 0.3:
                confidence_docs.append(
                    {
                        "path": path,
                        "confidence": result.actionable_info.confidence_score,
                        "commands_count": len(result.actionable_info.related_commands),
                        "actions_count": len(result.actionable_info.suggested_actions),
                    }
                )

        # Generate frequently accessed (top scoring results)
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        related_docs["frequently_accessed"] = [
            result.document_path for result in sorted_results[:5]
        ]

        # Enhanced by_topic with metadata from context
        for dir_path, paths in paths_by_dir.items():
            if len(paths) >= 1:  # Include single docs too, with metadata
                topic_name = dir_path.replace("/", " ").replace("-", " ").title()

                # Add metadata from enhanced context
                topic_info = {
                    "paths": paths[:3],
                    "doc_count": len(paths),
                    "avg_relevance": sum(
                        r.relevance_score
                        for r in sorted_results
                        if r.document_path in paths
                    )
                    / len(paths),
                }

                # Add common document types found in this topic
                doc_types = set()
                for result in results:
                    if result.document_path in paths and result.context:
                        doc_metadata = result.context.document_metadata
                        if "doc_type" in doc_metadata:
                            doc_types.add(doc_metadata["doc_type"])
                        # Infer type from path
                        path_lower = result.document_path.lower()
                        if "api" in path_lower:
                            doc_types.add("API")
                        elif "workflow" in path_lower:
                            doc_types.add("Workflow")
                        elif "setup" in path_lower or "install" in path_lower:
                            doc_types.add("Setup")

                if doc_types:
                    topic_info["document_types"] = list(doc_types)

                related_docs["by_topic"][topic_name] = topic_info

        # Enhanced recently_updated using context metadata
        recently_updated = []
        recent_indicators = ["2024", "2025", "v1", "v2", "latest", "current"]

        for result in results:
            path = result.document_path
            is_recent = False

            # Check path-based indicators
            if any(indicator in path.lower() for indicator in recent_indicators):
                is_recent = True

            # Check context metadata for recency indicators
            if result.context and result.context.document_metadata:
                metadata = result.context.document_metadata
                content_length = metadata.get("content_length", 0)
                total_lines = metadata.get("total_lines", 0)

                # Heuristic: larger docs are more likely to be recently updated
                if content_length > 5000 or total_lines > 200:
                    is_recent = True

            if is_recent:
                recently_updated.append(path)

        related_docs["recently_updated"] = recently_updated[:5]

        # New: Cross-referenced documents
        related_docs["cross_referenced"] = list(cross_references)[:5]

        # New: Documents by confidence (high actionable info confidence)
        confidence_sorted = sorted(
            confidence_docs, key=lambda x: x["confidence"], reverse=True
        )
        related_docs["by_confidence"] = [doc["path"] for doc in confidence_sorted[:3]]

        return related_docs

    def validate_output(self, output_json: str) -> bool:
        """
        Validate LLM output against schema.

        Args:
            output_json: JSON string to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            output_data = json.loads(output_json)
            jsonschema.validate(output_data, self.LLM_OUTPUT_SCHEMA)
            return True
        except (json.JSONDecodeError, jsonschema.ValidationError):
            return False

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema used for validation.

        Returns:
            JSON schema dictionary
        """
        return self.LLM_OUTPUT_SCHEMA.copy()
