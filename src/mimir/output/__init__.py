"""
Output formatters for mimir search results.

This package provides various output formatters for search results,
including LLM-optimized JSON formatting for Claude Code integration.
"""

from .llm_formatter import LLMOutputFormatter

__all__ = ["LLMOutputFormatter"]
