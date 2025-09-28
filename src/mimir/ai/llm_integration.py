"""
LLM Integration - OpenRouter API client for document analysis and enhancement

Provides LLM capabilities for:
- Document summarization
- Content analysis and insights
- Query enhancement and rewriting
- Advanced document question answering
"""

from typing import Dict, Any, List, Optional
import time
import json
from dataclasses import dataclass
from contextlib import contextmanager

try:
    from llm_cli_core import track_ai_call, OpenRouterTokens
except ImportError:  # pragma: no cover - fallback when telemetry extras missing

    @contextmanager
    def track_ai_call(*_args, **_kwargs):
        class _Tracker:
            def __init__(self):
                self.start_time = time.time()

            def record_tokens(self, *_a, **_kw):
                return None

            def record_cost(self, *_a, **_kw):
                return None

            def record_model(self, *_a, **_kw):
                return None

        tracker = _Tracker()
        try:
            yield tracker
        finally:
            pass

    class OpenRouterTokens:  # type: ignore
        def __init__(self, *_args, **_kwargs):
            self.input_tokens = 0
            self.output_tokens = 0


@dataclass
class LLMResponse:
    """Response from LLM API call."""

    content: str
    model: str
    usage: Dict[str, Any]
    response_time: float
    metadata: Dict[str, Any]


class OpenRouterClient:
    """Client for OpenRouter API integration."""

    def __init__(
        self, config: Dict[str, Any], telemetry_manager=None, cost_engine=None
    ):
        """Initialize OpenRouter client with configuration."""
        self.config = config.get("llm", {})
        self.api_key = self._get_api_key()
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = self.config.get("model", "meta-llama/llama-3.1-8b-instruct:free")
        self.max_tokens = self.config.get("max_tokens", 1000)
        self.temperature = self.config.get("temperature", 0.1)
        self.enabled = self.config.get("enabled", False) and bool(self.api_key)
        self.telemetry_manager = telemetry_manager
        self.cost_engine = cost_engine

        # Performance tracking
        self.last_response_time = 0
        self.total_requests = 0
        self.total_tokens_used = 0

    def _get_api_key(self) -> Optional[str]:
        """Get OpenRouter API key from environment or config."""
        import os

        # Try environment variable first
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            return api_key

        # Try config file
        return self.config.get("api_key")

    def is_available(self) -> bool:
        """Check if LLM integration is available and configured."""
        return self.enabled and bool(self.api_key)

    def _make_request(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Optional[LLMResponse]:
        """Make request to OpenRouter API."""
        if not self.is_available():
            return None

        try:
            # Try importing httpx (optional dependency)
            try:
                import httpx
            except ImportError:
                print("⚠️  httpx not installed. Install with: pip install httpx")
                return None

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mimir.local",
                "X-Title": "Mímir Tool",
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False,
            }

            # Enhanced telemetry with session correlation
            with track_ai_call("mimir", "document_analysis") as tracker:
                # Make request
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()

                # Record telemetry with automatic token extraction
                tracker.record_tokens(OpenRouterTokens(data))

                # Calculate cost (keeping existing logic)
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)
                estimated_cost = 0.0
                if self.cost_engine:
                    try:
                        metrics = self.cost_engine.track_usage(self.model, usage)
                        estimated_cost = metrics.estimated_cost_usd if metrics else 0.0
                    except Exception as e:
                        print(f"⚠️  Cost tracking failed: {e}")
                        estimated_cost = tokens_used * 0.000002
                else:
                    estimated_cost = tokens_used * 0.000002

                tracker.record_cost(estimated_cost)
                tracker.record_model(self.model)

            # Update internal counters
            response_time = time.time() - tracker.start_time
            self.last_response_time = response_time
            self.total_requests += 1
            if tokens_used:
                self.total_tokens_used += tokens_used

            # Keep existing telemetry manager for backward compatibility
            if self.telemetry_manager and self.telemetry_manager.is_enabled():
                self.telemetry_manager.update_llm_metrics(
                    tokens_used=tokens_used,
                    cost_usd=estimated_cost,
                    model=self.model,
                    llm_time_ms=response_time * 1000,
                )

            # Extract response
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                response_time=response_time,
                metadata={
                    "request_id": data.get("id"),
                    "created": data.get("created"),
                    "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                    "estimated_cost_usd": estimated_cost,
                },
            )

        except Exception as e:
            print(f"❌ LLM request failed: {e}")
            return None

    def summarize_document(
        self, title: str, content: str, max_length: int = 200
    ) -> Optional[str]:
        """Generate a concise summary of a document."""
        if not self.is_available():
            return None

        system_prompt = f"""You are a technical documentation expert. Provide concise, accurate summaries of documentation.

Guidelines:
- Focus on key concepts, features, and important information
- Use clear, technical language appropriate for developers
- Keep summary under {max_length} characters
- Highlight actionable information and key points
- Preserve important technical details and terminology"""

        user_prompt = f"""Summarize this technical document:

Title: {title}

Content:
{content[:2000]}{'...' if len(content) > 2000 else ''}

Provide a {max_length}-character summary focusing on the key technical concepts and actionable information."""

        response = self._make_request(user_prompt, system_prompt)
        return response.content if response else None

    def analyze_document_content(
        self, title: str, content: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze document content and extract insights."""
        if not self.is_available():
            return None

        system_prompt = """You are a technical documentation analyst. Analyze documents and extract structured insights.

Provide your analysis as JSON with these fields:
- "topics": List of main topics/concepts covered
- "difficulty": "beginner", "intermediate", or "advanced"
- "document_type": Type of document (guide, reference, tutorial, etc.)
- "key_concepts": List of important technical concepts
- "prerequisites": What knowledge is assumed
- "actionable_items": List of things a reader can do/implement
- "related_keywords": Keywords for improved searchability

Return only valid JSON, no other text."""

        user_prompt = f"""Analyze this technical document:

Title: {title}

Content:
{content[:1500]}{'...' if len(content) > 1500 else ''}

Provide structured analysis as JSON."""

        response = self._make_request(user_prompt, system_prompt)
        if not response:
            return None

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            print("⚠️  LLM returned invalid JSON for document analysis")
            return None

    def enhance_search_query(
        self, original_query: str, context: Optional[str] = None
    ) -> Optional[str]:
        """Enhance a search query with synonyms and related terms."""
        if not self.is_available():
            return None

        system_prompt = """You are a search optimization expert for technical documentation.

Your task is to enhance search queries by:
- Adding relevant synonyms and alternative terms
- Including related technical concepts
- Expanding abbreviations and acronyms
- Adding context-appropriate keywords
- Keeping the enhanced query concise and focused

Return only the enhanced search query, no explanations."""

        context_info = f"\n\nContext: {context}" if context else ""

        user_prompt = f"""Enhance this search query for better technical documentation search results:

Original query: "{original_query}"{context_info}

Provide an enhanced version with relevant synonyms, related terms, and expanded concepts."""

        response = self._make_request(user_prompt, system_prompt)
        return response.content.strip().strip('"') if response else None

    def answer_question_with_context(
        self, question: str, documents: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Answer a question using provided document context."""
        if not self.is_available() or not documents:
            return None

        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents[:3], 1):  # Limit to top 3 documents
            title = doc.get("title", "Untitled")
            content = doc.get("content_snippet", doc.get("content", ""))[:500]
            context_parts.append(f"Document {i}: {title}\n{content}")

        context = "\n\n".join(context_parts)

        system_prompt = """You are a helpful technical documentation assistant. Answer questions based on the provided document context.

Guidelines:
- Only use information from the provided documents
- Be accurate and cite specific documents when possible
- If the answer isn't in the context, say so clearly
- Provide practical, actionable information when available
- Use appropriate technical language for the audience"""

        user_prompt = f"""Based on the following documentation, answer this question:

Question: {question}

Documentation Context:
{context}

Provide a helpful answer based on the documentation provided."""

        response = self._make_request(user_prompt, system_prompt)
        return response.content if response else None

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for the LLM client."""
        return {
            "enabled": self.enabled,
            "available": self.is_available(),
            "model": self.model,
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "last_response_time_ms": round(self.last_response_time * 1000, 2),
            "avg_response_time_ms": (
                round((self.last_response_time * 1000) / max(1, self.total_requests), 2)
                if self.total_requests > 0
                else 0
            ),
        }


class DocumentLLMEnhancer:
    """High-level interface for LLM-enhanced document operations."""

    def __init__(
        self, config: Dict[str, Any], telemetry_manager=None, cost_engine=None
    ):
        """Initialize the document LLM enhancer."""
        self.config = config
        self.client = OpenRouterClient(config, telemetry_manager, cost_engine)
        self.cache = {}  # Simple in-memory cache for LLM responses

    def is_available(self) -> bool:
        """Check if LLM enhancement is available."""
        return self.client.is_available()

    def enhance_search_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance search results with LLM-generated insights."""
        if not self.is_available() or not results:
            return results

        enhanced_results = []
        for result in results:
            enhanced_result = result.copy()

            # Generate summary if content is long
            content = result.get("content", "")
            if len(content) > 500:
                summary = self.client.summarize_document(
                    result.get("title", ""), content, max_length=150
                )
                if summary:
                    enhanced_result["llm_summary"] = summary

            # Analyze document for additional metadata
            analysis = self.client.analyze_document_content(
                result.get("title", ""), content[:1000]
            )
            if analysis:
                enhanced_result["llm_analysis"] = analysis

            enhanced_results.append(enhanced_result)

        return enhanced_results

    def interactive_qa(
        self, question: str, search_results: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Provide interactive Q&A using search results as context."""
        if not self.is_available():
            return None

        return self.client.answer_question_with_context(question, search_results)

    def suggest_related_queries(self, original_query: str) -> List[str]:
        """Suggest related queries based on the original query."""
        if not self.is_available():
            return []

        enhanced_query = self.client.enhance_search_query(original_query)
        if enhanced_query and enhanced_query != original_query:
            # Simple parsing to extract different query variations
            # This could be enhanced with more sophisticated parsing
            variations = [enhanced_query]

            # Add the original query if it's different
            if original_query.lower() not in enhanced_query.lower():
                variations.append(original_query)

            return variations[:3]  # Limit to 3 suggestions

        return []

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get statistics about LLM enhancements."""
        return self.client.get_stats()
