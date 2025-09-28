"""
Simple Model Tracking Engine - Basic model and token tracking without complex cost calculations

Provides simple tracking for:
- Model names and providers
- Token counts (prompt/completion/total)
- Basic timing metrics
- Simple cost estimation (optional)
"""

from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BasicModelInfo:
    """Simple model information for tracking."""

    model_id: str
    provider: str
    is_free: bool = False

    @classmethod
    def from_model_id(cls, model_id: str) -> "BasicModelInfo":
        """Extract provider and free status from model ID."""
        # Simple heuristics to determine provider and free status
        if ":free" in model_id:
            provider = model_id.split("/")[0] if "/" in model_id else "unknown"
            return cls(model_id=model_id, provider=provider, is_free=True)
        elif "local" in model_id.lower():
            return cls(model_id=model_id, provider="local", is_free=True)
        elif "/" in model_id:
            provider = model_id.split("/")[0]
            return cls(model_id=model_id, provider=provider, is_free=False)
        else:
            return cls(model_id=model_id, provider="unknown", is_free=False)


@dataclass
class SimpleUsageMetrics:
    """Simple usage tracking without complex cost calculations."""

    model_id: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    is_free: bool
    timestamp: datetime

    @property
    def estimated_cost_usd(self) -> float:
        """Very rough cost estimation - mainly for tracking purposes."""
        if self.is_free:
            return 0.0
        # Very rough estimate: $0.000002 per token
        return self.total_tokens * 0.000002


class SimpleLLMTracker:
    """
    Simplified LLM tracking focused on basic metrics without complex pricing.

    Features:
    - Model name and provider tracking
    - Token count recording
    - Free vs paid model detection
    - Very basic cost estimation
    - No complex pricing tables or API calls
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the simple tracker."""
        self.config = config or {}
        self.tracking_enabled = self.config.get("cost_tracking", {}).get(
            "enabled", True
        )

        # Simple statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_estimated_cost = 0.0

    def track_usage(
        self, model_id: str, usage_data: Dict[str, Any]
    ) -> SimpleUsageMetrics:
        """Track simple usage metrics for a model."""
        if not self.tracking_enabled:
            return None

        model_info = BasicModelInfo.from_model_id(model_id)

        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)

        # Handle cases where individual counts aren't available
        if prompt_tokens == 0 and completion_tokens == 0 and total_tokens > 0:
            # Rough 60/40 split assumption
            prompt_tokens = int(total_tokens * 0.6)
            completion_tokens = int(total_tokens * 0.4)

        metrics = SimpleUsageMetrics(
            model_id=model_id,
            provider=model_info.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            is_free=model_info.is_free,
            timestamp=datetime.now(),
        )

        # Update simple statistics
        self.total_requests += 1
        self.total_tokens += total_tokens
        self.total_estimated_cost += metrics.estimated_cost_usd

        return metrics

    def get_model_info(self, model_id: str) -> BasicModelInfo:
        """Get basic model information."""
        return BasicModelInfo.from_model_id(model_id)

    def estimate_simple_cost(self, total_tokens: int, is_free: bool = False) -> float:
        """Simple cost estimation."""
        if is_free:
            return 0.0
        return total_tokens * 0.000002  # Very rough estimate

    def get_stats(self) -> Dict[str, Any]:
        """Get basic tracking statistics."""
        avg_tokens = self.total_tokens / max(1, self.total_requests)
        avg_cost = self.total_estimated_cost / max(1, self.total_requests)

        return {
            "tracking_enabled": self.tracking_enabled,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": round(avg_tokens, 1),
            "total_estimated_cost_usd": round(self.total_estimated_cost, 6),
            "avg_cost_per_request_usd": round(avg_cost, 6),
        }


# For backward compatibility, alias the simplified tracker as LLMCostEngine
LLMCostEngine = SimpleLLMTracker


def suggest_cost_apis() -> Dict[str, str]:
    """Suggest simple APIs for future cost integration."""
    return {
        "openrouter_models": "https://openrouter.ai/api/v1/models",
        "openai_pricing": "https://api.openai.com/v1/models",
        "anthropic_pricing": "https://console.anthropic.com/docs/api/pricing",
        "simple_cost_tracking": "Consider using existing telemetry data for cost trends",
    }
