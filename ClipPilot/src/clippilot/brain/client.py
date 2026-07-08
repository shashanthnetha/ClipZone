"""Vision clients: the real Anthropic-SDK client and a mock for tests.

`VisionClient.vision_understand(...)` returns the enrichment dict (validated
against prompt.ENRICHMENT_SCHEMA by the API's structured-output mode). The real
client lazy-imports `anthropic`, so the rest of the package (and the whole test
suite) works without the SDK or an API key installed.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Protocol

from .. import config as cfg
from ..understanding import Understanding
from . import env
from .prompt import build_vision_request

# Anthropic vision pricing (per 1M tokens) — for the cost note (claude-api skill).
PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class VisionClient(Protocol):
    def vision_understand(self, u: Understanding, keyframe_paths: list[str]) -> dict[str, Any]:
        ...


class ProviderVisionClient:
    """Real client. Sends keyframes as image blocks + forces the JSON schema via resolved provider."""

    def __init__(self, model: str, provider_name: str):
        self.model = model
        self.provider_name = provider_name

    def vision_understand(self, u: Understanding, keyframe_paths: list[str]) -> dict[str, Any]:
        from .provider import get_provider
        from .prompt import ENRICHMENT_SCHEMA

        settings = cfg.Settings.load()
        settings.llm_model = self.model
        settings.llm_provider = self.provider_name

        provider = get_provider(settings)
        if not provider.supports_vision():
            raise RuntimeError("The configured provider does not support vision capabilities.")

        req = build_vision_request(u, keyframe_paths, model=self.model)
        system = req.get("system")
        messages = req.get("messages", [])

        self.last_usage = {}
        result = provider.generate_vision(
            messages=messages,
            json_schema=ENRICHMENT_SCHEMA,
            system_prompt=system
        )
        if hasattr(provider, "last_usage"):
            self.last_usage = provider.last_usage
        return result


class MockVisionClient:
    """Deterministic stand-in so tests run without the SDK/key. Echoes shape."""

    def __init__(self, identifiable_person: bool = True):
        self.identifiable_person = identifiable_person
        self.calls: list[int] = []

    def vision_understand(self, u: Understanding, keyframe_paths: list[str]) -> dict[str, Any]:
        self.calls.append(len(keyframe_paths))
        return {
            "summary": "(mock) A short clip with one speaker and an energetic payoff.",
            "topics": ["mock-topic"],
            "entities": [],
            "on_screen_text": ["MOCK CAPTION"],
            "scene_descriptions": [{"idx": s.idx, "visual_desc": f"(mock) scene {s.idx}"}
                                   for s in u.scenes],
            "mood_label": "energetic",
            "identifiable_person_likely": self.identifiable_person,
            "highlight_candidates": [
                {"start": 1.0, "end": 4.0, "score": 0.82,
                 "reasons": ["(mock) energy peak", "(mock) quotable line"]},
            ],
        }


def get_client(settings: Optional[cfg.Settings] = None) -> Optional[VisionClient]:
    """Real client if the configured provider is importable and its key is present; else None."""
    if not env.has_api_key():
        return None
    settings = settings or cfg.Settings.load()
    provider_name = settings.llm_provider.lower()
    
    has_key = False
    if provider_name == "anthropic":
        has_key = bool(settings.llm_api_key or os.environ.get("ANTHROPIC_API_KEY"))
    elif provider_name == "openai":
        has_key = bool(settings.llm_api_key or os.environ.get("OPENAI_API_KEY"))
    elif provider_name == "openrouter":
        has_key = bool(settings.llm_api_key or os.environ.get("OPENROUTER_API_KEY"))
        
    if not has_key:
        return None

    if provider_name == "anthropic":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return None
    elif provider_name in ("openai", "openrouter"):
        try:
            import openai  # noqa: F401
        except ImportError:
            return None

    model = settings.vision_model or settings.brain_model or settings.llm_model
    if not model:
        if provider_name == "anthropic":
            model = "claude-opus-4-8"
        elif provider_name == "openai":
            model = "gpt-4o"
        elif provider_name == "openrouter":
            model = "openrouter/free"

    return ProviderVisionClient(model=model, provider_name=provider_name)


def estimate_cost_usd(model: str, frames: int, avg_frame_tokens: int = 600,
                      text_tokens: int = 10000, output_tokens: int = 4000) -> float:
    """Rough Claude spend for one understanding pass (docs/07 cost model)."""
    in_price, out_price = PRICING.get(model, PRICING["claude-opus-4-8"])
    input_tokens = frames * avg_frame_tokens + text_tokens
    return round(input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price, 4)
