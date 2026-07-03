"""Vision clients: the real Anthropic-SDK client and a mock for tests.

`VisionClient.vision_understand(...)` returns the enrichment dict (validated
against prompt.ENRICHMENT_SCHEMA by the API's structured-output mode). The real
client lazy-imports `anthropic`, so the rest of the package (and the whole test
suite) works without the SDK or an API key installed.
"""
from __future__ import annotations

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


class AnthropicVisionClient:
    """Real client. Sends keyframes as image blocks + forces the JSON schema."""

    def __init__(self, model: str = "claude-opus-4-8", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    def vision_understand(self, u: Understanding, keyframe_paths: list[str]) -> dict[str, Any]:
        from .provider import get_provider
        from .prompt import ENRICHMENT_SCHEMA

        # Resolve provider using get_provider, overriding model/api_key if custom values were supplied
        settings = cfg.Settings.load()
        if self.api_key:
            settings.llm_api_key = self.api_key
        if self.model:
            settings.llm_model = self.model

        provider = get_provider(settings)
        if not provider.supports_vision():
            raise RuntimeError("The configured provider does not support vision capabilities.")

        req = build_vision_request(u, keyframe_paths, model=self.model)
        system = req.get("system")
        messages = req.get("messages", [])

        return provider.generate_vision(
            messages=messages,
            json_schema=ENRICHMENT_SCHEMA,
            system_prompt=system
        )


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
    """Real client if `anthropic` is importable AND a key is present; else None.
    Callers fall back to the deterministic Understanding (no enrichment)."""
    settings = settings or cfg.Settings.load()
    if not env.has_api_key():
        return None
    try:
        import anthropic  # noqa: F401 — availability probe
    except ImportError:
        return None
    return AnthropicVisionClient(model=settings.brain_model)


def estimate_cost_usd(model: str, frames: int, avg_frame_tokens: int = 600,
                      text_tokens: int = 10000, output_tokens: int = 4000) -> float:
    """Rough Claude spend for one understanding pass (docs/07 cost model)."""
    in_price, out_price = PRICING.get(model, PRICING["claude-opus-4-8"])
    input_tokens = frames * avg_frame_tokens + text_tokens
    return round(input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price, 4)
