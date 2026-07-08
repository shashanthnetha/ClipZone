"""The Claude brain — the vision pass that turns the deterministic Understanding
(docs/07) into a human-like read by looking at the sampled keyframes.

Claude is the MCP *client* in the running app; here, for the local pipeline, we
call the Anthropic SDK directly to do the vision/judgment step. The client is
mockable so unit tests pass WITHOUT an API key (see tests/test_brain.py). Live
runs need ANTHROPIC_API_KEY in the environment or a gitignored .env.
"""
from .client import ProviderVisionClient, MockVisionClient, VisionClient, get_client
from .orchestrator import enrich_understanding
from .prompt import build_vision_request
AnthropicVisionClient = ProviderVisionClient

__all__ = [
    "VisionClient", "ProviderVisionClient", "AnthropicVisionClient", "MockVisionClient", "get_client",
    "build_vision_request", "enrich_understanding",
]
