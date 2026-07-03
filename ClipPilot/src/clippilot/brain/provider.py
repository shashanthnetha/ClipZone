"""LLM Provider abstraction layer: defines a common provider interface and implements the Anthropic engine.

Designed to be provider-agnostic, enabling alternative backends to be slotted in by implementing the LLMProvider protocol.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Protocol

from .. import config as cfg
from . import env as benv


class LLMProvider(Protocol):
    """Common interface for LLM provider implementations."""

    def supports_vision(self) -> bool:
        """Return True if this provider/model supports multimodal vision input."""
        ...

    def supports_json(self) -> bool:
        """Return True if this provider supports structured JSON output schemas."""
        ...

    def supports_streaming(self) -> bool:
        """Return True if this provider supports token streaming."""
        ...

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from a text prompt."""
        ...

    def generate_vision(
        self,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Process multimodal inputs (e.g., base64 keyframes) and return a parsed structured dict."""
        ...


class AnthropicProvider:
    """Anthropic Claude API Provider implementation."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or benv.get_api_key()
        self.base_url = base_url or os.environ.get("LLM_BASE_URL")

        # Validate initialization requirements
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is missing. Please set the ANTHROPIC_API_KEY or LLM_API_KEY environment variable, "
                "or configure it in your settings.json."
            )

        self._client = None  # Lazy-loaded

    def _ensure_client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = Anthropic(**kwargs)
        return self._client

    def supports_vision(self) -> bool:
        return True

    def supports_json(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        client = self._ensure_client()
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        resp = client.messages.create(**kwargs)
        return "".join(
            getattr(b, "text", "")
            for b in resp.content
            if getattr(b, "type", None) == "text"
        )

    def generate_vision(
        self,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8000,
            "thinking": {"type": "adaptive"},
            "messages": messages,
            "output_config": {"format": {"type": "json_schema", "schema": json_schema}},
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        resp = client.messages.parse(**kwargs)
        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:
            raise RuntimeError(
                f"No parsed_output returned from Anthropic provider (stop_reason={getattr(resp, 'stop_reason', '?')})"
            )
        return dict(parsed)


def get_provider(settings: Optional[cfg.Settings] = None) -> LLMProvider:
    """Resolve and return the configured LLMProvider instance."""
    s = settings or cfg.Settings.load()

    # Read config settings with environment variable overrides
    provider_name = (
        os.environ.get("LLM_PROVIDER")
        or getattr(s, "llm_provider", "anthropic")
    ).lower()
    
    model = (
        os.environ.get("LLM_MODEL")
        or getattr(s, "llm_model", None)
        or getattr(s, "brain_model", "claude-opus-4-8")
    )

    base_url = os.environ.get("LLM_BASE_URL") or getattr(s, "llm_base_url", None)
    api_key = os.environ.get("LLM_API_KEY") or getattr(s, "llm_api_key", None)

    if provider_name == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key, base_url=base_url)

    raise ValueError(f"Unsupported or unrecognized LLM provider type: {provider_name}")
