# -*- coding: utf-8 -*-
"""ClipPilot Brain Vision Provider Abstraction.

Defines the common VisionProvider Protocol and implements the
OpenAICompatibleVisionProvider utilizing standard base64 image requests.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import requests
from clippilot.config import Settings


@dataclass
class VisionRequest:
    """Represents a request payload sent to the Vision API."""
    image_paths: list[str]
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 1024


@dataclass
class VisionResult:
    """Represents the text response returned from the Vision API."""
    text: str
    usage: dict[str, Any] = field(default_factory=dict)


class VisionProvider(Protocol):
    """Protocol defining the standard interface for all Vision API backends."""

    def analyze_images(self, request: VisionRequest) -> VisionResult:
        """Submit images and a text prompt to the Vision model for analysis."""
        ...

    def validate(self) -> None:
        """Validate credentials, API keys, or connections on startup."""
        ...

    def supports_vision(self) -> bool:
        """Returns True if the provider supports image analysis."""
        ...


class OpenAICompatibleVisionProvider:
    """Vision Provider implementation using OpenAI Chat Completions API format."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        settings = Settings.load()
        self.api_key = api_key or settings.llm_api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or settings.llm_base_url or "https://api.openai.com/v1"
        self.model_name = model_name or settings.llm_model or "gpt-4o"

    def validate(self) -> None:
        """Assert API credentials are fully configured."""
        if not self.api_key:
            raise ValueError(
                "API key is missing. Please set LLM_API_KEY or OPENAI_API_KEY."
            )

    def supports_vision(self) -> bool:
        return True

    def _encode_image(self, path: str) -> str:
        """Read and encode an image file to base64 format."""
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_images(self, request: VisionRequest) -> VisionResult:
        """Send base64 encoded images and prompt to the completions endpoint."""
        self.validate()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Build messages payload structure
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        user_content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]

        for img_path in request.image_paths:
            if not os.path.exists(img_path):
                continue
            b64_data = self._encode_image(img_path)
            # Guess mime type from suffix
            ext = Path(img_path).suffix.lower()
            mime = "image/jpeg"
            if ext == ".png":
                mime = "image/png"
            elif ext == ".webp":
                mime = "image/webp"

            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64_data}"
                    }
                }
            )

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens,
        }

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Vision API returned empty choices: {data}")

        text = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        return VisionResult(text=text, usage=usage)


def get_vision_provider(provider_name: str = "openai-compatible") -> VisionProvider:
    """Factory to retrieve the active, configured VisionProvider."""
    if provider_name.lower() in ["openai-compatible", "openrouter", "openai"]:
        return OpenAICompatibleVisionProvider()
    raise ValueError(f"Unknown vision provider: {provider_name}")
