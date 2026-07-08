# -*- coding: utf-8 -*-
"""ClipPilot Test Suite Package Initialization.

Sets CLIPPILOT_TESTING to prevent loading of local developer .env files,
and clears/mocks API keys to ensure isolated sandboxed unit test execution.
"""
from __future__ import annotations

import os

os.environ["CLIPPILOT_TESTING"] = "true"

# Overwrite/clear API keys so tests default to deterministic mocked fallbacks
# unless a test explicitly sets a dummy key.
for key in [
    "LLM_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PEXELS_API_KEY",
    "PIXABAY_API_KEY",
    "UNSPLASH_API_KEY",
    "YOUTUBE_CLIENT_ID",
    "YOUTUBE_CLIENT_SECRET",
    "YOUTUBE_REFRESH_TOKEN",
    "GEN_IMAGE_API_KEY",
]:
    os.environ[key] = ""
