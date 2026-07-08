"""Minimal .env loader + API-key resolution (stdlib only).

We do NOT take a dependency on python-dotenv. Secrets live in the environment
(`ANTHROPIC_API_KEY`) or a gitignored `.env` at the project root — never pasted
into chat or committed. Existing env vars are never overwritten by .env.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .. import config as cfg

API_KEY_VAR = "ANTHROPIC_API_KEY"


def load_dotenv(path: Optional[Path] = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no override)."""
    if os.environ.get("CLIPPILOT_TESTING") == "true":
        return
    path = path or (cfg.PROJECT_ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def get_api_key() -> Optional[str]:
    """Resolve the Anthropic API key from env (loading .env first)."""
    load_dotenv()
    key = os.environ.get(API_KEY_VAR)
    return key or None


def has_api_key() -> bool:
    load_dotenv()
    try:
        from ..config import Settings
        s = Settings.load()
        if s.llm_api_key:
            return True
    except Exception:
        pass
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )
