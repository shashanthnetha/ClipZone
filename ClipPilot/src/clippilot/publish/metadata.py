"""Social-metadata generation (title / caption / hashtags).

Ported from MoneyPrinterTurbo `app/services/llm.py` (MIT, (c) 2024 Harry) — the
per-platform spec table, the prompt contract, the JSON-with-code-fence recovery,
and hashtag normalization — re-authored and decoupled from its provider zoo. The
LLM call is supplied by the caller (`generate_fn`), so this is fully unit-testable
with a mock and, in production, routes through ClipPilot's own Claude brain.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

# Per-platform caps (from MoneyPrinterTurbo SOCIAL_PLATFORMS). MIT.
SOCIAL_PLATFORMS: dict[str, dict[str, int]] = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}

# A callable that takes a prompt and returns the raw LLM text.
GenerateFn = Callable[[str], str]


def build_metadata_prompt(subject: str, script: str, platform: str, language: str = "en") -> str:
    spec = SOCIAL_PLATFORMS.get(platform, SOCIAL_PLATFORMS["youtube_shorts"])
    n = spec["hashtag_count"]
    return (
        f"You write high-performing {platform.replace('_', ' ')} metadata.\n"
        f"VIDEO SUBJECT: {subject}\n"
        f"VIDEO CONTENT: {script[:2000]}\n\n"
        f"Write the post metadata in {language}. Respond with ONLY a single valid minified "
        f"JSON object — no prose, no code fence — with exactly these keys:\n"
        f'  "title": a scroll-stopping title, max {spec["title_max"]} characters.\n'
        f'  "caption": an engaging caption, max {spec["caption_max"]} characters.\n'
        f'  "hashtags": a JSON array of exactly {n} strings, each starting with "#", no spaces, no punctuation.\n'
    )


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    return t


def parse_metadata(text: str) -> Optional[dict[str, Any]]:
    """Recover the JSON object from an LLM response (handles code fences / chatter)."""
    if not text:
        return None
    try:
        return json.loads(_strip_code_fence(text))
    except (json.JSONDecodeError, ValueError):
        # Recover the first JSON object even when wrapped in prose. raw_decode
        # stops at the object's end, so trailing chatter (e.g. a ":}" emoticon)
        # no longer extends a greedy {.*} match and breaks parsing.
        idx = text.find("{")
        if idx != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[idx:])
                return obj
            except (json.JSONDecodeError, ValueError):
                return None
    return None


def normalize_hashtags(items: list[Any], count: int) -> list[str]:
    """Strip non-word chars, dedupe (case-insensitive), prefix '#', cap to `count`."""
    if count <= 0:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        word = re.sub(r"[^\w]", "", str(item), flags=re.UNICODE)
        if not word:
            continue
        low = word.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append("#" + word)
        if len(out) >= count:
            break
    return out


def fallback_metadata(subject: str, platform: str) -> dict[str, Any]:
    """Heuristic metadata when the LLM is unavailable (no key / error)."""
    spec = SOCIAL_PLATFORMS.get(platform, SOCIAL_PLATFORMS["youtube_shorts"])
    subject = (subject or "Watch this").strip()
    return {
        "title": subject[: spec["title_max"]],
        "caption": subject[: spec["caption_max"]],
        "hashtags": normalize_hashtags(["shorts", "viral", "fyp", "trending", "reels"],
                                       spec["hashtag_count"]),
        "_fallback": True,
    }


def generate_social_metadata(subject: str, script: str, platform: str = "youtube_shorts",
                             generate_fn: Optional[GenerateFn] = None,
                             language: str = "en") -> dict[str, Any]:
    """Produce {title, caption, hashtags} for `platform`, enforcing the caps.

    `generate_fn(prompt) -> text` is the LLM call (mock in tests, Claude in prod).
    Falls back to a heuristic when no generator or on parse failure.
    """
    spec = SOCIAL_PLATFORMS.get(platform, SOCIAL_PLATFORMS["youtube_shorts"])
    if generate_fn is None:
        return fallback_metadata(subject, platform)

    prompt = build_metadata_prompt(subject, script, platform, language)
    try:
        raw = generate_fn(prompt)
    except Exception:  # noqa: BLE001 — network/LLM failure → heuristic
        return fallback_metadata(subject, platform)

    data = parse_metadata(raw)
    if not isinstance(data, dict) or "title" not in data:  # a JSON list has no .get
        return fallback_metadata(subject, platform)

    return {
        "title": str(data.get("title", subject))[: spec["title_max"]],
        "caption": str(data.get("caption", ""))[: spec["caption_max"]],
        "hashtags": normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"]),
    }


def from_understanding(understanding: dict[str, Any], platform: str = "youtube_shorts",
                       generate_fn: Optional[GenerateFn] = None) -> dict[str, Any]:
    """Build metadata from an `Understanding`-shaped dict (summary + transcript + topics)."""
    summary = (understanding.get("summary") or "").strip()
    topics = understanding.get("topics") or []
    scenes = understanding.get("scenes") or []
    transcript = " ".join(s.get("transcript_excerpt", "") for s in scenes).strip()
    subject = summary or (topics[0] if topics else "Watch this")
    script = (summary + "\n" + transcript + "\nTopics: " + ", ".join(map(str, topics))).strip()
    return generate_social_metadata(subject, script, platform, generate_fn)


def claude_text_generator(settings) -> Optional[GenerateFn]:
    """A `generate_fn` backed by Claude (text). None when no key/SDK — caller then
    uses the heuristic fallback."""
    from ..brain import env as benv
    if not benv.has_api_key():
        return None
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return None

    from ..brain.provider import get_provider
    provider = get_provider(settings)

    def gen(prompt: str) -> str:
        return provider.generate_text(prompt)
    return gen
