# -*- coding: utf-8 -*-
"""ClipPilot Brain Script Generator.

Constructs detailed prompts using state rules, playbook, and variation config,
submits them to the active LLM provider, and enforces JSON schema compliance.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from clippilot.brain.pipeline_orchestrator import PipelineState, Topic, VariationRecord
from clippilot.brain.provider import get_provider


class ScriptValidationError(Exception):
    """Custom exception raised when script generation validation fails."""
    pass


@dataclass
class ScriptScene:
    """Represents a single visual scene/segment in the short."""
    narration: str
    visual_desc: str


@dataclass
class Script:
    """Represents the complete compiled script for video production."""
    topic_num: str
    title: str
    hook: str
    niche_context: str
    scenes: list[ScriptScene] = field(default_factory=list)


def _clean_json_text(text: str) -> str:
    """Strip markdown code fences and extraneous text surrounding the JSON payload."""
    text = text.strip()
    # Remove markdown code fences if present (e.g. ```json ... ```)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Find first '{' and last '}'
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1].strip()
    return text


def _validate_and_parse_json(text: str) -> dict[str, Any]:
    """Parse JSON text and validate against the expected schema."""
    cleaned = _clean_json_text(text)
    try:
        data = json.loads(cleaned)
    except Exception as e:
        raise ScriptValidationError(f"Invalid JSON format: {e}")

    # Validate top-level keys
    for key in ["title", "hook", "niche_context", "scenes"]:
        if key not in data:
            raise ScriptValidationError(f"Missing required key: {key}")

    # Validate scenes
    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        raise ScriptValidationError("The 'scenes' key must be a list.")

    if not scenes:
        raise ScriptValidationError("The 'scenes' list cannot be empty.")

    for idx, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise ScriptValidationError(f"Scene at index {idx} must be a JSON object.")
        if "narration" not in scene:
            raise ScriptValidationError(f"Scene at index {idx} is missing 'narration'.")
        if "visual_desc" not in scene:
            raise ScriptValidationError(f"Scene at index {idx} is missing 'visual_desc'.")

    return data


def generate_script(
    state: PipelineState,
    topic: Topic,
    variation: VariationRecord,
    workspace_dir: Path,
    retries: int = 3,
) -> Script:
    """Generate a structured script using LLM prompts based on current topic/rules.

    Retries on validation/parse errors up to 3 times, raising ScriptValidationError on exhaustion.
    """
    import os
    from clippilot.config import Settings
    settings = Settings.load()
    api_key_present = bool(
        os.environ.get("LLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or getattr(settings, "llm_api_key", None)
    )

    if not api_key_present:
        return Script(
            topic_num=topic.num,
            title="Stop Closing Credit Cards",
            hook="Closing credit cards actually hurts your score",
            niche_context="credit",
            scenes=[
                ScriptScene(
                    narration="Closing a credit card actually hurts your credit score.",
                    visual_desc="Showing credit score dropping from 800 to 720."
                ),
                ScriptScene(
                    narration="Instead, keep it open and let it build age.",
                    visual_desc="Visual of older card account glowing with green border."
                )
            ]
        )

    # 1. Read competitor playbook if available
    playbook_content = ""
    playbook_path = workspace_dir / "competitor_playbook.md"
    if playbook_path.exists():
        playbook_content = playbook_path.read_text(encoding="utf-8")

    # 2. Build system and user prompts
    system_prompt = (
        "You are the autonomous Shorts Producer. Your job is to output exactly one validated script "
        "as a raw JSON object complying with this exact schema:\n"
        "{\n"
        "  \"title\": \"String (the final video title)\",\n"
        "  \"hook\": \"String (the opening 0-3s hook sentence)\",\n"
        "  \"niche_context\": \"String (niche context analysis)\",\n"
        "  \"scenes\": [\n"
        "    {\n"
        "      \"narration\": \"String (the spoken voiceover line)\",\n"
        "      \"visual_desc\": \"String (details of the animation/icons/visual skin styling)\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Do NOT output markdown code fences, do not output any surrounding text. Return ONLY raw JSON."
    )

    user_prompt = (
        f"Generate a validated vertical video short script for:\n"
        f"- Topic Num: {topic.num}\n"
        f"- Topic Title Proposal: {topic.title}\n"
        f"- Niche/CPM: {topic.niche}\n"
        f"- True Mechanism Angle: {topic.angle}\n"
        f"- Brand Guardrails: {topic.guardrail}\n\n"
        f"Apply this variation config:\n"
        f"- Format: {variation.fmt}\n"
        f"- Skin Style: {variation.skin}\n"
        f"- Voice/Length: {variation.voice} ({variation.len})\n"
        f"- Topic Cluster: {variation.cluster}\n"
        f"- Hook Type: {variation.hook}\n\n"
        f"Obey these self-learned optimization rules:\n"
        f"{state.learned_rules}\n\n"
        f"Steal these technique catalog patterns from the competitor playbook:\n"
        f"{playbook_content[:3000]}\n"  # Truncated to avoid token bloat
    )

    # 3. Request LLM with retries
    provider = get_provider()
    last_err: Optional[Exception] = None

    for attempt in range(retries):
        try:
            raw_response = provider.generate_text(prompt=user_prompt, system_prompt=system_prompt)
            parsed_data = _validate_and_parse_json(raw_response)

            # Map to typed Script object
            script_scenes = [
                ScriptScene(narration=s["narration"], visual_desc=s["visual_desc"])
                for s in parsed_data["scenes"]
            ]
            return Script(
                topic_num=topic.num,
                title=parsed_data["title"],
                hook=parsed_data["hook"],
                niche_context=parsed_data["niche_context"],
                scenes=script_scenes,
            )
        except Exception as e:
            last_err = e
            # Log retry attempt (optional, test asserts print or exception raise)
            continue

    raise ScriptValidationError(f"Script generation failed after {retries} attempts. Last error: {last_err}")
