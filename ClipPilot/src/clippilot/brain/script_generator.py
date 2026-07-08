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
from clippilot.brain.creative_models import CreativeBlueprint
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
    metadata: dict[str, Any] = field(default_factory=dict)

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "hook": {"type": "string"},
        "niche_context": {"type": "string"},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "narration": {"type": "string"},
                    "visual_desc": {"type": "string"}
                },
                "required": ["narration", "visual_desc"],
                "additionalProperties": False
            }
        }
    },
    "required": ["title", "hook", "niche_context", "scenes"],
    "additionalProperties": False
}



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
    from clippilot.brain.provider import tolerant_json_loads
    try:
        data = tolerant_json_loads(text)
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
    fallback_to_mock: bool = True,
    blueprint: Optional[CreativeBlueprint] = None,
) -> Script:
    """Generate a structured script using LLM prompts based on current topic/rules.

    Retries on validation/parse errors up to 3 times, raising ScriptValidationError on exhaustion.
    """
    import os
    import time
    from clippilot.config import Settings

    # Derive a dynamic video title from the topic title
    topic_title = topic.title.replace("-", " ")
    title_words = topic_title.strip().replace(".", "").replace("!", "").replace("?", "").split()
    capitalized = []
    for w in title_words:
        if w.lower() in ["is", "on", "your", "now", "in", "and", "the", "a", "an", "to", "for", "with", "by", "at"] and len(capitalized) > 0:
            capitalized.append(w.lower())
        else:
            capitalized.append(w.capitalize())
    mock_title_derived = " ".join(capitalized)
    if not any(mock_title_derived.startswith(prefix) for prefix in ["Why", "How", "Stop", "The", "What"]):
        mock_title_derived = f"Why {mock_title_derived}"

    mock_script = Script(
        topic_num=topic.num,
        title=mock_title_derived,
        hook=topic.angle or topic.title,
        niche_context=topic.niche or "credit",
        scenes=[
            ScriptScene(
                narration=f"Here is the truth about {topic.title.lower()}.",
                visual_desc=f"Showing visual representation of {topic.title}."
            ),
            ScriptScene(
                narration=f"Remember this key fact: {topic.angle or topic.title}.",
                visual_desc="Visual of older card account glowing with green border."
            )
        ],
        metadata={
            "provider": "mock",
            "model": "mock",
            "fallback_flag": True,
            "retries": 0,
            "latency": 0.0,
            "title": mock_title_derived,
        }
    )

    settings = Settings.load()

    from clippilot.brain.env import has_api_key
    if fallback_to_mock and not has_api_key():
        print("⚠️ No API key configured. Using deterministic mock script.")
        mock_script.metadata.update({
            "provider": "MockProvider",
            "requested_model": "mock",
            "actual_model": "mock",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": 0.0,
            "estimated_cost": 0.0,
            "cost": 0.0,
            "fallback_flag": True
        })
        return mock_script

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

    if blueprint:
        from clippilot.brain.prompt_builder import build_blueprint_instructions
        blueprint_instructions = build_blueprint_instructions(blueprint)
        user_prompt = (
            f"Generate a validated vertical video short script for:\n"
            f"- Topic Num: {topic.num}\n"
            f"- Topic Title Proposal: {topic.title}\n"
            f"- Niche/CPM: {topic.niche}\n"
            f"- True Mechanism Angle: {topic.angle}\n"
            f"- Brand Guardrails: {topic.guardrail}\n\n"
            f"{blueprint_instructions}\n\n"
            f"Obey these self-learned optimization rules:\n"
            f"{state.learned_rules}\n\n"
            f"Steal these technique catalog patterns from the competitor playbook:\n"
            f"{playbook_content[:3000]}\n"  # Truncated to avoid token bloat
        )
    else:
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
    try:
        models = settings.script_models
        if not models:
            models = [settings.script_model] if settings.script_model else ([settings.llm_model] if settings.llm_model else [])
        provider = get_provider(settings, models=models)
    except Exception as e:
        if fallback_to_mock:
            print(f"⚠️ Provider initialization failed: {e}. Falling back to deterministic mock script.")
            mock_script.metadata.update({
                "provider": "MockProvider",
                "requested_model": "mock",
                "actual_model": "mock",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency": 0.0,
                "estimated_cost": 0.0,
                "cost": 0.0,
                "fallback_flag": True
            })
            return mock_script
        else:
            raise ScriptValidationError(f"Provider initialization failed: {e}")

    last_err: Optional[Exception] = None
    start_time = time.time()
    raw_response = None
    api_succeeded = False
    attempt = 0

    for attempt in range(retries):
        try:
            raw_response = provider.generate_text(prompt=user_prompt, system_prompt=system_prompt, json_schema=SCRIPT_SCHEMA)
            api_succeeded = True
            break
        except Exception as e:
            from clippilot.brain.provider import JSONParsingError
            if isinstance(e, JSONParsingError):
                last_err = e
                break
            last_err = e
            continue

    if not api_succeeded:
        if fallback_to_mock:
            latency = round(time.time() - start_time, 4)
            err_reason = "JSON validation failure"
            if "schema validation" in str(last_err).lower() or "missing required key" in str(last_err).lower():
                err_reason = "Schema validation failure"
            elif last_err and "leakage" in str(last_err).lower():
                err_reason = "Reasoning leakage"
            print(f"⚠️ Script generation API failed after {attempt + 1} attempts. Last error: {last_err}. Falling back to deterministic mock script.")
            mock_script.metadata.update({
                "provider": provider.__class__.__name__ if 'provider' in locals() and provider else "MockProvider",
                "requested_model": getattr(provider, "requested_model", "mock") if 'provider' in locals() and provider else "mock",
                "actual_model": getattr(provider, "model", "mock") if 'provider' in locals() and provider else "mock",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency": latency,
                "estimated_cost": 0.0,
                "cost": 0.0,
                "retries": attempt + 1,
                "fallback_flag": True
            })
            return mock_script
        else:
            raise ScriptValidationError(f"Script generation failed: {last_err}")

    # Once raw_response is fetched successfully: DO NOT retry the API.
    # Parse and validate the response
    try:
        parsed_data = _validate_and_parse_json(raw_response)
        
        # Map to typed Script object
        script_scenes = [
            ScriptScene(narration=s["narration"], visual_desc=s["visual_desc"])
            for s in parsed_data["scenes"]
        ]
        latency = round(time.time() - start_time, 4)
        
        provider_model = getattr(provider, "model", "unknown")
        requested_model = getattr(provider, "requested_model", provider_model)
        print(f"✔️ Successfully generated script using provider '{provider.__class__.__name__}' (model: {provider_model}) in {latency}s.")
        
        last_usage = getattr(provider, "last_usage", {})
        meta_dict = {
            "provider": provider.__class__.__name__,
            "requested_model": requested_model,
            "actual_model": provider_model,
            "input_tokens": last_usage.get("input_tokens", 0),
            "output_tokens": last_usage.get("output_tokens", 0),
            "latency": latency,
            "estimated_cost": last_usage.get("estimated_cost", 0.0),
            "cost": last_usage.get("estimated_cost", 0.0),
            "fallback_flag": False,
            "retries": attempt + 1,
            "title": parsed_data["title"],
        }
            
        return Script(
            topic_num=topic.num,
            title=parsed_data["title"],
            hook=parsed_data["hook"],
            niche_context=parsed_data["niche_context"],
            scenes=script_scenes,
            metadata=meta_dict,
        )
    except Exception as parse_err:
        from clippilot.logger import get_logger
        logger = get_logger("clippilot.pipeline")
        
        is_leakage = False
        if raw_response:
            from clippilot.brain.provider import detects_reasoning_leakage
            is_leakage = detects_reasoning_leakage(raw_response)
            
        reason = "JSON validation failure"
        if is_leakage:
            reason = "Reasoning leakage"
        elif "Missing required key" in str(parse_err) or "must be a list" in str(parse_err) or "cannot be empty" in str(parse_err) or "schema validation" in str(parse_err).lower():
            reason = "Schema validation failure"
            
        logger.error(f"Unable to repair malformed JSON. Using deterministic mock fallback. Failure Reason: {reason}. Error: {parse_err}")
        from clippilot.brain.provider import save_failed_response
        save_failed_response(
            stage_name="script",
            raw_response=raw_response,
            reason=reason,
            provider=provider.__class__.__name__,
            requested_model=getattr(provider, "requested_model", None),
            actual_model=getattr(provider, "model", None),
            correction_attempted=True,
            repair_attempted=True,
            schema_validation_status="failed" if reason == "Schema validation failure" else "not_applicable"
        )
        if fallback_to_mock:
            latency = round(time.time() - start_time, 4)
            last_usage = getattr(provider, "last_usage", {})
            provider_model = getattr(provider, "model", "unknown")
            requested_model = getattr(provider, "requested_model", provider_model)
            mock_script.metadata.update({
                "provider": provider.__class__.__name__,
                "requested_model": requested_model,
                "actual_model": provider_model,
                "input_tokens": last_usage.get("input_tokens", 0),
                "output_tokens": last_usage.get("output_tokens", 0),
                "latency": latency,
                "estimated_cost": last_usage.get("estimated_cost", 0.0),
                "cost": last_usage.get("estimated_cost", 0.0),
                "retries": attempt + 1,
                "fallback_flag": True
            })
            return mock_script
        else:
            raise ScriptValidationError(f"JSON validation failed: {parse_err}")
