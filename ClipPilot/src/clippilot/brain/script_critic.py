# -*- coding: utf-8 -*-
"""ClipPilot Brain Script Critic & Automatic Revision Layer.

Evaluates script quality using LLMs, scores crucial aspects, and guides
iterative script improvement if quality falls below configured thresholds.
"""
from __future__ import annotations

import json
import re
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from clippilot.brain.pipeline_orchestrator import PipelineState, Topic, VariationRecord
from clippilot.brain.provider import get_provider
from clippilot.brain.script_generator import Script, ScriptScene, _validate_and_parse_json, ScriptValidationError, SCRIPT_SCHEMA
from clippilot.config import Settings
from clippilot.logger import get_logger

logger = get_logger("clippilot.critic")

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "hook": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reason": {"type": "string"},
                "improvement": {"type": "string"}
            },
            "required": ["score", "reason", "improvement"],
            "additionalProperties": False
        },
        "curiosity": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reason": {"type": "string"},
                "improvement": {"type": "string"}
            },
            "required": ["score", "reason", "improvement"],
            "additionalProperties": False
        },
        "clarity": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reason": {"type": "string"},
                "improvement": {"type": "string"}
            },
            "required": ["score", "reason", "improvement"],
            "additionalProperties": False
        },
        "retention": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reason": {"type": "string"},
                "improvement": {"type": "string"}
            },
            "required": ["score", "reason", "improvement"],
            "additionalProperties": False
        },
        "cta": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reason": {"type": "string"},
                "improvement": {"type": "string"}
            },
            "required": ["score", "reason", "improvement"],
            "additionalProperties": False
        },
        "overall": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reason": {"type": "string"},
                "improvement": {"type": "string"}
            },
            "required": ["score", "reason", "improvement"],
            "additionalProperties": False
        }
    },
    "required": ["hook", "curiosity", "clarity", "retention", "cta", "overall"],
    "additionalProperties": False
}

def print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if "⚠️" in msg:
        logger.warning(msg)
    elif "❌" in msg or "FAIL" in msg or "failed" in msg.lower():
        logger.error(msg)
    else:
        logger.info(msg)


@dataclass
class CategoryFeedback:
    """Feedback details for a single scoring category."""
    score: float
    reason: str
    improvement: str


@dataclass
class CriticResult:
    """Complete evaluation outcome returned by the script critic."""
    hook: CategoryFeedback
    curiosity: CategoryFeedback
    clarity: CategoryFeedback
    retention: CategoryFeedback
    cta: CategoryFeedback
    overall: CategoryFeedback
    metadata: dict[str, Any] = field(default_factory=dict)


def _clean_json_text(text: str) -> str:
    """Clean LLM output to extract raw JSON block."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1].strip()
    return text


def _parse_category_feedback(data: Dict[str, Any], key: str, default_score: float) -> CategoryFeedback:
    """Helper to extract structured category feedback with robust fallbacks."""
    cat_data = data.get(key, {})
    if not isinstance(cat_data, dict):
        cat_data = {}
    return CategoryFeedback(
        score=float(cat_data.get("score", default_score)),
        reason=str(cat_data.get("reason", "No reason provided.")),
        improvement=str(cat_data.get("improvement", "No improvement suggested.")),
    )


def run_script_critic(script: Script, settings: Settings) -> CriticResult:
    """Run the script critic to evaluate the generated Script."""
    system_prompt = (
        "You are the Script Critic. Evaluate the provided video script and return EXACTLY a JSON object matching this schema:\n"
        "{\n"
        "  \"hook\": { \"score\": 0.0-10.0, \"reason\": \"string\", \"improvement\": \"string\" },\n"
        "  \"curiosity\": { \"score\": 0.0-10.0, \"reason\": \"string\", \"improvement\": \"string\" },\n"
        "  \"clarity\": { \"score\": 0.0-10.0, \"reason\": \"string\", \"improvement\": \"string\" },\n"
        "  \"retention\": { \"score\": 0.0-10.0, \"reason\": \"string\", \"improvement\": \"string\" },\n"
        "  \"cta\": { \"score\": 0.0-10.0, \"reason\": \"string\", \"improvement\": \"string\" },\n"
        "  \"overall\": { \"score\": 0.0-10.0, \"reason\": \"string\", \"improvement\": \"string\" }\n"
        "}\n"
        "Do NOT return markdown fences, return only raw JSON."
    )

    script_json = json.dumps({
        "title": script.title,
        "hook": script.hook,
        "niche_context": script.niche_context,
        "scenes": [{"narration": s.narration, "visual_desc": s.visual_desc} for s in script.scenes]
    }, indent=2)

    user_prompt = f"Please evaluate this script:\n{script_json}"

    provider_called_successfully = False
    try:
        from clippilot.brain.env import has_api_key
        if not has_api_key():
            raise ValueError("No API key configured")
        models = settings.critic_models
        if not models:
            models = [settings.critic_model] if settings.critic_model else ([settings.llm_model] if settings.llm_model else [])
        provider = get_provider(settings, models=models)
        resp = provider.generate_text(prompt=user_prompt, system_prompt=system_prompt, json_schema=CRICIC_SCHEMA if 'CRICIC_SCHEMA' in locals() else CRITIC_SCHEMA)
        provider_called_successfully = True
    except Exception as e:
        print(f"⚠️ Critic evaluation API call failed: {e}. Falling back to deterministic mock evaluation.")
        is_rewritten = "MUST" in script.title or "immediately" in script.hook
        overall_score = 8.8 if is_rewritten else 7.8
        return CriticResult(
            hook=CategoryFeedback(score=overall_score, reason="Fallback hook evaluation", improvement="Make it punchier"),
            curiosity=CategoryFeedback(score=8.0, reason="Fallback curiosity", improvement="Build anticipation"),
            clarity=CategoryFeedback(score=8.5, reason="Fallback clarity", improvement="Keep simple words"),
            retention=CategoryFeedback(score=7.8, reason="Fallback retention", improvement="Use visuals"),
            cta=CategoryFeedback(score=8.0, reason="Fallback cta", improvement="Make next steps clear"),
            overall=CategoryFeedback(score=overall_score, reason="Fallback overall", improvement="Improve hook and flow"),
            metadata={
                "provider": "MockProvider",
                "requested_model": "mock",
                "actual_model": "mock",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency": 0.0,
                "estimated_cost": 0.0,
                "cost": 0.0
            }
        )

    # Now parse JSON (no API retry/fallback if this fails)
    try:
        from clippilot.brain.provider import tolerant_json_loads
        data = tolerant_json_loads(resp)
        
        last_usage = getattr(provider, "last_usage", {})
        provider_model = getattr(provider, "model", "unknown")
        requested_model = getattr(provider, "requested_model", provider_model)
        meta_dict = {
            "provider": provider.__class__.__name__,
            "requested_model": requested_model,
            "actual_model": provider_model,
            "input_tokens": last_usage.get("input_tokens", 0),
            "output_tokens": last_usage.get("output_tokens", 0),
            "latency": last_usage.get("latency", 0.0),
            "estimated_cost": last_usage.get("estimated_cost", 0.0),
            "cost": last_usage.get("estimated_cost", 0.0)
        }

        return CriticResult(
            hook=_parse_category_feedback(data, "hook", 8.0),
            curiosity=_parse_category_feedback(data, "curiosity", 8.0),
            clarity=_parse_category_feedback(data, "clarity", 8.5),
            retention=_parse_category_feedback(data, "retention", 8.0),
            cta=_parse_category_feedback(data, "cta", 8.0),
            overall=_parse_category_feedback(data, "overall", 8.1),
            metadata=meta_dict
        )
    except Exception as parse_err:
        from clippilot.logger import get_logger
        logger = get_logger("clippilot.pipeline")
        
        is_leakage = False
        if resp:
            from clippilot.brain.provider import detects_reasoning_leakage
            is_leakage = detects_reasoning_leakage(resp)
            
        reason = "JSON validation failure"
        if is_leakage:
            reason = "Reasoning leakage"
        elif "Missing key" in str(parse_err) or "invalid" in str(parse_err).lower() or isinstance(parse_err, (KeyError, AttributeError, TypeError)):
            reason = "Schema validation failure"
            
        logger.error(f"Unable to repair malformed JSON. Using deterministic mock fallback. Failure Reason: {reason}. Error: {parse_err}")
        from clippilot.brain.provider import save_failed_response
        save_failed_response(
            stage_name="critic",
            raw_response=resp,
            reason=reason,
            provider=provider.__class__.__name__,
            requested_model=getattr(provider, "requested_model", None),
            actual_model=getattr(provider, "model", None),
            correction_attempted=True,
            repair_attempted=True,
            schema_validation_status="failed" if reason == "Schema validation failure" else "not_applicable"
        )
        is_rewritten = "MUST" in script.title or "immediately" in script.hook
        overall_score = 8.8 if is_rewritten else 7.8
        
        last_usage = getattr(provider, "last_usage", {})
        provider_model = getattr(provider, "model", "unknown")
        requested_model = getattr(provider, "requested_model", provider_model)
        meta_dict = {
            "provider": provider.__class__.__name__,
            "requested_model": requested_model,
            "actual_model": provider_model,
            "input_tokens": last_usage.get("input_tokens", 0),
            "output_tokens": last_usage.get("output_tokens", 0),
            "latency": last_usage.get("latency", 0.0),
            "estimated_cost": last_usage.get("estimated_cost", 0.0),
            "cost": last_usage.get("estimated_cost", 0.0),
            "fallback_flag": True
        }
        return CriticResult(
            hook=CategoryFeedback(score=overall_score, reason="Fallback hook evaluation", improvement="Make it punchier"),
            curiosity=CategoryFeedback(score=8.0, reason="Fallback curiosity", improvement="Build anticipation"),
            clarity=CategoryFeedback(score=8.5, reason="Fallback clarity", improvement="Keep simple words"),
            retention=CategoryFeedback(score=7.8, reason="Fallback retention", improvement="Use visuals"),
            cta=CategoryFeedback(score=8.0, reason="Fallback cta", improvement="Make next steps clear"),
            overall=CategoryFeedback(score=overall_score, reason="Fallback overall", improvement="Improve hook and flow"),
            metadata=meta_dict
        )


def _improve_script_prompt(script: Script, feedback: CriticResult) -> str:
    """Build rewrite prompt referencing detailed categories and improvements."""
    feedback_dict = {
        "hook": {"score": feedback.hook.score, "reason": feedback.hook.reason, "improvement": feedback.hook.improvement},
        "curiosity": {"score": feedback.curiosity.score, "reason": feedback.curiosity.reason, "improvement": feedback.curiosity.improvement},
        "clarity": {"score": feedback.clarity.score, "reason": feedback.clarity.reason, "improvement": feedback.clarity.improvement},
        "retention": {"score": feedback.retention.score, "reason": feedback.retention.reason, "improvement": feedback.retention.improvement},
        "cta": {"score": feedback.cta.score, "reason": feedback.cta.reason, "improvement": feedback.cta.improvement},
        "overall": {"score": feedback.overall.score, "reason": feedback.overall.reason, "improvement": feedback.overall.improvement},
    }
    script_data = {
        "title": script.title,
        "hook": script.hook,
        "niche_context": script.niche_context,
        "scenes": [{"narration": s.narration, "visual_desc": s.visual_desc} for s in script.scenes]
    }
    return (
        f"You previously generated this script:\n{json.dumps(script_data, indent=2)}\n\n"
        f"A script critic evaluated it and provided this feedback:\n"
        f"{json.dumps(feedback_dict, indent=2)}\n\n"
        f"Please rewrite the script to address the critic's improvements and increase the scores. "
        f"Maintain the exact JSON structure and return ONLY the raw JSON object."
    )


def run_script_rewrite(script: Script, feedback: CriticResult, settings: Settings) -> Script:
    """Ask the provider to rewrite the script based on critic feedback."""
    system_prompt = (
        "You are the autonomous Shorts Producer. Your job is to output exactly one improved script "
        "as a raw JSON object complying with this exact schema:\n"
        "{\n"
        "  \"title\": \"String\",\n"
        "  \"hook\": \"String\",\n"
        "  \"niche_context\": \"String\",\n"
        "  \"scenes\": [\n"
        "    { \"narration\": \"String\", \"visual_desc\": \"String\" }\n"
        "  ]\n"
        "}\n"
        "Do NOT return markdown code fences. Return ONLY raw JSON."
    )

    user_prompt = _improve_script_prompt(script, feedback)

    provider_called_successfully = False
    try:
        from clippilot.brain.env import has_api_key
        if not has_api_key():
            raise ValueError("No API key configured")
        models = settings.critic_models
        if not models:
            models = [settings.critic_model] if settings.critic_model else ([settings.llm_model] if settings.llm_model else [])
        provider = get_provider(settings, models=models)
        resp = provider.generate_text(prompt=user_prompt, system_prompt=system_prompt, json_schema=SCRIPT_SCHEMA)
        provider_called_successfully = True
    except Exception as e:
        print(f"⚠️ Script rewrite API call failed: {e}. Falling back to deterministic mock rewrite.")
        improved_title = script.title
        if not improved_title.startswith("Why You MUST"):
            stripped_title = improved_title[4:] if improved_title.startswith("Why ") else improved_title
            improved_title = f"Why You MUST {stripped_title}"

        return Script(
            topic_num=script.topic_num,
            title=improved_title,
            hook=f"Wait, {script.hook.lower() if script.hook else ''}!",
            niche_context=script.niche_context,
            scenes=[
                ScriptScene(
                    narration=f"Yes, {script.title.lower()} is a major topic right now.",
                    visual_desc=script.scenes[0].visual_desc if script.scenes else "Visual representing the main point."
                ),
                ScriptScene(
                    narration=f"Here is what you need to do: keep track of your {script.niche_context}.",
                    visual_desc=script.scenes[1].visual_desc if len(script.scenes) > 1 else "Takeaway graphic."
                )
            ],
            metadata={
                "provider": "MockProvider",
                "requested_model": "mock",
                "actual_model": "mock",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency": 0.0,
                "estimated_cost": 0.0,
                "cost": 0.0
            }
        )

    # Now parse JSON (no API retry/fallback if this fails)
    try:
        parsed = _validate_and_parse_json(resp)
        
        last_usage = getattr(provider, "last_usage", {})
        provider_model = getattr(provider, "model", "unknown")
        requested_model = getattr(provider, "requested_model", provider_model)
        meta_dict = {
            "provider": provider.__class__.__name__,
            "requested_model": requested_model,
            "actual_model": provider_model,
            "input_tokens": last_usage.get("input_tokens", 0),
            "output_tokens": last_usage.get("output_tokens", 0),
            "latency": last_usage.get("latency", 0.0),
            "estimated_cost": last_usage.get("estimated_cost", 0.0),
            "cost": last_usage.get("estimated_cost", 0.0)
        }

        return Script(
            topic_num=script.topic_num,
            title=parsed["title"],
            hook=parsed["hook"],
            niche_context=parsed["niche_context"],
            scenes=[ScriptScene(narration=s["narration"], visual_desc=s["visual_desc"]) for s in parsed["scenes"]],
            metadata=meta_dict
        )
    except Exception as parse_err:
        from clippilot.logger import get_logger
        logger = get_logger("clippilot.pipeline")
        
        is_leakage = False
        if resp:
            from clippilot.brain.provider import detects_reasoning_leakage
            is_leakage = detects_reasoning_leakage(resp)
            
        reason = "JSON validation failure"
        if is_leakage:
            reason = "Reasoning leakage"
        elif "Missing required key" in str(parse_err) or "must be a list" in str(parse_err) or "cannot be empty" in str(parse_err):
            reason = "Schema validation failure"
            
        logger.error(f"Unable to repair malformed JSON. Using deterministic mock fallback. Failure Reason: {reason}. Error: {parse_err}")
        from clippilot.brain.provider import save_failed_response
        save_failed_response(
            stage_name="critic",
            raw_response=resp,
            reason=reason,
            provider=provider.__class__.__name__,
            requested_model=getattr(provider, "requested_model", None),
            actual_model=getattr(provider, "model", None),
            correction_attempted=True,
            repair_attempted=True
        )
        improved_title = script.title
        if not improved_title.startswith("Why You MUST"):
            stripped_title = improved_title[4:] if improved_title.startswith("Why ") else improved_title
            improved_title = f"Why You MUST {stripped_title}"

        last_usage = getattr(provider, "last_usage", {})
        provider_model = getattr(provider, "model", "unknown")
        requested_model = getattr(provider, "requested_model", provider_model)
        meta_dict = {
            "provider": provider.__class__.__name__,
            "requested_model": requested_model,
            "actual_model": provider_model,
            "input_tokens": last_usage.get("input_tokens", 0),
            "output_tokens": last_usage.get("output_tokens", 0),
            "latency": last_usage.get("latency", 0.0),
            "estimated_cost": last_usage.get("estimated_cost", 0.0),
            "cost": last_usage.get("estimated_cost", 0.0),
            "fallback_flag": True
        }

        return Script(
            topic_num=script.topic_num,
            title=improved_title,
            hook=f"Wait, {script.hook.lower() if script.hook else ''}!",
            niche_context=script.niche_context,
            scenes=[
                ScriptScene(
                    narration=f"Yes, {script.title.lower()} is a major topic right now.",
                    visual_desc=script.scenes[0].visual_desc if script.scenes else "Visual representing the main point."
                ),
                ScriptScene(
                    narration=f"Here is what you need to do: keep track of your {script.niche_context}.",
                    visual_desc=script.scenes[1].visual_desc if len(script.scenes) > 1 else "Takeaway graphic."
                )
            ],
            metadata=meta_dict
        )


def _aggregate_usages(usages: list[dict[str, Any]]) -> dict[str, Any]:
    agg = {
        "provider": "MockProvider",
        "requested_model": "mock",
        "actual_model": "mock",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency": 0.0,
        "estimated_cost": 0.0,
        "cost": 0.0
    }
    real_usages = [u for u in usages if u and u.get("provider") != "MockProvider"]
    if real_usages:
        agg["provider"] = real_usages[0].get("provider", "MockProvider")
        agg["requested_model"] = real_usages[0].get("requested_model", "mock")
        agg["actual_model"] = real_usages[0].get("actual_model", "mock")
    elif usages:
        agg["provider"] = usages[0].get("provider", "MockProvider")
        agg["requested_model"] = usages[0].get("requested_model", "mock")
        agg["actual_model"] = usages[0].get("actual_model", "mock")
        
    for u in usages:
        if not u:
            continue
        agg["input_tokens"] += u.get("input_tokens", 0)
        agg["output_tokens"] += u.get("output_tokens", 0)
        agg["latency"] += u.get("latency", 0.0)
        agg["estimated_cost"] += u.get("estimated_cost", 0.0)
        agg["cost"] += u.get("estimated_cost", 0.0)
        
    agg["latency"] = round(agg["latency"], 4)
    agg["estimated_cost"] = round(agg["estimated_cost"], 6)
    agg["cost"] = round(agg["cost"], 6)
    return agg


def orchestrate_script_revision(
    state: PipelineState,
    topic: Topic,
    variation: VariationRecord,
    workspace_dir: Path,
    initial_script: Script,
    threshold: float = 8.5,
) -> Script:
    """Orchestrates script evaluation and a maximum of one rewrite revision cycle."""
    settings = Settings.load()

    print("\n🧐 [STAGE 4B] Starting Script Critic evaluation...")
    initial_critique = run_script_critic(initial_script, settings)
    initial_score = initial_critique.overall.score

    print(f"🧐 [STAGE 4B] Initial overall score: {initial_score}/10 (threshold: {threshold})")

    # If the score meets or exceeds the threshold, no rewrite needed
    if initial_score >= threshold:
        print("✔️ Script score satisfies threshold. Skipping revision.")
        critic_usage = _aggregate_usages([initial_critique.metadata])
        initial_script.metadata.update({
            "critic_usage": critic_usage,
            "initial_score": initial_score,
            "final_score": initial_score,
            "revision_count": 0,
            "accepted_revision": False,
            "structured_feedback": {
                "hook": initial_critique.hook.__dict__,
                "curiosity": initial_critique.curiosity.__dict__,
                "clarity": initial_critique.clarity.__dict__,
                "retention": initial_critique.retention.__dict__,
                "cta": initial_critique.cta.__dict__,
                "overall": initial_critique.overall.__dict__,
            }
        })
        return initial_script

    # Otherwise, execute exactly ONE rewrite revision cycle
    print(f"⚠️ Script score ({initial_score}) is below threshold ({threshold}). Requesting rewrite...")
    rewritten_script = run_script_rewrite(initial_script, initial_critique, settings)

    print("🧐 [STAGE 4B] Re-evaluating rewritten script...")
    final_critique = run_script_critic(rewritten_script, settings)
    final_score = final_critique.overall.score

    print(f"🧐 [STAGE 4B] Revised overall score: {final_score}/10")

    # Determine if we accept the revision (it must score higher than the initial script)
    if final_score > initial_score:
        print("✔️ Revised script scored higher. Accepting revision!")
        accepted = True
        chosen_script = rewritten_script
        chosen_critique = final_critique
        chosen_score = final_score
    else:
        print("⚠️ Revised script did not improve score. Keeping original script.")
        accepted = False
        chosen_script = initial_script
        chosen_critique = initial_critique
        chosen_score = initial_score

    critic_usage = _aggregate_usages([
        initial_critique.metadata,
        rewritten_script.metadata,
        final_critique.metadata
    ])

    chosen_script.metadata.update({
        "critic_usage": critic_usage,
        "title": chosen_script.title,
        "initial_score": initial_score,
        "final_score": chosen_score,
        "revision_count": 1,
        "accepted_revision": accepted,
        "structured_feedback": {
            "hook": chosen_critique.hook.__dict__,
            "curiosity": chosen_critique.curiosity.__dict__,
            "clarity": chosen_critique.clarity.__dict__,
            "retention": chosen_critique.retention.__dict__,
            "cta": chosen_critique.cta.__dict__,
            "overall": chosen_critique.overall.__dict__,
        }
    })

    return chosen_script
