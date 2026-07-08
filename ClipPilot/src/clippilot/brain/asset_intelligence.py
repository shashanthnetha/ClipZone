# -*- coding: utf-8 -*-
"""Asset Intelligence Engine for ClipPilot.

Deduces visual and transition assets required for video generation.
Uses existing LLM providers if available, or falls back to a realistic,
deterministic asset planner otherwise.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional
from clippilot.brain.script_generator import Script
from clippilot.brain.asset_models import VideoAssetPlan, SceneAssetPlan, AssetReference
from clippilot.config import Settings
from clippilot.brain.provider import get_provider
from clippilot.logger import get_logger
from clippilot.brain.creative_models import CreativeBlueprint, SceneBlueprint

logger = get_logger("clippilot.assets")

ASSET_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_number": {"type": "integer"},
                    "background": {"type": "string"},
                    "stock_video_queries": {"type": "array", "items": {"type": "string"}},
                    "image_queries": {"type": "array", "items": {"type": "string"}},
                    "icon_queries": {"type": "array", "items": {"type": "string"}},
                    "chart_type": {"type": "string"},
                    "overlay_text": {"type": "string"},
                    "animations": {"type": "array", "items": {"type": "string"}},
                    "transitions": {"type": "array", "items": {"type": "string"}},
                    "fallback_assets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "asset_type": {"type": "string"},
                                "provider": {"type": "string"},
                                "search_query": {"type": "string"},
                                "local_path": {"type": "string"},
                                "priority": {"type": "integer"},
                                "required": {"type": "boolean"}
                            },
                            "required": ["asset_type", "provider", "search_query", "local_path", "priority", "required"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": [
                    "scene_number", "background", "stock_video_queries", "image_queries", 
                    "icon_queries", "chart_type", "overlay_text", "animations", "transitions", "fallback_assets"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["scenes"],
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


class AssetIntelligenceEngine:
    """Intelligent engine for analyzing scripts and planning video visual assets."""

    def __init__(self) -> None:
        pass

    def build_asset_plan(
        self,
        script: Script,
        render_tree: Any,  # ScenePlan or RenderTree
        settings: Optional[Settings] = None,
        creative_blueprint: Optional[CreativeBlueprint] = None,
        scene_blueprints: Optional[list[SceneBlueprint]] = None
    ) -> VideoAssetPlan:
        """Determines background, stock footage queries, icons, charts, and transitions for every scene."""
        if settings is None:
            settings = Settings.load()

        if not script.scenes:
            return VideoAssetPlan(scenes=[], metadata={"provider": "mock", "model": "none"})

        # Check for provider API keys
        has_api_key = bool(
            os.environ.get("LLM_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or getattr(settings, "llm_api_key", None)
        )

        if not has_api_key:
            return self._build_deterministic_plan(script, render_tree, creative_blueprint, scene_blueprints)

        # 1. Prepare Prompt
        system_prompt = (
            "You are the autonomous Visual Asset Planner. Your job is to output exactly one validated "
            "Asset Plan for a video script as a raw JSON object complying with this exact schema:\n"
            "{\n"
            "  \"scenes\": [\n"
            "    {\n"
            "      \"scene_number\": 1,\n"
            "      \"background\": \"dark_finance\",\n"
            "      \"stock_video_queries\": [\"credit card payment\"],\n"
            "      \"image_queries\": [\"credit card swipe\"],\n"
            "      \"icon_queries\": [\"credit-card\", \"warning\", \"bank\"],\n"
            "      \"chart_type\": \"credit score\",\n"
            "      \"overlay_text\": \"Why credit scores matter\",\n"
            "      \"animations\": [\"fade-in\"],\n"
            "      \"transitions\": [\"swipe\"],\n"
            "      \"fallback_assets\": [\n"
            "        {\n"
            "          \"asset_type\": \"image\",\n"
            "          \"provider\": \"mock\",\n"
            "          \"search_query\": \"credit card\",\n"
            "          \"local_path\": \"assets/fallback.png\",\n"
            "          \"priority\": 1,\n"
            "          \"required\": true\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Do NOT return markdown code fences. Return ONLY raw JSON."
        )

        user_prompt = f"Topic Title: {script.title}\nScript Niche: {script.niche_context}\n"
        for i, s in enumerate(script.scenes):
            user_prompt += f"Scene {i+1}: Narration: {s.narration} | Visual Description: {s.visual_desc}\n"

        if scene_blueprints:
            user_prompt += "\nUse the following detailed Scene Blueprints as the single source of truth for planning assets:\n"
            for bp in scene_blueprints:
                user_prompt += (
                    f"Scene {bp.scene_number}:\n"
                    f"- Retention Stage: {bp.retention_stage}\n"
                    f"- Scene Goal: {bp.scene_goal}\n"
                    f"- Visual Intent: {bp.visual_intent}\n"
                    f"- Visual Priority: {bp.visual_priority}\n"
                    f"Please generate the 'stock_video_queries', 'image_queries', and 'icon_queries' strictly derived from 'Visual Intent' (Flow: Scene Goal -> Visual Intent -> Search Queries).\n"
                )

        # 2. Call LLM
        provider_called_successfully = False
        try:
            from clippilot.brain.env import has_api_key
            if not has_api_key():
                raise ValueError("No API key configured")
            models = settings.asset_models
            if not models:
                models = [settings.asset_model] if settings.asset_model else ([settings.llm_model] if settings.llm_model else [])
            provider = get_provider(settings, models=models)
            raw_response = provider.generate_text(prompt=user_prompt, system_prompt=system_prompt, json_schema=ASSET_SCHEMA)
            provider_called_successfully = True
        except Exception as e:
            print(f"⚠️ Asset Intelligence Provider call failed: {e}. Falling back to deterministic plan.")
            return self._build_deterministic_plan(script, render_tree, creative_blueprint, scene_blueprints)

        # 3. Parse JSON (only if provider succeeded)
        try:
            from clippilot.brain.provider import tolerant_json_loads
            parsed = tolerant_json_loads(raw_response)
            scenes_plan = []
            for sp in parsed.get("scenes", []):
                fallback_refs = [
                    AssetReference(
                        asset_type=fr.get("asset_type", "image"),
                        provider=fr.get("provider", "mock"),
                        search_query=fr.get("search_query", ""),
                        local_path=fr.get("local_path", ""),
                        priority=fr.get("priority", 1),
                        required=fr.get("required", True)
                    )
                    for fr in sp.get("fallback_assets", [])
                ]
                scenes_plan.append(
                    SceneAssetPlan(
                        scene_number=sp.get("scene_number", 1),
                        background=sp.get("background", "dark_finance"),
                        stock_video_queries=sp.get("stock_video_queries", []),
                        image_queries=sp.get("image_queries", []),
                        icon_queries=sp.get("icon_queries", []),
                        chart_type=sp.get("chart_type", ""),
                        overlay_text=sp.get("overlay_text", ""),
                        animations=sp.get("animations", []),
                        transitions=sp.get("transitions", []),
                        fallback_assets=fallback_refs
                    )
                )

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

            return VideoAssetPlan(
                scenes=scenes_plan,
                metadata=meta_dict
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
            elif isinstance(parse_err, (KeyError, AttributeError, TypeError)):
                reason = "Schema validation failure"
                
            logger.error(f"Unable to repair malformed JSON. Using deterministic mock fallback. Failure Reason: {reason}. Error: {parse_err}")
            from clippilot.brain.provider import save_failed_response
            save_failed_response(
                stage_name="asset",
                raw_response=raw_response,
                reason=reason,
                provider=provider.__class__.__name__ if "provider" in locals() else "unknown",
                requested_model=getattr(provider, "requested_model", None) if "provider" in locals() else None,
                actual_model=getattr(provider, "model", None) if "provider" in locals() else None,
                correction_attempted=True,
                repair_attempted=True,
                schema_validation_status="failed" if reason == "Schema validation failure" else "not_applicable"
            )
            
            last_usage = getattr(provider, "last_usage", {}) if "provider" in locals() else {}
            provider_model = getattr(provider, "model", "unknown") if "provider" in locals() else "unknown"
            requested_model = getattr(provider, "requested_model", provider_model) if "provider" in locals() else "unknown"
            meta_dict = {
                "provider": provider.__class__.__name__ if "provider" in locals() else "MockProvider",
                "requested_model": requested_model,
                "actual_model": provider_model,
                "input_tokens": last_usage.get("input_tokens", 0),
                "output_tokens": last_usage.get("output_tokens", 0),
                "latency": last_usage.get("latency", 0.0),
                "estimated_cost": last_usage.get("estimated_cost", 0.0),
                "cost": last_usage.get("estimated_cost", 0.0),
                "fallback_flag": True
            }
            plan = self._build_deterministic_plan(script, render_tree, creative_blueprint, scene_blueprints)
            plan.metadata = meta_dict
            return plan

    def _build_deterministic_plan(
        self,
        script: Script,
        render_tree: Any,
        creative_blueprint: Optional[CreativeBlueprint] = None,
        scene_blueprints: Optional[list[SceneBlueprint]] = None
    ) -> VideoAssetPlan:
        """Deterministically extracts background style, stock/icon queries, and motion elements."""
        scenes_plan = []

        # Derive background from render_tree background if available
        bg_id = "dark_finance"
        if render_tree and hasattr(render_tree, "scenes") and render_tree.scenes:
            first_scene = render_tree.scenes[0]
            if hasattr(first_scene, "background_id") and first_scene.background_id:
                bg_id = first_scene.background_id

        for idx, scene in enumerate(script.scenes):
            scene_num = idx + 1
            narr_lower = scene.narration.lower()

            if scene_blueprints and idx < len(scene_blueprints):
                scene_blueprint = scene_blueprints[idx]
                
                # Derive Queries Directly from visual_intent/search_queries of SceneBlueprint
                stock_queries = list(scene_blueprint.search_queries)
                bg = bg_id
                bg_lower = scene_blueprint.visual_style.lower()
                if "paper" in bg_lower or "warm" in bg_lower:
                    bg = "warm_paper"
                elif "neon" in bg_lower or "night" in bg_lower:
                    bg = "neon_vignette"
                elif "blueprint" in bg_lower:
                    bg = "blueprint_grid"
                elif "minimal" in bg_lower:
                    bg = "near_black"
                
                icon_queries = [scene_blueprint.fallback_icon.replace(".png", "")]
                chart_type = scene_blueprint.primary_visual
                animations = [scene_blueprint.animation_style]
                transitions = [scene_blueprint.transition]
            else:
                # 1. Deterministic background styling
                if "paper" in bg_id or "warm" in bg_id:
                    bg = "warm_paper"
                elif "neon" in bg_id or "night" in bg_id:
                    bg = "neon_vignette"
                elif "blueprint" in bg_id:
                    bg = "blueprint_grid"
                else:
                    bg = bg_id

                # 2. Deterministic stock video queries
                stock_queries = []
                if "credit" in narr_lower or "card" in narr_lower:
                    stock_queries.append("credit card payment")
                elif "phone" in narr_lower or "bill" in narr_lower:
                    stock_queries.append("smartphone bill payment")
                elif "free trial" in narr_lower or "subscription" in narr_lower:
                    stock_queries.append("calendar free trial subscription")
                elif "close" in narr_lower or "closing" in narr_lower:
                    stock_queries.append("closing bank account")
                elif "score" in narr_lower or "rating" in narr_lower:
                    stock_queries.append("credit score rating")
                else:
                    stock_queries.append("financial analytics graph")

                # 3. Deterministic icon queries
                icon_queries = []
                if "credit" in narr_lower or "card" in narr_lower:
                    icon_queries.append("credit-card")
                if "bank" in narr_lower or "account" in narr_lower:
                    icon_queries.append("bank")
                if any(w in narr_lower for w in ["warning", "hurt", "fail", "lose", "charging"]):
                    icon_queries.append("warning")
                if not icon_queries:
                    icon_queries = ["trending-up", "dollar-sign"]

                # 4. Deterministic chart type
                chart_type = ""
                if "score" in narr_lower:
                    chart_type = "credit score"
                elif any(w in narr_lower for w in ["trend", "analytics", "rise", "drop"]):
                    chart_type = "line chart"
                    
                animations = ["fade-in"]
                transitions = ["swipe"]

            image_queries = [f"{q} close up" for q in stock_queries]

            # 6. Fallback assets mapping
            fallback_refs = [
                AssetReference(
                    asset_type="image",
                    provider="mock",
                    search_query=query,
                    local_path=f"assets/graphics/fallback_{query.replace(' ', '_')}.png",
                    priority=1,
                    required=True
                )
                for query in image_queries
            ]

            scenes_plan.append(
                SceneAssetPlan(
                    scene_number=scene_num,
                    background=bg,
                    stock_video_queries=stock_queries,
                    image_queries=image_queries,
                    icon_queries=icon_queries,
                    chart_type=chart_type,
                    overlay_text=scene.narration[:30] + "..." if len(scene.narration) > 30 else scene.narration,
                    animations=animations,
                    transitions=transitions,
                    fallback_assets=fallback_refs
                )
            )

        return VideoAssetPlan(
            scenes=scenes_plan,
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
