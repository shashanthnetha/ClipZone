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


class AssetIntelligenceEngine:
    """Intelligent engine for analyzing scripts and planning video visual assets."""

    def __init__(self) -> None:
        pass

    def build_asset_plan(
        self,
        script: Script,
        render_tree: Any,  # ScenePlan or RenderTree
        settings: Optional[Settings] = None
    ) -> VideoAssetPlan:
        """Determines background, stock footage queries, icons, charts, and transitions for every scene."""
        if settings is None:
            settings = Settings.load()

        # Check for provider API keys
        has_api_key = bool(
            os.environ.get("LLM_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or getattr(settings, "llm_api_key", None)
        )

        if not has_api_key:
            return self._build_deterministic_plan(script, render_tree)

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

        # 2. Call LLM
        try:
            provider = get_provider(settings)
            raw_response = provider.generate_text(prompt=user_prompt, system_prompt=system_prompt)
            # Remove potential JSON markdown fences
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
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

            return VideoAssetPlan(
                scenes=scenes_plan,
                metadata={"provider": provider.__class__.__name__, "model": getattr(provider, "model", "unknown")}
            )

        except Exception as e:
            print(f"⚠️ Asset Intelligence Provider call failed: {e}. Falling back to deterministic plan.")
            return self._build_deterministic_plan(script, render_tree)

    def _build_deterministic_plan(self, script: Script, render_tree: Any) -> VideoAssetPlan:
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

            # 3. Deterministic image queries
            image_queries = [f"{q} close up" for q in stock_queries]

            # 4. Deterministic icon queries
            icon_queries = []
            if "credit" in narr_lower or "card" in narr_lower:
                icon_queries.append("credit-card")
            if "bank" in narr_lower or "account" in narr_lower:
                icon_queries.append("bank")
            if any(w in narr_lower for w in ["warning", "hurt", "fail", "lose", "charging"]):
                icon_queries.append("warning")
            if not icon_queries:
                icon_queries = ["trending-up", "dollar-sign"]

            # 5. Deterministic chart type
            chart_type = ""
            if "score" in narr_lower:
                chart_type = "credit score"
            elif any(w in narr_lower for w in ["trend", "analytics", "rise", "drop"]):
                chart_type = "line chart"

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
                    animations=["fade-in"],
                    transitions=["swipe"],
                    fallback_assets=fallback_refs
                )
            )

        return VideoAssetPlan(
            scenes=scenes_plan,
            metadata={"provider": "mock", "model": "deterministic"}
        )
