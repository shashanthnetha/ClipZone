# -*- coding: utf-8 -*-
"""ClipPilot Brain Scene Quality.

Calculates a deterministic quality score for each scene and provides
a single deterministic repair capability to fix any score below threshold.
"""
from __future__ import annotations

import logging
from typing import Optional
from clippilot.brain.creative_models import SceneBlueprint

logger = logging.getLogger("clippilot.scene_quality")

def calculate_scene_quality_score(
    scene: SceneBlueprint,
    framework_name: str,
    total_scenes: int,
    prev_scene: Optional[SceneBlueprint] = None
) -> float:
    """Calculates a deterministic quality score out of 100.0 for a SceneBlueprint."""
    score = 100.0

    # 1. Pacing Consistency (Safe Duration Bounds)
    if scene.duration < 1.5 or scene.duration > 20.0:
        score -= 25.0

    # 2. Retention Flow (Stage Matching Position)
    if scene.scene_number == 1 and scene.retention_stage != "Hook":
        score -= 20.0
    if scene.scene_number == total_scenes and scene.retention_stage != "CTA":
        score -= 20.0

    # 3. Emotional Progression
    if scene.scene_number == 1 and scene.emotional_intensity == "Low":
        score -= 15.0

    # 4. Scene Diversity & Visual Variety (Adjacent duplicates check)
    if prev_scene:
        if scene.visual_priority == prev_scene.visual_priority:
            score -= 15.0
        if scene.visual_style == prev_scene.visual_style and total_scenes > 1:
            # Although visual style might be globally defined, consecutive duplicate priority is a variety issue
            pass

    return max(0.0, score)

def repair_scene_blueprint(
    scene: SceneBlueprint,
    framework_name: str,
    total_scenes: int,
    prev_scene: Optional[SceneBlueprint] = None
) -> SceneBlueprint:
    """Deterministically repairs deficient fields in a SceneBlueprint to improve score."""
    # Create a repaired copy
    repaired = SceneBlueprint(
        scene_number=scene.scene_number,
        duration=scene.duration,
        scene_goal=scene.scene_goal,
        visual_intent=scene.visual_intent,
        visual_priority=scene.visual_priority,
        narration_purpose=scene.narration_purpose,
        emotional_intensity=scene.emotional_intensity,
        retention_stage=scene.retention_stage,
        visual_style=scene.visual_style,
        animation_style=scene.animation_style,
        transition=scene.transition,
        camera_language=scene.camera_language,
        caption_behavior=scene.caption_behavior,
        search_queries=list(scene.search_queries),
        fallback_icon=scene.fallback_icon,
        primary_visual=scene.primary_visual,
        scene_quality_score=scene.scene_quality_score
    )

    # 1. Repair duration
    if repaired.duration < 1.5:
        repaired.duration = 1.5
    elif repaired.duration > 20.0:
        repaired.duration = 10.0

    # 2. Repair retention stage & emotional intensity for boundary scenes
    if repaired.scene_number == 1:
        repaired.retention_stage = "Hook"
        repaired.emotional_intensity = "High"
        repaired.visual_priority = "stock_video"
    elif repaired.scene_number == total_scenes:
        repaired.retention_stage = "CTA"
        repaired.emotional_intensity = "Medium"
        repaired.visual_priority = "text_only"

    # 3. Repair visual priority clash with previous scene
    if prev_scene and repaired.visual_priority == prev_scene.visual_priority:
        priorities = ["stock_video", "stock_image", "motion_graphic", "icon", "text_only"]
        # Find next available priority that is not the previous one
        for p in priorities:
            if p != prev_scene.visual_priority:
                if repaired.scene_number == total_scenes and p != "text_only":
                    # Keep CTA text-only if possible, or pick another
                    continue
                repaired.visual_priority = p
                break

    return repaired
