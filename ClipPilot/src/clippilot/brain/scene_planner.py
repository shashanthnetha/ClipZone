# -*- coding: utf-8 -*-
"""ClipPilot Brain Scene Planner.

Deterministically transforms generated scripts and variation logs into structured,
time-indexed, and asset-bound visual/audio scene plans.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from clippilot.brain.pipeline_orchestrator import VariationRecord
from clippilot.brain.script_generator import Script
from clippilot.brain.creative_models import SceneBlueprint


@dataclass
class ScenePlanScene:
    """Represents a time-indexed, asset-bound segment of the video pipeline."""
    scene_index: int
    narration: str
    duration_seconds: float
    subtitle_words: list[str] = field(default_factory=list)
    subtitle_timings: list[tuple[float, float]] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)
    animation_ids: list[str] = field(default_factory=list)
    transition_ids: list[str] = field(default_factory=list)
    sfx_ids: list[str] = field(default_factory=list)
    background_id: str = ""
    chart_id: str = ""
    camera_language: str = ""
    caption_behavior: str = ""


@dataclass
class ScenePlan:
    """Represents the complete blueprint for video rendering."""
    topic_num: str
    title: str
    hook: str
    skin_id: str
    format_id: str
    voice_id: str
    scenes: list[ScenePlanScene] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    caption_style: str = ""
    emphasis_color: str = ""
    visual_style: str = ""


def plan_scenes(
    script: Script,
    variation: VariationRecord,
    scene_blueprints: Optional[list[SceneBlueprint]] = None
) -> ScenePlan:
    """Deterministically maps a script and its variation configs into a ScenePlan."""
    # 1. Niche skin configuration mappings
    skin_id = variation.skin
    format_id = variation.fmt

    # Map S-id to visual asset, animation, background, and chart signatures
    bg_map = {
        "S1": "dark_radial",
        "S2": "blueprint_grid",
        "S3": "cream_pop",
        "S4": "near_black",
        "S5": "warm_paper",
        "S6": "neon_vignette",
    }
    background_id = bg_map.get(skin_id, "dark_radial")

    chart_map = {
        "S1": "ScoreDial",
        "S2": "schematic_line",
        "S3": "flat_illustrations",
        "S4": "dashboard_split",
        "S5": "marker_props",
        "S6": "glass_cards",
    }
    chart_id = chart_map.get(skin_id, "ScoreDial")

    anim_map = {
        "S1": ["spring", "counter_roll"],
        "S2": ["draw_path", "bounce"],
        "S3": ["bounce", "overshoot"],
        "S4": ["counter_roll", "glow_pulse"],
        "S5": ["snappy_drop", "paper_flip"],
        "S6": ["glow_pulse", "float_parallax"],
    }
    animations = anim_map.get(skin_id, ["spring"])

    transition_map = {
        "S2": ["reveal_wipe"],
        "S5": ["hardcut"],
    }
    transitions = transition_map.get(skin_id, ["crossfade"])

    # 2. Process each scene
    planned_scenes: list[ScenePlanScene] = []
    total_duration = 0.0

    for idx, scene in enumerate(script.scenes):
        scene_idx = idx + 1
        narration = scene.narration

        # If Scene Blueprint is provided, override duration/transitions/animations/etc
        if scene_blueprints and idx < len(scene_blueprints):
            scene_blueprint = scene_blueprints[idx]
            duration = scene_blueprint.duration
            
            # Map visual style representation to background_id key
            bg_lower = scene_blueprint.visual_style.lower()
            if "paper" in bg_lower or "warm" in bg_lower:
                bg = "warm_paper"
            elif "neon" in bg_lower or "night" in bg_lower:
                bg = "neon_vignette"
            elif "blueprint" in bg_lower:
                bg = "blueprint_grid"
            elif "minimal" in bg_lower:
                bg = "near_black"
            else:
                bg = background_id
                
            anims = [scene_blueprint.animation_style]
            trans = [scene_blueprint.transition]
            camera_lang = scene_blueprint.camera_language
            caption_beh = scene_blueprint.caption_behavior
            assets = [f"asset_{skin_id.lower()}_{scene_idx}"]
            
            # If visual priority matches list or split columns
            if format_id == "F3" or scene_blueprint.visual_priority == "motion_graphic":
                assets.append(f"list_item_{scene_idx}")
            elif format_id == "F5":
                assets.append("split_column")
        else:
            # Fallback to estimating duration: average 2.5 words per second, minimum 1.5 seconds
            words = narration.split()
            duration = max(1.5, len(words) / 2.5)
            bg = background_id
            anims = animations
            trans = transitions
            camera_lang = ""
            caption_beh = ""
            assets = [f"asset_{skin_id.lower()}_{scene_idx}"]
            if format_id == "F3":
                assets.append(f"list_item_{scene_idx}")
            elif format_id == "F5":
                assets.append("split_column")

        words = narration.split()
        num_words = len(words)
        # Estimate subtitle word-by-word timings
        timings: list[tuple[float, float]] = []
        if num_words > 0:
            step = duration / num_words
            for i in range(num_words):
                start = i * step
                end = (i + 1) * step
                timings.append((round(start, 2), round(end, 2)))

        # Assign SFX: first scene gets whoosh, last gets ding, others get pop
        if scene_idx == 1:
            sfx = ["whoosh"]
        elif scene_idx == len(script.scenes):
            sfx = ["ding"]
        else:
            sfx = ["pop"]

        planned_scenes.append(
            ScenePlanScene(
                scene_index=scene_idx,
                narration=narration,
                duration_seconds=round(duration, 2),
                subtitle_words=words,
                subtitle_timings=timings,
                asset_ids=assets,
                animation_ids=anims,
                transition_ids=trans,
                sfx_ids=sfx,
                background_id=bg,
                chart_id=chart_id,
                camera_language=camera_lang,
                caption_behavior=caption_beh
            )
        )
        total_duration += duration

    # Extract global properties if blueprints are available
    caption_style = ""
    emphasis_color = ""
    visual_style_str = ""
    if scene_blueprints and len(scene_blueprints) > 0:
        # Resolve from variation skin settings dynamically using rules
        from clippilot.brain.director_rules import get_visual_and_caption_style, get_emphasis_color
        _, caption_style, _ = get_visual_and_caption_style(skin_id)
        emphasis_color = get_emphasis_color(skin_id)
        visual_style_str = bg_map.get(skin_id, "dark_radial")

    return ScenePlan(
        topic_num=script.topic_num,
        title=script.title,
        hook=script.hook,
        skin_id=skin_id,
        format_id=format_id,
        voice_id=variation.voice,
        scenes=planned_scenes,
        total_duration_seconds=round(total_duration, 2),
        caption_style=caption_style,
        emphasis_color=emphasis_color,
        visual_style=visual_style_str
    )
