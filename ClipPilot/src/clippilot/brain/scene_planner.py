# -*- coding: utf-8 -*-
"""ClipPilot Brain Scene Planner.

Deterministically transforms generated scripts and variation logs into structured,
time-indexed, and asset-bound visual/audio scene plans.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clippilot.brain.pipeline_orchestrator import VariationRecord
from clippilot.brain.script_generator import Script


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


def plan_scenes(script: Script, variation: VariationRecord) -> ScenePlan:
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

        # Estimate duration: average 2.5 words per second, minimum 1.5 seconds
        words = narration.split()
        num_words = len(words)
        duration = max(1.5, num_words / 2.5)

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

        # Determine visual asset identifiers based on skin/format
        assets = [f"asset_{skin_id.lower()}_{scene_idx}"]
        if format_id == "F3":  # Listicle
            assets.append(f"list_item_{scene_idx}")
        elif format_id == "F5":  # Comparison
            assets.append("split_column")

        planned_scenes.append(
            ScenePlanScene(
                scene_index=scene_idx,
                narration=narration,
                duration_seconds=round(duration, 2),
                subtitle_words=words,
                subtitle_timings=timings,
                asset_ids=assets,
                animation_ids=animations,
                transition_ids=transitions,
                sfx_ids=sfx,
                background_id=background_id,
                chart_id=chart_id,
            )
        )
        total_duration += duration

    return ScenePlan(
        topic_num=script.topic_num,
        title=script.title,
        hook=script.hook,
        skin_id=skin_id,
        format_id=format_id,
        voice_id=variation.voice,
        scenes=planned_scenes,
        total_duration_seconds=round(total_duration, 2),
    )
