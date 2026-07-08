# -*- coding: utf-8 -*-
"""ClipPilot Brain Prompt Builder.

Translates CreativeBlueprint decisions into clear instructions/constraints
for LLM prompt ingestion. This keeps creative logic separate from prompt engineering.
"""
from __future__ import annotations

from clippilot.brain.creative_models import CreativeBlueprint

def build_blueprint_instructions(blueprint: CreativeBlueprint) -> str:
    """Formats the CreativeBlueprint decisions into structured prompt instructions."""
    return (
        "STRICT ADHERENCE TO THE FOLLOWING CREATIVE BLUEPRINT IS REQUIRED:\n"
        f"- Target Audience: {blueprint.audience}\n"
        f"- Target Duration: {blueprint.target_duration}s (Limit spoken narration content to fit this length; target scene count is exactly {blueprint.target_scene_count} scenes).\n"
        f"- Story Framework: Use the '{blueprint.story_framework.name}' framework. Structure: {blueprint.story_framework.description}\n"
        f"- Hook Strategy: {blueprint.hook_strategy}\n"
        f"- Emotion Curve: Follow the '{blueprint.emotion_curve.pattern}' progression. Flow: {blueprint.emotion_curve.description}\n"
        f"- Pacing: {blueprint.pacing} pacing. Deliver script beats matching this style.\n"
        f"- Tone: {blueprint.tone} tone throughout narration. Avoid deviations.\n"
        f"- Call-To-Action (CTA): {blueprint.cta_strategy}\n"
        f"- Caption Styling: {blueprint.caption_style}\n"
        f"- Visual Skin Direction: {blueprint.visual_style}\n"
        f"- Primary Goal: {blueprint.primary_goal}\n"
    )
