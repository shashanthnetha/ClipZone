# -*- coding: utf-8 -*-
"""ClipPilot Brain AI Director Engine.

Orchestrates the deterministic creation of the Creative Blueprint using
rules and mappings from clippilot.brain.director_rules.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from clippilot.brain.pipeline_orchestrator import Topic
from clippilot.brain.strategy_engine import StrategyDecision
from clippilot.brain.creative_models import CreativeBlueprint

import clippilot.brain.director_rules as rules

def make_creative_blueprint(
    topic: Topic,
    decision: Optional[StrategyDecision] = None,
    learning_summary: Optional[Any] = None,
    history: Optional[List[Any]] = None,
) -> CreativeBlueprint:
    """Orchestrates deterministic creative decisions into a unified CreativeBlueprint."""
    decision_reasoning = {}
    
    # 1. Audience
    audience, aud_reason = rules.get_audience(topic.niche)
    decision_reasoning["audience"] = aud_reason
    
    # 2. Target Duration and Target Scene Count
    target_duration = 60.0
    duration_reason = "Default target duration set to 60.0s (L3)."
    if decision and decision.variation:
        length_key = decision.variation.len or "L3"
        if length_key == "L1":
            target_duration = 30.0
            duration_reason = "Target duration set to 30.0s (L1) based on variation length."
        elif length_key == "L2":
            target_duration = 45.0
            duration_reason = "Target duration set to 45.0s (L2) based on variation length."
        else:
            duration_reason = "Target duration set to 60.0s (L3) based on variation length."
            
    decision_reasoning["target_duration"] = duration_reason
    
    target_scene_count = 5
    if target_duration <= 30.0:
        target_scene_count = 3
    elif target_duration <= 45.0:
        target_scene_count = 4
        
    # 3. Story Framework
    story_framework, fw_reason = rules.get_story_framework(topic.title, topic.angle or "")
    decision_reasoning["story_framework"] = fw_reason
    
    # 4. Emotion Curve
    emotion_curve, curve_reason = rules.get_emotion_curve(story_framework.name)
    decision_reasoning["emotion_curve"] = curve_reason
    
    # 5. Pacing
    variation_pace = decision.variation.pace if (decision and decision.variation) else None
    pacing, pace_reason = rules.get_pacing_strategy(variation_pace, learning_summary)
    decision_reasoning["pacing"] = pace_reason
    
    # 6. CTA Strategy
    format_str = decision.format if decision else None
    cta_strategy, cta_reason = rules.get_cta_strategy(format_str)
    decision_reasoning["cta_strategy"] = cta_reason
    
    # 7. Hook Strategy
    selected_hook = decision.hook if decision else None
    hook_strategy, hook_reason = rules.get_hook_strategy(selected_hook, learning_summary)
    decision_reasoning["hook_strategy"] = hook_reason
    
    # 8. Visual & Caption Styles
    skin_str = decision.skin if decision else None
    visual_style, caption_style, style_reason = rules.get_visual_and_caption_style(skin_str)
    decision_reasoning["visual_style"] = style_reason
    decision_reasoning["caption_style"] = style_reason

    # Visual Language settings
    transition_style = rules.get_transition_style(skin_str)
    animation_style = rules.get_animation_style(skin_str)
    camera_language = rules.get_camera_language(skin_str)
    emphasis_color = rules.get_emphasis_color(skin_str)
    visual_pacing = "fast" if pacing == "Fast/Dynamic" else "steady"
    
    # 9. Primary Goal
    primary_goal = "Educate the audience on topic mechanisms and maximize viewer retention rate."
    
    # Compile metadata
    metadata = {
        "decision_reasoning": decision_reasoning,
        "creative_version": "1.0"
    }
    
    tone = "Informative & Professional"
    if "Listicle" in story_framework.name:
        tone = "Informative & Professional"
    elif "Problem-Solution" in story_framework.name:
        tone = "Empathetic & Authoritative"
    elif "Myth-Busting" in story_framework.name:
        tone = "Authoritative & Action-Oriented"
    
    return CreativeBlueprint(
        audience=audience,
        target_duration=target_duration,
        hook_strategy=hook_strategy,
        story_framework=story_framework,
        emotion_curve=emotion_curve,
        pacing=pacing,
        tone=tone,
        cta_strategy=cta_strategy,
        caption_style=caption_style,
        visual_style=visual_style,
        target_scene_count=target_scene_count,
        primary_goal=primary_goal,
        transition_style=transition_style,
        animation_style=animation_style,
        camera_language=camera_language,
        emphasis_color=emphasis_color,
        visual_pacing=visual_pacing,
        metadata=metadata
    )
