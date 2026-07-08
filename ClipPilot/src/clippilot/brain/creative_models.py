# -*- coding: utf-8 -*-
"""ClipPilot Brain Creative Models.

Defines the structure of the Creative Blueprint, Story Frameworks,
Scene Blueprint, and other deterministic director-level creative decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

@dataclass
class StoryFramework:
    name: str  # e.g., "Problem-Solution", "AIDA", "Myth-Busting", "Listicle/Three-Facts"
    description: str

@dataclass
class EmotionCurve:
    pattern: str  # e.g., "Hook-Dip-Rise"
    description: str

@dataclass
class RetentionStrategy:
    hook_type: str
    re_hook_interval_seconds: float
    visual_pattern: str

@dataclass
class CreativeBlueprint:
    audience: str
    target_duration: float
    hook_strategy: str
    story_framework: StoryFramework
    emotion_curve: EmotionCurve
    pacing: str
    tone: str
    cta_strategy: str
    caption_style: str
    visual_style: str
    target_scene_count: int
    primary_goal: str
    transition_style: str = ""
    animation_style: str = ""
    camera_language: str = ""
    emphasis_color: str = ""
    visual_pacing: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the blueprint to a serializable dictionary representation."""
        return asdict(self)

@dataclass
class SceneBlueprint:
    scene_number: int
    duration: float
    scene_goal: str
    visual_intent: str
    visual_priority: str  # stock_video, stock_image, motion_graphic, icon, text_only
    narration_purpose: str
    emotional_intensity: str  # High, Medium, Low
    retention_stage: str  # Hook, Build, Rehook, Reveal, Payoff, CTA
    visual_style: str
    animation_style: str
    transition: str
    camera_language: str
    caption_behavior: str
    search_queries: List[str]
    fallback_icon: str
    primary_visual: str
    scene_quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the scene blueprint to a serializable dictionary representation."""
        return asdict(self)
