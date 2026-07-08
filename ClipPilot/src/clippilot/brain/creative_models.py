# -*- coding: utf-8 -*-
"""ClipPilot Brain Creative Models.

Defines the structure of the Creative Blueprint, Story Frameworks,
and other deterministic director-level creative decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict

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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the blueprint to a serializable dictionary representation."""
        return asdict(self)
