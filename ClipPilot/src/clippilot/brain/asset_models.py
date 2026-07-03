# -*- coding: utf-8 -*-
"""Asset models representing structured visual asset plans for video generation."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AssetReference:
    """Represents a reference to a specific media asset from any provider."""
    asset_type: str  # video, image, icon, audio, sfx, font, lottie
    provider: str    # pexels, pixabay, unsplash, icons8, svgrepo, lottiefiles, local, mock
    search_query: str
    local_path: str
    priority: int = 1
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneAssetPlan:
    """Defines all visual/motion assets required to render a single scene."""
    scene_number: int
    background: str
    stock_video_queries: List[str] = field(default_factory=list)
    image_queries: List[str] = field(default_factory=list)
    icon_queries: List[str] = field(default_factory=list)
    chart_type: str = ""
    overlay_text: str = ""
    animations: List[str] = field(default_factory=list)
    transitions: List[str] = field(default_factory=list)
    fallback_assets: List[AssetReference] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_number": self.scene_number,
            "background": self.background,
            "stock_video_queries": self.stock_video_queries,
            "image_queries": self.image_queries,
            "icon_queries": self.icon_queries,
            "chart_type": self.chart_type,
            "overlay_text": self.overlay_text,
            "animations": self.animations,
            "transitions": self.transitions,
            "fallback_assets": [f.to_dict() for f in self.fallback_assets]
        }


@dataclass
class VideoAssetPlan:
    """Complete video-level collection of scene asset plans."""
    scenes: List[SceneAssetPlan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenes": [s.to_dict() for s in self.scenes],
            "metadata": self.metadata
        }
