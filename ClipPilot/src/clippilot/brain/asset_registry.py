# -*- coding: utf-8 -*-
"""ClipPilot Brain Asset Registry.

Provides typed asset models and a unified registry to register, resolve,
and validate visual and audio assets with fallback defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Asset:
    """Base structural class representing a layout or media asset."""
    asset_id: str
    category: str


@dataclass
class BackgroundAsset(Asset):
    """Visual background style asset."""
    bg_type: str = "dark_radial"
    source_path: Optional[str] = None


@dataclass
class ImageAsset(Asset):
    """Static visual image asset."""
    source_path: str = ""


@dataclass
class VideoAsset(Asset):
    """Dynamic motion video segment asset."""
    source_path: str = ""


@dataclass
class ChartAsset(Asset):
    """Visual chart/graph layout asset."""
    chart_type: str = "ScoreDial"


@dataclass
class AnimationAsset(Asset):
    """Motion animation definition asset."""
    animation_type: str = "spring"
    default_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioAsset(Asset):
    """Audio narration or sound effects asset."""
    source_path: str = ""
    is_voiceover: bool = False


@dataclass
class FontAsset(Asset):
    """Text font family layout asset."""
    font_family: str = "PP Neue Montreal"
    source_url: Optional[str] = None


@dataclass
class TransitionAsset(Asset):
    """Visual transition layout asset."""
    transition_type: str = "crossfade"
    default_duration_frames: int = 15


@dataclass
class IconAsset(Asset):
    """Visual vector/icon asset."""
    icon_name: str = ""
    svg_path: Optional[str] = None


class AssetRegistry:
    """Unified register and resolver for all video pipeline assets."""

    def __init__(self) -> None:
        self._assets: dict[str, Asset] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        # Seed default backgrounds
        for bg in ["dark_radial", "blueprint_grid", "cream_pop", "near_black", "warm_paper", "neon_vignette"]:
            self.register(BackgroundAsset(bg, "backgrounds", bg))

        # Seed default charts
        for chart in ["ScoreDial", "schematic_line", "flat_illustrations", "dashboard_split", "marker_props", "glass_cards"]:
            self.register(ChartAsset(chart, "charts", chart))

        # Seed default animations
        for anim in ["spring", "counter_roll", "bounce", "overshoot", "glow_pulse", "float_parallax"]:
            self.register(AnimationAsset(anim, "animations", anim))

        # Seed default transitions
        self.register(TransitionAsset("crossfade", "transitions", "crossfade", 15))
        self.register(TransitionAsset("reveal_wipe", "transitions", "reveal_wipe", 15))
        self.register(TransitionAsset("hardcut", "transitions", "hardcut", 0))

        # Seed default fonts
        for font in ["Syne", "Space Grotesk", "Outfit", "PP Neue Montreal", "Cormorant Garamond"]:
            self.register(FontAsset(font, "fonts", font))

        # Seed default audio effects
        self.register(AudioAsset("whoosh", "audio", "audio/sfx/whoosh.mp3"))
        self.register(AudioAsset("pop", "audio", "audio/sfx/pop.mp3"))
        self.register(AudioAsset("ding", "audio", "audio/sfx/ding.mp3"))

    def register(self, asset: Asset) -> None:
        """Register an asset instance."""
        self._assets[asset.asset_id] = asset

    def resolve(self, asset_id: str, category: Optional[str] = None) -> Asset:
        """Resolve and validate an asset by its ID, returning a default fallback if missing."""
        if asset_id in self._assets:
            asset = self._assets[asset_id]
            if category and asset.category != category:
                raise ValueError(f"Asset '{asset_id}' category mismatch. Expected '{category}', got '{asset.category}'")
            return asset

        # Provide fallback defaults dynamically
        if category == "backgrounds":
            return BackgroundAsset(asset_id, "backgrounds", "dark_radial")
        elif category == "charts":
            return ChartAsset(asset_id, "charts", "ScoreDial")
        elif category == "animations":
            return AnimationAsset(asset_id, "animations", "spring")
        elif category == "transitions":
            return TransitionAsset(asset_id, "transitions", "crossfade", 15)
        elif category == "fonts":
            return FontAsset(asset_id, "fonts", "PP Neue Montreal")
        elif category == "audio":
            return AudioAsset(asset_id, "audio", f"audio/sfx/{asset_id}.mp3")
        elif category == "icons":
            return IconAsset(asset_id, "icons", asset_id)

        # General fallback Asset
        return Asset(asset_id, category or "general")
