# -*- coding: utf-8 -*-
"""ClipPilot Composition Engine.

Compiles layout plans, asset mappings, and timelines into a deterministic,
layered multi-sceneComposition model ready for rendering/remotion code generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from clippilot.brain.scene_planner import ScenePlan


@dataclass
class VisualLayer:
    """Represents a renderable layer in the visual stack."""
    layer_id: str
    z_index: int
    layer_type: str  # "background", "chart", "asset", "subtitle"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetRef:
    """Represents a reference to a static or dynamic graphical asset."""
    asset_id: str
    source_path: str


@dataclass
class AnimationRef:
    """Represents a motion animation binding."""
    animation_id: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubtitleLayer:
    """Represents formatted captions overlays."""
    words: list[str] = field(default_factory=list)
    timings: list[tuple[float, float]] = field(default_factory=list)
    style: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionScene:
    """A fully-composed segment carrying layers, assets, and timings."""
    scene_index: int
    start_time: float
    duration: float
    background_layer: VisualLayer
    chart_layer: Optional[VisualLayer] = None
    visual_layers: list[VisualLayer] = field(default_factory=list)  # ordered by z_index
    asset_references: list[AssetRef] = field(default_factory=list)
    animation_references: list[AnimationRef] = field(default_factory=list)
    subtitle_layers: list[SubtitleLayer] = field(default_factory=list)
    transition: Optional[str] = None
    audio_references: list[str] = field(default_factory=list)


@dataclass
class Composition:
    """The master layout output defining dimensions, fps, and composed scenes."""
    title: str
    width: int
    height: int
    fps: int
    duration_seconds: float
    scenes: list[CompositionScene] = field(default_factory=list)


def compose_video(scene_plan: ScenePlan, width: int = 1080, height: int = 1920, fps: int = 30) -> Composition:
    """Transforms a ScenePlan into a deterministic, layered Composition blueprint."""
    composed_scenes: list[CompositionScene] = []
    current_time = 0.0

    # Dynamic styling depending on Skin ID
    font_map = {
        "S1": "Syne",
        "S2": "Space Grotesk",
        "S3": "Outfit",
        "S4": "PP Neue Montreal",
        "S5": "Cormorant Garamond",
        "S6": "Outfit",
    }
    font_family = font_map.get(scene_plan.skin_id, "PP Neue Montreal")

    for scene in scene_plan.scenes:
        duration = scene.duration_seconds

        # 1. Background layer
        bg_layer = VisualLayer(
            layer_id=f"bg_{scene.scene_index}_{scene.background_id}",
            z_index=0,
            layer_type="background",
            properties={"bg_type": scene.background_id},
        )

        # 2. Chart layer (if set)
        chart_layer = None
        if scene.chart_id:
            chart_layer = VisualLayer(
                layer_id=f"chart_{scene.scene_index}_{scene.chart_id}",
                z_index=10,
                layer_type="chart",
                properties={"chart_type": scene.chart_id},
            )

        # 3. Subtitle layer
        sub_style = {
            "fontFamily": font_family,
            "fontSize": 72,
            "color": "#FFFFFF",
            "textShadow": "0px 4px 10px rgba(0,0,0,0.5)",
        }
        if getattr(scene_plan, "emphasis_color", None):
            sub_style["emphasisColor"] = scene_plan.emphasis_color
        if getattr(scene_plan, "caption_style", None):
            sub_style["captionStyle"] = scene_plan.caption_style

        sub_layer = SubtitleLayer(
            words=list(scene.subtitle_words),
            timings=list(scene.subtitle_timings),
            style=sub_style,
        )

        # 4. Compile and order visual layers stack
        vis_layers = [bg_layer]
        if chart_layer:
            vis_layers.append(chart_layer)

        # Asset layers
        asset_refs = []
        for idx, asset_id in enumerate(scene.asset_ids):
            z_idx = 20 + idx
            vis_layers.append(
                VisualLayer(
                    layer_id=f"layer_asset_{scene.scene_index}_{asset_id}",
                    z_index=z_idx,
                    layer_type="asset",
                    properties={"asset_id": asset_id},
                )
            )
            # Map asset to source path
            asset_refs.append(
                AssetRef(asset_id=asset_id, source_path=f"assets/graphics/{asset_id}.png")
            )

        # Subtitle visual representation layer
        vis_layers.append(
            VisualLayer(
                layer_id=f"layer_subtitle_{scene.scene_index}",
                z_index=50,
                layer_type="subtitle",
                properties={"style": sub_style},
            )
        )

        # Sort layers by z_index
        vis_layers.sort(key=lambda x: x.z_index)

        # 5. Map animation references
        anim_refs = [
            AnimationRef(animation_id=anim_id, properties={"duration_frames": int(duration * fps)})
            for anim_id in scene.animation_ids
        ]

        # 6. Audio resources mapping (narration voiceover + sfx)
        audio_refs = [f"audio/voice/{scene_plan.topic_num}_scene_{scene.scene_index}.mp3"]
        for sfx in scene.sfx_ids:
            audio_refs.append(f"audio/sfx/{sfx}.mp3")

        # 7. Transition settings
        trans = scene.transition_ids[0] if scene.transition_ids else None

        composed_scenes.append(
            CompositionScene(
                scene_index=scene.scene_index,
                start_time=round(current_time, 2),
                duration=duration,
                background_layer=bg_layer,
                chart_layer=chart_layer,
                visual_layers=vis_layers,
                asset_references=asset_refs,
                animation_references=anim_refs,
                subtitle_layers=[sub_layer],
                transition=trans,
                audio_references=audio_refs,
            )
        )
        current_time += duration

    return Composition(
        title=scene_plan.title,
        width=width,
        height=height,
        fps=fps,
        duration_seconds=round(current_time, 2),
        scenes=composed_scenes,
    )
