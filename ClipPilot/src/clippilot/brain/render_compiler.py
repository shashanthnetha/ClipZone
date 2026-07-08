# -*- coding: utf-8 -*-
"""ClipPilot Brain Render Compiler.

Deterministically compiles resolved RenderGraphs into hierarchical, timing-locked
SceneComponentTrees mapping layout dimensions, visual bounds, animations, and transitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from clippilot.brain.asset_registry import (
    Asset,
    BackgroundAsset,
    ChartAsset,
    FontAsset,
    TransitionAsset,
    AnimationAsset,
    AudioAsset,
)
from clippilot.brain.render_graph import RenderGraph, TimingReference


@dataclass
class ComponentBase:
    """Base class for compiled scene components."""
    id: str
    timing: TimingReference
    layout: dict[str, Any] = field(default_factory=dict)
    animations: list[AnimationComponent] = field(default_factory=list)
    transitions: list[TransitionComponent] = field(default_factory=list)


@dataclass
class BackgroundComponent(ComponentBase):
    """Compiled background component layout details."""
    asset: BackgroundAsset = field(default_factory=BackgroundAsset)


@dataclass
class SubtitleComponent(ComponentBase):
    """Compiled caption subtitles layout details."""
    asset: FontAsset = field(default_factory=FontAsset)
    words: list[str] = field(default_factory=list)
    timings: list[tuple[float, float]] = field(default_factory=list)
    font_size: int = 72
    emphasis_color: str = ""
    caption_style: str = ""


@dataclass
class ChartComponent(ComponentBase):
    """Compiled chart/graph layout details."""
    asset: ChartAsset = field(default_factory=ChartAsset)


@dataclass
class ImageComponent(ComponentBase):
    """Compiled static visual image component details."""
    asset: Asset = field(default_factory=Asset)
    source_path: str = ""


@dataclass
class VideoComponent(ComponentBase):
    """Compiled dynamic motion video component details."""
    asset: Asset = field(default_factory=Asset)
    source_path: str = ""


@dataclass
class AudioComponent(ComponentBase):
    """Compiled audio track component details."""
    asset: AudioAsset = field(default_factory=AudioAsset)
    source_path: str = ""
    is_voiceover: bool = False


@dataclass
class TransitionComponent(ComponentBase):
    """Compiled scene transitions component layout details."""
    asset: TransitionAsset = field(default_factory=TransitionAsset)
    transition_type: str = "crossfade"


@dataclass
class AnimationComponent(ComponentBase):
    """Compiled motion animations component layout details."""
    asset: AnimationAsset = field(default_factory=AnimationAsset)
    target_component_id: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneComponent:
    """Compiled representation of a complete scene component node."""
    id: str
    timing: TimingReference
    background: BackgroundComponent
    subtitle: Optional[SubtitleComponent] = None
    chart: Optional[ChartComponent] = None
    images: list[ImageComponent] = field(default_factory=list)
    videos: list[VideoComponent] = field(default_factory=list)
    audios: list[AudioComponent] = field(default_factory=list)
    animations: list[AnimationComponent] = field(default_factory=list)
    transitions: list[TransitionComponent] = field(default_factory=list)
    children: list[ComponentBase] = field(default_factory=list)  # Visual children z-sorted


@dataclass
class SceneComponentTree:
    """The master root component tree defining compilation dimensions and scenes."""
    title: str
    width: int
    height: int
    fps: int
    timing: TimingReference
    scenes: list[SceneComponent] = field(default_factory=list)


def compile_render_graph(graph: RenderGraph) -> SceneComponentTree:
    """Compiles a RenderGraph blueprint into a SceneComponentTree hierarchy."""
    compiled_scenes: list[SceneComponent] = []

    for scene in graph.scenes:
        # Determine standard layout positions based on dimension constraints
        w, h = graph.width, graph.height

        # Background fills screen
        bg_layout = {"x": 0, "y": 0, "width": w, "height": h}
        # Subtitles bottom area
        sub_layout = {"x": 100, "y": int(h * 0.75), "width": w - 200, "height": 300}
        # Chart center area
        chart_layout = {"x": 140, "y": int(h * 0.25), "width": w - 280, "height": int(h * 0.4)}
        # Default dynamic images center-aligned
        image_layout = {"x": 390, "y": int(h * 0.35), "width": 300, "height": 300}

        # 1. Background Component
        compiled_bg = BackgroundComponent(
            id=f"comp_bg_{scene.scene_index}",
            timing=scene.timing,
            layout=bg_layout,
            asset=scene.background.bg_type,
        )

        # 2. Subtitles Component
        compiled_sub = None
        if scene.subtitle:
            compiled_sub = SubtitleComponent(
                id=f"comp_sub_{scene.scene_index}",
                timing=scene.timing,
                layout=sub_layout,
                asset=scene.subtitle.font_family,
                words=list(scene.subtitle.words),
                timings=list(scene.subtitle.timings),
                font_size=scene.subtitle.font_size,
                emphasis_color=scene.subtitle.emphasis_color,
                caption_style=scene.subtitle.caption_style,
            )

        # 3. Chart Component
        compiled_chart = None
        if scene.chart:
            compiled_chart = ChartComponent(
                id=f"comp_chart_{scene.scene_index}",
                timing=scene.timing,
                layout=chart_layout,
                asset=scene.chart.chart_type,
            )

        # 4. Image Components
        compiled_images = []
        for idx, img in enumerate(scene.images):
            compiled_images.append(
                ImageComponent(
                    id=f"comp_img_{scene.scene_index}_{idx}",
                    timing=scene.timing,
                    layout=image_layout,
                    asset=img.asset if img.asset else Asset(img.layer_id, "images"),
                    source_path=img.source_path,
                )
            )

        # 5. Audio Components
        compiled_audios = []
        for idx, audio in enumerate(scene.audios):
            compiled_audios.append(
                AudioComponent(
                    id=f"comp_audio_{scene.scene_index}_{idx}",
                    timing=scene.timing,
                    layout={},
                    asset=audio.audio_asset if audio.audio_asset else AudioAsset(audio.layer_id, "audio"),
                    source_path=audio.source_path,
                    is_voiceover=audio.is_voiceover,
                )
            )

        # 6. Transitions Component
        compiled_transitions = []
        if scene.transition:
            compiled_transitions.append(
                TransitionComponent(
                    id=f"comp_trans_{scene.scene_index}",
                    timing=scene.timing,
                    layout={},
                    asset=scene.transition.transition_type,
                    transition_type=scene.transition.transition_type.transition_type,
                )
            )

        # 7. Animations Component
        compiled_animations = []
        for idx, anim in enumerate(scene.animations):
            compiled_animations.append(
                AnimationComponent(
                    id=f"comp_anim_{scene.scene_index}_{idx}",
                    timing=scene.timing,
                    layout={},
                    asset=anim.animation,
                    target_component_id=f"comp_chart_{scene.scene_index}" if scene.chart else f"comp_bg_{scene.scene_index}",
                    properties=dict(anim.properties),
                )
            )

        # Apply animations & transitions references to each subcomponent
        compiled_bg.animations = list(compiled_animations)
        compiled_bg.transitions = list(compiled_transitions)
        if compiled_sub:
            compiled_sub.animations = list(compiled_animations)
            compiled_sub.transitions = list(compiled_transitions)
        if compiled_chart:
            compiled_chart.animations = list(compiled_animations)
            compiled_chart.transitions = list(compiled_transitions)
        for img_comp in compiled_images:
            img_comp.animations = list(compiled_animations)
            img_comp.transitions = list(compiled_transitions)

        # Visual children z-sorted ordering
        visual_children: list[ComponentBase] = [compiled_bg]
        if compiled_chart:
            visual_children.append(compiled_chart)
        visual_children.extend(compiled_images)
        if compiled_sub:
            visual_children.append(compiled_sub)

        compiled_scenes.append(
            SceneComponent(
                id=f"scene_node_{scene.scene_index}",
                timing=scene.timing,
                background=compiled_bg,
                subtitle=compiled_sub,
                chart=compiled_chart,
                images=compiled_images,
                videos=[],
                audios=compiled_audios,
                animations=compiled_animations,
                transitions=compiled_transitions,
                children=visual_children,
            )
        )

    return SceneComponentTree(
        title=graph.title,
        width=graph.width,
        height=graph.height,
        fps=graph.fps,
        timing=graph.timing,
        scenes=compiled_scenes,
    )
