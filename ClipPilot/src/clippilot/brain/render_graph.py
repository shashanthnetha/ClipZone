# -*- coding: utf-8 -*-
"""ClipPilot Brain Render Graph.

Deterministic intermediate representation mapping high-level composition layers,
frame boundaries, z-indexes, and assets into a pure render blueprint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from clippilot.brain.asset_registry import (
    Asset,
    AssetRegistry,
    BackgroundAsset,
    ChartAsset,
    FontAsset,
    TransitionAsset,
    AnimationAsset,
    AudioAsset,
    ImageAsset,
    IconAsset,
)
from clippilot.brain.composition_engine import Composition


@dataclass
class TimingReference:
    """Represents time boundaries in seconds and frames."""
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int
    duration_frames: int


@dataclass
class AssetReference:
    """Represents asset file details and mappings."""
    asset_id: str
    source_path: str
    asset: Asset


@dataclass
class AnimationReference:
    """Represents animation configurations."""
    animation_id: str
    animation: AnimationAsset
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderLayer:
    """Base visual/audio layer specification."""
    layer_id: str
    z_index: int
    layer_type: str  # "background", "subtitle", "chart", "image", "video", "audio", "transition"


@dataclass
class BackgroundLayer(RenderLayer):
    """Visual background overlay specification."""
    bg_type: BackgroundAsset


@dataclass
class SubtitleLayer(RenderLayer):
    """Formatted caption subtitle layer."""
    words: list[str] = field(default_factory=list)
    timings: list[tuple[float, float]] = field(default_factory=list)
    font_family: FontAsset = field(default_factory=FontAsset)
    font_size: int = 72
    emphasis_color: str = ""
    caption_style: str = ""


@dataclass
class ChartLayer(RenderLayer):
    """Visual chart/graph presentation layer."""
    chart_type: ChartAsset


@dataclass
class ImageLayer(RenderLayer):
    """Static visual image asset layer."""
    source_path: str = ""
    asset: Optional[Asset] = None


@dataclass
class VideoLayer(RenderLayer):
    """Dynamic motion video segment layer."""
    source_path: str = ""
    asset: Optional[Asset] = None


@dataclass
class AudioLayer(RenderLayer):
    """Audio narration or sound effects layer."""
    source_path: str = ""
    is_voiceover: bool = False
    audio_asset: Optional[AudioAsset] = None


@dataclass
class TransitionLayer(RenderLayer):
    """Visual scene transitions layout specification."""
    transition_type: TransitionAsset
    duration_frames: int = 15


@dataclass
class RenderScene:
    """Fully resolved intermediate render scene definition."""
    scene_index: int
    timing: TimingReference
    background: BackgroundLayer
    subtitle: Optional[SubtitleLayer] = None
    chart: Optional[ChartLayer] = None
    images: list[ImageLayer] = field(default_factory=list)
    videos: list[VideoLayer] = field(default_factory=list)
    audios: list[AudioLayer] = field(default_factory=list)
    transition: Optional[TransitionLayer] = None
    layers: list[RenderLayer] = field(default_factory=list)  # Sorted by z_index
    assets: list[AssetReference] = field(default_factory=list)
    animations: list[AnimationReference] = field(default_factory=list)


@dataclass
class RenderGraph:
    """The master intermediate representation ready for Remotion generator compilation."""
    title: str
    width: int
    height: int
    fps: int
    timing: TimingReference
    scenes: list[RenderScene] = field(default_factory=list)


def build_render_graph(composition: Composition, registry: Optional[AssetRegistry] = None) -> RenderGraph:
    """Deterministically transforms a layered Composition into a resolved RenderGraph."""
    if registry is None:
        registry = AssetRegistry()

    fps = composition.fps
    scenes: list[RenderScene] = []

    for comp_scene in composition.scenes:
        duration_sec = comp_scene.duration
        start_sec = comp_scene.start_time
        end_sec = start_sec + duration_sec

        start_frame = int(round(start_sec * fps))
        duration_frames = int(round(duration_sec * fps))
        end_frame = start_frame + duration_frames

        timing_ref = TimingReference(
            start_seconds=start_sec,
            end_seconds=round(end_sec, 2),
            start_frame=start_frame,
            end_frame=end_frame,
            duration_frames=duration_frames,
        )

        # 1. Background layer
        bg_name = comp_scene.background_layer.properties.get("bg_type", "dark_radial")
        bg_asset = registry.resolve(bg_name, "backgrounds")
        bg = BackgroundLayer(
            layer_id=comp_scene.background_layer.layer_id,
            z_index=comp_scene.background_layer.z_index,
            layer_type="background",
            bg_type=bg_asset,  # Reference resolved background Asset object
        )

        # 2. Chart layer
        chart = None
        if comp_scene.chart_layer:
            chart_name = comp_scene.chart_layer.properties.get("chart_type", "ScoreDial")
            chart_asset = registry.resolve(chart_name, "charts")
            chart = ChartLayer(
                layer_id=comp_scene.chart_layer.layer_id,
                z_index=comp_scene.chart_layer.z_index,
                layer_type="chart",
                chart_type=chart_asset,  # Reference resolved chart Asset object
            )

        # 3. Subtitles
        subtitle = None
        if comp_scene.subtitle_layers:
            sub = comp_scene.subtitle_layers[0]
            font_name = sub.style.get("fontFamily", "PP Neue Montreal")
            font_asset = registry.resolve(font_name, "fonts")
            subtitle = SubtitleLayer(
                layer_id=f"sub_{comp_scene.scene_index}",
                z_index=50,
                layer_type="subtitle",
                words=list(sub.words),
                timings=list(sub.timings),
                font_family=font_asset,  # Reference resolved font Asset object
                font_size=sub.style.get("fontSize", 72),
                emphasis_color=sub.style.get("emphasisColor", "#fca311"),
                caption_style=sub.style.get("captionStyle", ""),
            )

        # 4. Images (mapping asset references to visual layers)
        images = []
        for asset in comp_scene.asset_references:
            img_asset = registry.resolve(asset.asset_id)
            # If it's a dynamic image path, register or update it
            if not isinstance(img_asset, ImageAsset) and not isinstance(img_asset, IconAsset):
                img_asset = ImageAsset(asset.asset_id, "images", asset.source_path)
                registry.register(img_asset)

            images.append(
                ImageLayer(
                    layer_id=f"img_{asset.asset_id}",
                    z_index=20,
                    layer_type="image",
                    source_path=asset.source_path,
                    asset=img_asset,
                )
            )

        # 5. Audios (narration & sound effects)
        audios = []
        for idx, audio_path in enumerate(comp_scene.audio_references):
            is_vo = "voice/" in audio_path
            # Extract audio filename/id
            audio_id = audio_path.split("/")[-1].replace(".mp3", "")
            audio_asset = registry.resolve(audio_id, "audio")
            if not isinstance(audio_asset, AudioAsset):
                audio_asset = AudioAsset(audio_id, "audio", audio_path, is_vo)
                registry.register(audio_asset)

            audios.append(
                AudioLayer(
                    layer_id=f"audio_{comp_scene.scene_index}_{idx}",
                    z_index=0,
                    layer_type="audio",
                    source_path=audio_path,
                    is_voiceover=is_vo,
                    audio_asset=audio_asset,
                )
            )

        # 6. Transition
        transition = None
        if comp_scene.transition:
            trans_asset = registry.resolve(comp_scene.transition, "transitions")
            transition = TransitionLayer(
                layer_id=f"trans_{comp_scene.scene_index}",
                z_index=100,
                layer_type="transition",
                transition_type=trans_asset,
                duration_frames=trans_asset.default_duration_frames,
            )

        # Create sorted list of visual layers
        all_layers: list[RenderLayer] = [bg]
        if chart:
            all_layers.append(chart)
        if subtitle:
            all_layers.append(subtitle)
        all_layers.extend(images)
        if transition:
            all_layers.append(transition)

        all_layers.sort(key=lambda x: x.z_index)

        # Map AssetReferences referencing typed Asset objects
        assets_ref = []
        for a in comp_scene.asset_references:
            resolved_asset = registry.resolve(a.asset_id)
            if not isinstance(resolved_asset, ImageAsset) and not isinstance(resolved_asset, IconAsset):
                resolved_asset = ImageAsset(a.asset_id, "images", a.source_path)
            assets_ref.append(
                AssetReference(asset_id=a.asset_id, source_path=a.source_path, asset=resolved_asset)
            )

        # Map AnimationReferences referencing typed AnimationAsset objects
        animations_ref = []
        for anim in comp_scene.animation_references:
            anim_asset = registry.resolve(anim.animation_id, "animations")
            animations_ref.append(
                AnimationReference(
                    animation_id=anim.animation_id,
                    animation=anim_asset,
                    properties=dict(anim.properties),
                )
            )

        scenes.append(
            RenderScene(
                scene_index=comp_scene.scene_index,
                timing=timing_ref,
                background=bg,
                subtitle=subtitle,
                chart=chart,
                images=images,
                videos=[],
                audios=audios,
                transition=transition,
                layers=all_layers,
                assets=assets_ref,
                animations=animations_ref,
            )
        )

    # Master timeline reference
    total_sec = composition.duration_seconds
    total_frames = int(round(total_sec * fps))
    master_timing = TimingReference(
        start_seconds=0.0,
        end_seconds=total_sec,
        start_frame=0,
        end_frame=total_frames,
        duration_frames=total_frames,
    )

    return RenderGraph(
        title=composition.title,
        width=composition.width,
        height=composition.height,
        fps=composition.fps,
        timing=master_timing,
        scenes=scenes,
    )
