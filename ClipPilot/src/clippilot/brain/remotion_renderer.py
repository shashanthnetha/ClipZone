# -*- coding: utf-8 -*-
"""ClipPilot Remotion Renderer.

Deterministically compiles SceneComponentTree models into ready-to-render React/TSX
Remotion component compositions without mutating the local filesystem.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from clippilot.brain.render_compiler import (
    SceneComponentTree,
    SceneComponent,
    BackgroundComponent,
    SubtitleComponent,
    ChartComponent,
    ImageComponent,
    VideoComponent,
    AudioComponent,
    TransitionComponent,
    AnimationComponent,
    ComponentBase,
)


@dataclass
class RemotionRenderOutput:
    """Carries the compiled TSX source strings representing the Remotion visual nodes."""
    react_component_tree: str
    remotion_composition: str


def _to_react_style(layout: dict[str, Any]) -> str:
    """Helper to convert layout dictionary properties into a inline React style string."""
    style_items = []
    for k, v in layout.items():
        # Map x/y to absolute positioning
        if k == "x":
            style_items.append(f"left: {v}")
        elif k == "y":
            style_items.append(f"top: {v}")
        elif k in ["width", "height", "fontSize"]:
            style_items.append(f"{k}: {v}")
        elif isinstance(v, str):
            style_items.append(f"{k}: '{v}'")
        else:
            style_items.append(f"{k}: {v}")

    if style_items:
        # Add absolute positioning if x/y are present
        if "x" in layout or "y" in layout:
            style_items.append("position: 'absolute'")
        return "{{" + ", ".join(style_items) + "}}"
    return "{{}}"


SILENT_MP3_B64 = (
    "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjYyLjEyLjEwMQAAAAAAAAAAAAAA//tAwAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAoAAAQ9gAQEBYWHR0dIyMpKSkvLzU1NTs7QUFBSEhOTk5UVFpaWmBgZmZmbGxycnJ5eX9/f4WFi4uLkZGXl5ednaSkpKqqsLCwtra8vLzCwsjIyM7O1dXV29vh4eHn5+3t7fPz+fn5//8AAAAATGF2YzYyLjI4AAAAAAAAAAAAAAAAJAV8AAAAAAAAEPYp+CflAAAAAAD/+xDEAAPAAAGkAAAAIAAANIAAAARMQU1FMy4xMDBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMQpg8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxFMDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDEfIPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMSmA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxM+DwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQ=="
)

def _ensure_default_assets(public_dir: Any) -> None:
    import base64
    public_dir.mkdir(parents=True, exist_ok=True)
    
    default_sfx = public_dir / "default_sfx.mp3"
    if not default_sfx.exists():
        default_sfx.write_bytes(base64.b64decode(SILENT_MP3_B64))
        
    tiny_png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    for img_name in ["default_background.png", "default_chart.png", "default_icon.png"]:
        img_path = public_dir / img_name
        if not img_path.exists():
            img_path.write_bytes(tiny_png)


def render_to_remotion(tree: SceneComponentTree) -> RemotionRenderOutput:
    """Translates a SceneComponentTree into deterministic React & Remotion composition TSX strings."""
    import os
    import inspect
    from pathlib import Path

    # Determine public directory dynamically using caller frame inspection
    workspace_dir = None
    for frame_info in inspect.stack():
        locals_dict = frame_info.frame.f_locals
        if locals_dict.get("workspace_dir") is not None:
            workspace_dir = Path(locals_dict["workspace_dir"])
            break
        if locals_dict.get("explainer_dir") is not None:
            workspace_dir = Path(locals_dict["explainer_dir"]).parent.parent
            break

    if workspace_dir:
        public_dir = workspace_dir / "ClipPilot" / "remotion_explainer" / "public"
    else:
        public_dir = Path.cwd() / "ClipPilot" / "remotion_explainer" / "public"

    if not public_dir.exists():
        public_dir = Path(__file__).resolve().parent.parent.parent.parent / "remotion_explainer" / "public"

    _ensure_default_assets(public_dir)

    # 1. Compile components source code templates
    component_lines = []

    for idx, scene in enumerate(tree.scenes):
        scene_num = idx + 1
        scene_lines = [
            f"// Scene {scene_num} Component",
            f"export const Scene{scene_num}: React.FC = () => {{",
            "  return (",
            "    <div style={{ position: 'relative', width: '100%', height: '100%' }}>"
        ]

        # Render each child element (Background, Chart, Images, Subtitles)
        for child in scene.children:
            # Layout conversion
            layout_str = _to_react_style(child.layout)

            # Build animation and transition descriptors
            anim_props = []
            for anim in child.animations:
                anim_props.append(
                    f"{{ type: '{anim.asset.animation_type}', targetId: '{anim.target_component_id}', "
                    f"properties: {json.dumps(anim.properties)} }}"
                )
            anim_str = f"[{', '.join(anim_props)}]"

            trans_props = []
            for trans in child.transitions:
                trans_props.append(
                    f"{{ type: '{trans.transition_type}', durationFrames: {trans.asset.default_duration_frames} }}"
                )
            trans_str = f"[{', '.join(trans_props)}]"

            if isinstance(child, BackgroundComponent):
                scene_lines.append(
                    f"      <Background bgType='{child.asset.bg_type}' style={layout_str} "
                    f"animations={{{anim_str}}} transitions={{{trans_str}}} />"
                )
            elif isinstance(child, ChartComponent):
                scene_lines.append(
                    f"      <Chart chartType='{child.asset.chart_type}' style={layout_str} "
                    f"animations={{{anim_str}}} transitions={{{trans_str}}} />"
                )
            elif isinstance(child, ImageComponent):
                img_path = child.source_path
                if not (public_dir / img_path).exists():
                    print(f"⚠️ Warning: Missing image asset '{img_path}'. Falling back to 'default_icon.png'")
                    img_path = "default_icon.png"
                scene_lines.append(
                    f"      <Image src={{staticFile('{img_path}')}} style={layout_str} "
                    f"animations={{{anim_str}}} transitions={{{trans_str}}} />"
                )
            elif isinstance(child, VideoComponent):
                video_path = child.source_path
                if not (public_dir / video_path).exists():
                    print(f"⚠️ Warning: Missing video asset '{video_path}'. Falling back to 'default_icon.png'")
                    video_path = "default_icon.png"
                scene_lines.append(
                    f"      <Video src={{staticFile('{video_path}')}} style={layout_str} "
                    f"animations={{{anim_str}}} transitions={{{trans_str}}} />"
                )
            elif isinstance(child, SubtitleComponent):
                words_json = json.dumps(child.words)
                timings_json = json.dumps(child.timings)
                scene_lines.append(
                    f"      <Subtitle words={{{words_json}}} timings={{{timings_json}}} style={layout_str} "
                    f"animations={{{anim_str}}} transitions={{{trans_str}}} />"
                )

        # Audio tracks rendering (Narration and SFX)
        for audio in scene.audios:
            audio_path = audio.source_path
            if not (public_dir / audio_path).exists():
                print(f"⚠️ Warning: Missing audio asset '{audio_path}'. Falling back to 'default_sfx.mp3'")
                audio_path = "default_sfx.mp3"
            scene_lines.append(
                f"      <Audio src={{staticFile('{audio_path}')}} volume={{1.0}} startFrame={{{scene.timing.start_frame}}} />"
            )

        scene_lines.extend([
            "    </div>",
            "  );",
            "};",
            ""
        ])
        component_lines.extend(scene_lines)

    imports = [
        "import React from 'react';",
        "import { Audio, staticFile } from 'remotion';",
        "import { Background, Chart, Image, Subtitle, Video, Transition, Animation } from './components';",
        ""
    ]
    react_component_tree = "\n".join(imports + component_lines)

    # 2. Compile composition registry Root.tsx source code
    root_lines = [
        "import { Composition, Sequence } from 'remotion';",
        "import React from 'react';",
        ""
    ]
    # Add imports of all scene components
    for idx, scene in enumerate(tree.scenes):
        root_lines.append(f"import {{ Scene{idx + 1} }} from './scenes';")

    import re
    sanitized_id = re.sub(r'[^a-zA-Z0-9-]', '', tree.title.replace(' ', '-').replace('_', '-'))

    root_lines.extend([
        "",
        "export const RemotionRoot: React.FC = () => {",
        "  return (",
        "    <>",
        f"      <Composition",
        f"        id='{sanitized_id}'",
        f"        component={{() => (",
        f"          <div style={{{{ width: '100%', height: '100%' }}}}>"
    ])

    # Lay out each scene within a Sequence
    for idx, scene in enumerate(tree.scenes):
        root_lines.append(
            f"            <Sequence from={{{scene.timing.start_frame}}} durationInFrames={{{scene.timing.duration_frames}}}>"
        )
        root_lines.append(f"              <Scene{idx + 1} />")
        root_lines.append("            </Sequence>")

    root_lines.extend([
        "          </div>",
        "        )}",
        f"        width={{{tree.width}}}",
        f"        height={{{tree.height}}}",
        f"        fps={{{tree.fps}}}",
        f"        durationInFrames={{{tree.timing.duration_frames}}}",
        "      />",
        "    </>",
        "  );",
        "};"
    ])

    remotion_composition = "\n".join(root_lines)

    return RemotionRenderOutput(
        react_component_tree=react_component_tree,
        remotion_composition=remotion_composition,
    )
