# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Render Compiler."""
from __future__ import annotations

import unittest

from clippilot.brain.composition_engine import compose_video
from clippilot.brain.pipeline_orchestrator import VariationRecord
from clippilot.brain.render_compiler import (
    compile_render_graph,
    BackgroundComponent,
    SubtitleComponent,
    ChartComponent,
    ImageComponent,
    AudioComponent,
    AnimationComponent,
    TransitionComponent,
)
from clippilot.brain.render_graph import build_render_graph
from clippilot.brain.scene_planner import plan_scenes
from clippilot.brain.script_generator import Script, ScriptScene


class TestRenderCompiler(unittest.TestCase):
    """Tests for scene components tree compilation, z-ordered rendering stacks, and bindings."""

    def test_compile_render_graph_basic(self) -> None:
        script = Script(
            topic_num="006",
            title="Closing card hurts score",
            hook="Stop closing your cards",
            niche_context="credit",
            scenes=[
                ScriptScene("Sentence number one.", "visual 1"),
                ScriptScene("Sentence number two.", "visual 2"),
            ]
        )
        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "credit", "question")
        scene_plan = plan_scenes(script, variation)
        comp = compose_video(scene_plan, width=1080, height=1920, fps=30)
        graph = build_render_graph(comp)

        tree = compile_render_graph(graph)

        # 1. Structural properties check
        self.assertEqual(tree.title, "Closing card hurts score")
        self.assertEqual(tree.width, 1080)
        self.assertEqual(tree.height, 1920)
        self.assertEqual(tree.fps, 30)
        self.assertEqual(len(tree.scenes), 2)

        scene1 = tree.scenes[0]
        self.assertEqual(scene1.id, "scene_node_1")

        # 2. Check strongly typed subcomponents
        self.assertIsInstance(scene1.background, BackgroundComponent)
        self.assertIsInstance(scene1.subtitle, SubtitleComponent)
        self.assertIsInstance(scene1.chart, ChartComponent)
        self.assertGreater(len(scene1.images), 0)
        self.assertIsInstance(scene1.images[0], ImageComponent)
        self.assertGreater(len(scene1.audios), 0)
        self.assertIsInstance(scene1.audios[0], AudioComponent)

        # Check visual children sorting: background -> chart -> images -> subtitle
        children = scene1.children
        self.assertEqual(children[0], scene1.background)
        self.assertEqual(children[1], scene1.chart)
        self.assertEqual(children[-1], scene1.subtitle)

        # 3. Layout validation
        self.assertEqual(scene1.background.layout["width"], 1080)
        self.assertEqual(scene1.subtitle.layout["x"], 100)

        # 4. Timing verification
        self.assertEqual(scene1.background.timing.start_frame, scene1.timing.start_frame)
        self.assertEqual(scene1.background.timing.duration_frames, scene1.timing.duration_frames)

        # 5. Animations & transitions verification
        self.assertGreater(len(scene1.background.animations), 0)
        self.assertIsInstance(scene1.background.animations[0], AnimationComponent)
        self.assertEqual(scene1.background.animations[0].target_component_id, f"comp_chart_1")

        self.assertGreater(len(scene1.background.transitions), 0)
        self.assertIsInstance(scene1.background.transitions[0], TransitionComponent)
