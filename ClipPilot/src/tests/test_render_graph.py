# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Render Graph."""
from __future__ import annotations

import unittest

from clippilot.brain.composition_engine import compose_video
from clippilot.brain.pipeline_orchestrator import VariationRecord
from clippilot.brain.render_graph import build_render_graph
from clippilot.brain.scene_planner import plan_scenes
from clippilot.brain.script_generator import Script, ScriptScene


class TestRenderGraph(unittest.TestCase):
    """Tests for intermediate graph compilation, timing/frame calculations, and z-index ordering."""

    def test_build_render_graph_basic(self) -> None:
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

        # 1. Master timing calculations
        self.assertEqual(graph.title, "Closing card hurts score")
        self.assertEqual(graph.width, 1080)
        self.assertEqual(graph.height, 1920)
        self.assertEqual(graph.fps, 30)
        self.assertEqual(graph.timing.start_frame, 0)
        self.assertGreater(graph.timing.duration_frames, 0)
        self.assertEqual(graph.timing.end_frame, graph.timing.duration_frames)

        # 2. Scene mappings checks
        self.assertEqual(len(graph.scenes), 2)
        scene1 = graph.scenes[0]

        # Check timing frame calculation match
        expected_frames = int(round(scene1.timing.duration_frames))
        self.assertEqual(scene1.timing.end_frame - scene1.timing.start_frame, expected_frames)

        # 3. Layer list checks & z-index ordering
        layers = scene1.layers
        self.assertGreater(len(layers), 2)
        for idx in range(len(layers) - 1):
            self.assertLessEqual(layers[idx].z_index, layers[idx + 1].z_index)

        # Verify layer type distributions
        self.assertEqual(scene1.background.layer_type, "background")
        self.assertIsNotNone(scene1.subtitle)
        self.assertEqual(scene1.subtitle.layer_type, "subtitle")

        # Check typed Asset object references
        from clippilot.brain.asset_registry import BackgroundAsset, FontAsset, ImageAsset
        self.assertIsInstance(scene1.background.bg_type, BackgroundAsset)
        self.assertIsInstance(scene1.subtitle.font_family, FontAsset)

        # 4. References checks
        self.assertGreater(len(scene1.assets), 0)
        self.assertEqual(scene1.assets[0].source_path, f"assets/graphics/{scene1.assets[0].asset_id}.png")
        self.assertIsInstance(scene1.assets[0].asset, ImageAsset)

        self.assertGreater(len(scene1.animations), 0)
        self.assertEqual(scene1.animations[0].properties["duration_frames"], expected_frames)
