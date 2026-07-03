# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Composition Engine."""
from __future__ import annotations

import unittest

from clippilot.brain.composition_engine import compose_video
from clippilot.brain.pipeline_orchestrator import VariationRecord
from clippilot.brain.scene_planner import plan_scenes
from clippilot.brain.script_generator import Script, ScriptScene


class TestCompositionEngine(unittest.TestCase):
    """Tests for visual layers compilation, z-indexing, asset references, and timing verification."""

    def test_compose_video_basic(self) -> None:
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

        # Check basic dimension defaults
        self.assertEqual(comp.width, 1080)
        self.assertEqual(comp.height, 1920)
        self.assertEqual(comp.fps, 30)
        self.assertEqual(comp.title, "Closing card hurts score")

        # Check scene compilation count
        self.assertEqual(len(comp.scenes), 2)

        # Verify z-index ordering: background(0) -> chart(10) -> asset(20+) -> subtitle(50)
        scene1 = comp.scenes[0]
        self.assertEqual(scene1.start_time, 0.0)
        self.assertGreater(scene1.duration, 0.0)

        layers = scene1.visual_layers
        self.assertGreaterEqual(len(layers), 3)  # background, chart, subtitle, maybe asset
        for idx in range(len(layers) - 1):
            self.assertLessEqual(layers[idx].z_index, layers[idx + 1].z_index)

        # Verify font mapping (S1 -> Syne)
        self.assertEqual(scene1.subtitle_layers[0].style["fontFamily"], "Syne")

        # Verify audio assets
        self.assertIn("audio/sfx/whoosh.mp3", scene1.audio_references)
        self.assertIn("audio/voice/006_scene_1.mp3", scene1.audio_references)

        # Verify asset resolution path mapping
        self.assertEqual(scene1.asset_references[0].source_path, f"assets/graphics/{scene1.asset_references[0].asset_id}.png")
