# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Scene Planner."""
from __future__ import annotations

import unittest

from clippilot.brain.pipeline_orchestrator import VariationRecord
from clippilot.brain.scene_planner import plan_scenes
from clippilot.brain.script_generator import Script, ScriptScene


class TestScenePlanner(unittest.TestCase):
    """Tests for deterministic layout mapping, timing calculations, and asset selections."""

    def test_plan_scenes_basic(self) -> None:
        script = Script(
            topic_num="006",
            title="Closing card hurts score",
            hook="Stop closing your cards",
            niche_context="credit",
            scenes=[
                ScriptScene("This is the first sentence.", "visual 1"),
                ScriptScene("Here is an intermediate detail.", "visual 2"),
                ScriptScene("And this is the final payoff.", "visual 3"),
            ]
        )
        # S2: Blueprint, F1: Explainer
        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S2", "V1", "L3", "0%", "credit", "question")

        plan = plan_scenes(script, variation)

        # Check total structure mappings
        self.assertEqual(plan.topic_num, "006")
        self.assertEqual(plan.skin_id, "S2")
        self.assertEqual(plan.format_id, "F1")
        self.assertEqual(len(plan.scenes), 3)

        # Check background and chart resolution for S2 (Blueprint)
        self.assertEqual(plan.scenes[0].background_id, "blueprint_grid")
        self.assertEqual(plan.scenes[0].chart_id, "schematic_line")
        self.assertIn("draw_path", plan.scenes[0].animation_ids)
        self.assertIn("reveal_wipe", plan.scenes[0].transition_ids)

        # Check SFX allocation (first: whoosh, intermediate: pop, final: ding)
        self.assertEqual(plan.scenes[0].sfx_ids, ["whoosh"])
        self.assertEqual(plan.scenes[1].sfx_ids, ["pop"])
        self.assertEqual(plan.scenes[2].sfx_ids, ["ding"])

        # Check duration and timing
        # "This is the first sentence." = 5 words. At 2.5 wps = 2.0s
        self.assertEqual(plan.scenes[0].duration_seconds, 2.0)
        self.assertEqual(len(plan.scenes[0].subtitle_words), 5)
        self.assertEqual(len(plan.scenes[0].subtitle_timings), 5)
        # First word timing: 0.0 to 0.4s
        self.assertEqual(plan.scenes[0].subtitle_timings[0], (0.0, 0.4))
        # Last word timing: 1.6 to 2.0s
        self.assertEqual(plan.scenes[0].subtitle_timings[4], (1.6, 2.0))

        # Check sum matches
        expected_total = sum(s.duration_seconds for s in plan.scenes)
        self.assertAlmostEqual(plan.total_duration_seconds, expected_total, places=2)

    def test_plan_scenes_format_assets(self) -> None:
        script = Script(
            topic_num="006",
            title="Closing card hurts score",
            hook="Stop closing your cards",
            niche_context="credit",
            scenes=[
                ScriptScene("List item one.", "visual 1"),
            ]
        )

        # Test Format F3 (Listicle) -> should append list_item_1 asset
        variation_list = VariationRecord("2026-07-03", "slug", "T01", "F3", "S1", "V1", "L3", "0%", "credit", "question")
        plan_list = plan_scenes(script, variation_list)
        self.assertIn("list_item_1", plan_list.scenes[0].asset_ids)

        # Test Format F5 (Comparison) -> should append split_column asset
        variation_comp = VariationRecord("2026-07-03", "slug", "T01", "F5", "S1", "V1", "L3", "0%", "credit", "question")
        plan_comp = plan_scenes(script, variation_comp)
        self.assertIn("split_column", plan_comp.scenes[0].asset_ids)
