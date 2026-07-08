# -*- coding: utf-8 -*-
"""Unit and integration tests for the Asset Intelligence Engine."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from clippilot.brain.asset_intelligence import AssetIntelligenceEngine
from clippilot.brain.asset_models import AssetReference, SceneAssetPlan, VideoAssetPlan
from clippilot.brain.script_generator import Script, ScriptScene
from clippilot.brain.scene_planner import ScenePlan, ScenePlanScene
from clippilot.config import Settings


class TestAssetIntelligence(unittest.TestCase):
    """Test suite for AssetIntelligenceEngine asset planning logic."""

    def setUp(self) -> None:
        import os
        self._keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "LLM_API_KEY")
        self._saved = {k: os.environ.pop(k, None) for k in self._keys}
        
        self.mock_settings = Settings(
            llm_provider="openrouter",
            llm_api_key="",
            pexels_api_key="",
            pixabay_api_key="",
            unsplash_api_key=""
        )
        self.load_patcher = patch("clippilot.config.Settings.load", return_value=self.mock_settings)
        self.load_patcher.start()

        from clippilot.brain import env
        self.orig_load_dotenv = env.load_dotenv
        env.load_dotenv = lambda *a, **k: None

        self.engine = AssetIntelligenceEngine()

    def test_empty_script(self) -> None:
        """Verifies an empty script is handled gracefully, yielding an empty plan."""
        script = Script(topic_num="001", title="Empty Title", hook="Hook", niche_context="credit", scenes=[])
        render_tree = ScenePlan(topic_num="001", title="Empty Title", hook="Hook", skin_id="S1", format_id="F1", voice_id="V1", scenes=[])
        
        plan = self.engine.build_asset_plan(script, render_tree)
        
        self.assertEqual(len(plan.scenes), 0)
        self.assertEqual(plan.metadata["provider"], "mock")

    def test_single_scene(self) -> None:
        """Verifies a script with a single scene is planned correctly."""
        script = Script(
            topic_num="001",
            title="Single Scene Topic",
            hook="Angle",
            niche_context="credit",
            scenes=[ScriptScene(narration="This is a credit card test.", visual_desc="Card visual")]
        )
        render_tree = ScenePlan(
            topic_num="001",
            title="Single Scene Topic",
            hook="Angle",
            skin_id="S1",
            format_id="F1",
            voice_id="V1",
            scenes=[ScenePlanScene(scene_index=1, narration="This is a credit card test.", duration_seconds=5.0, background_id="dark_radial")]
        )

        plan = self.engine.build_asset_plan(script, render_tree)

        self.assertEqual(len(plan.scenes), 1)
        scene_plan = plan.scenes[0]
        self.assertEqual(scene_plan.scene_number, 1)
        self.assertEqual(scene_plan.background, "dark_radial")
        self.assertIn("credit card payment", scene_plan.stock_video_queries)
        self.assertIn("credit-card", scene_plan.icon_queries)

    def test_multiple_scenes_deterministic(self) -> None:
        """Verifies deterministic output across multiple scenes with different keywords."""
        script = Script(
            topic_num="002",
            title="Multi Scene",
            hook="Angle",
            niche_context="telecom",
            scenes=[
                ScriptScene(narration="Check your phone bill statement.", visual_desc="Phone visual"),
                ScriptScene(narration="Your score might drop.", visual_desc="Score visual")
            ]
        )
        render_tree = ScenePlan(
            topic_num="002",
            title="Multi Scene",
            hook="Angle",
            skin_id="S2",
            format_id="F2",
            voice_id="V2",
            scenes=[
                ScenePlanScene(scene_index=1, narration="Check your phone bill statement.", duration_seconds=5.0, background_id="blueprint_grid"),
                ScenePlanScene(scene_index=2, narration="Your score might drop.", duration_seconds=5.0, background_id="blueprint_grid")
            ]
        )

        plan = self.engine.build_asset_plan(script, render_tree)

        self.assertEqual(len(plan.scenes), 2)
        
        # Scene 1: Phone bill keywords
        s1 = plan.scenes[0]
        self.assertEqual(s1.background, "blueprint_grid")
        self.assertIn("smartphone bill payment", s1.stock_video_queries)

        # Scene 2: Score keywords
        s2 = plan.scenes[1]
        self.assertIn("credit score rating", s2.stock_video_queries)
        self.assertEqual(s2.chart_type, "credit score")

    @patch("clippilot.brain.asset_intelligence.get_provider")
    @patch("os.environ.get")
    def test_provider_mode(self, mock_env_get, mock_get_provider) -> None:
        """Verifies that the engine invokes the provider when API keys are present."""
        mock_env_get.side_effect = lambda k, default=None: "api-key" if k == "ANTHROPIC_API_KEY" else default

        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_provider.model = "claude-3-5"

        mock_json_response = """
        {
          "scenes": [
            {
              "scene_number": 1,
              "background": "neon_vignette",
              "stock_video_queries": ["neon night stock market"],
              "image_queries": ["neon image query"],
              "icon_queries": ["dollar-sign", "trending-up"],
              "chart_type": "bar chart",
              "overlay_text": "Investments",
              "animations": ["bounce"],
              "transitions": ["fade"],
              "fallback_assets": [
                {
                  "asset_type": "video",
                  "provider": "pexels",
                  "search_query": "neon stock market",
                  "local_path": "assets/test.mp4",
                  "priority": 1,
                  "required": true
                }
              ]
            }
          ]
        }
        """
        mock_provider.generate_text.return_value = mock_json_response

        script = Script(
            topic_num="003",
            title="Provider test",
            hook="Hook",
            niche_context="invest",
            scenes=[ScriptScene(narration="Narration text", visual_desc="Visual desc")]
        )
        render_tree = ScenePlan(topic_num="003", title="Title", hook="H", skin_id="S1", format_id="F1", voice_id="V1")

        plan = self.engine.build_asset_plan(script, render_tree)

        self.assertEqual(plan.metadata["provider"], "MagicMock")
        self.assertEqual(plan.metadata["actual_model"], "claude-3-5")
        self.assertEqual(len(plan.scenes), 1)

        s_plan = plan.scenes[0]
        self.assertEqual(s_plan.background, "neon_vignette")
        self.assertEqual(s_plan.stock_video_queries, ["neon night stock market"])
        self.assertEqual(len(s_plan.fallback_assets), 1)
        self.assertEqual(s_plan.fallback_assets[0].provider, "pexels")

    def test_json_serialization(self) -> None:
        """Verifies that VideoAssetPlan converts cleanly to a JSON string."""
        script = Script(
            topic_num="004",
            title="JSON test",
            hook="H",
            niche_context="niche",
            scenes=[ScriptScene(narration="Test narration", visual_desc="Test visual")]
        )
        render_tree = ScenePlan(topic_num="004", title="Title", hook="H", skin_id="S1", format_id="F1", voice_id="V1")

        plan = self.engine.build_asset_plan(script, render_tree)
        plan_dict = plan.to_dict()

        # Check serialization safety
        serialized = json.dumps(plan_dict)
        deserialized = json.loads(serialized)

        self.assertIn("scenes", deserialized)
        self.assertEqual(len(deserialized["scenes"]), 1)
        self.assertEqual(deserialized["scenes"][0]["scene_number"], 1)
        self.assertEqual(deserialized["scenes"][0]["fallback_assets"][0]["provider"], "mock")

    def tearDown(self) -> None:
        self.load_patcher.stop()
        from clippilot.brain import env
        env.load_dotenv = self.orig_load_dotenv
        import os
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
