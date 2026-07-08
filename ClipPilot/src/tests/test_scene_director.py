# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json

from clippilot.brain.pipeline_orchestrator import Topic, VariationRecord, PipelineState
from clippilot.brain.strategy_engine import StrategyDecision
from clippilot.brain.creative_models import CreativeBlueprint, StoryFramework, EmotionCurve, SceneBlueprint
from clippilot.brain.director_engine import make_creative_blueprint
from clippilot.brain.director_rules import make_scene_blueprints
from clippilot.brain.scene_quality import calculate_scene_quality_score, repair_scene_blueprint
from clippilot.brain.scene_planner import plan_scenes, ScenePlan
from clippilot.brain.asset_intelligence import AssetIntelligenceEngine
from clippilot.brain.script_generator import generate_script, Script
from clippilot.brain.provider import JSONParsingError

class TestSceneDirector(unittest.TestCase):
    """Unit tests for Stage 2B Scene Director, visual priority, and quality scoring."""

    def test_make_scene_blueprints(self):
        topic = Topic(num="001", status="backlog", title="Why Credit Score Matters", niche="personal finance", angle="FICO mechanism", guardrail="None")
        blueprint = make_creative_blueprint(topic)
        
        scene_blueprints = make_scene_blueprints(blueprint)
        self.assertEqual(len(scene_blueprints), blueprint.target_scene_count)
        
        # Check first scene (Hook) and last scene (CTA) properties
        first = scene_blueprints[0]
        last = scene_blueprints[-1]
        
        self.assertEqual(first.retention_stage, "Hook")
        self.assertEqual(first.emotional_intensity, "High")
        self.assertEqual(first.visual_priority, "stock_video")
        self.assertTrue(len(first.visual_intent) > 0)
        self.assertTrue(len(first.search_queries) > 0)
        
        self.assertEqual(last.retention_stage, "CTA")
        self.assertEqual(last.emotional_intensity, "Medium")
        self.assertEqual(last.visual_priority, "text_only")

    def test_scene_quality_scoring_and_single_repair(self):
        # Create a deficient SceneBlueprint
        deficient_scene = SceneBlueprint(
            scene_number=1,
            duration=0.5,  # Too short (deduct 25)
            scene_goal="explain",
            visual_intent="nothing",
            visual_priority="stock_video",
            narration_purpose="narration",
            emotional_intensity="Low",  # First scene is Low intensity (deduct 15)
            retention_stage="Build",  # First scene is not Hook (deduct 20)
            visual_style="tech",
            animation_style="spring",
            transition="swipe",
            camera_language="static",
            caption_behavior="karaoke",
            search_queries=[],
            fallback_icon="default_icon.png",
            primary_visual="attention_icon"
        )
        
        score = calculate_scene_quality_score(deficient_scene, "Problem-Solution", 5)
        # Score must be 100 - 25 - 20 - 15 = 40
        self.assertEqual(score, 40.0)
        
        # Deterministic repair
        repaired = repair_scene_blueprint(deficient_scene, "Problem-Solution", 5)
        new_score = calculate_scene_quality_score(repaired, "Problem-Solution", 5)
        
        # After repair: duration = 1.5, retention_stage = "Hook", emotional_intensity = "High", score should be 100
        self.assertEqual(new_score, 100.0)
        self.assertEqual(repaired.duration, 1.5)
        self.assertEqual(repaired.retention_stage, "Hook")
        self.assertEqual(repaired.emotional_intensity, "High")

    def test_downstream_integration(self):
        # Verify plan_scenes and AssetIntelligence consume SceneBlueprint
        topic = Topic(num="001", status="backlog", title="Why Credit Score Matters", niche="personal finance", angle="FICO mechanism", guardrail="None")
        blueprint = make_creative_blueprint(topic)
        scene_blueprints = make_scene_blueprints(blueprint)
        
        script = Script(
            topic_num="001",
            title="Why Credit Score Matters",
            hook="Hook",
            niche_context="personal finance",
            scenes=[
                MagicMock(narration="Narration 1", visual_desc="Visual 1"),
                MagicMock(narration="Narration 2", visual_desc="Visual 2"),
                MagicMock(narration="Narration 3", visual_desc="Visual 3"),
                MagicMock(narration="Narration 4", visual_desc="Visual 4"),
                MagicMock(narration="Narration 5", visual_desc="Visual 5"),
            ]
        )
        variation = VariationRecord(date="2026-07-08", slug="daily_001", title="T01", fmt="F1", skin="S1", voice="V1", len="L3", pace="0%", cluster="finance", hook="question")
        
        scene_plan = plan_scenes(script, variation, scene_blueprints=scene_blueprints)
        
        # Verify scene_plan durations match blueprints
        for i, s in enumerate(scene_plan.scenes):
            self.assertEqual(s.duration_seconds, scene_blueprints[i].duration)
            self.assertEqual(s.transition_ids, [scene_blueprints[i].transition])
            self.assertEqual(s.animation_ids, [scene_blueprints[i].animation_style])
            
        asset_engine = AssetIntelligenceEngine()
        asset_plan = asset_engine.build_asset_plan(script, scene_plan, creative_blueprint=blueprint, scene_blueprints=scene_blueprints)
        
        # Verify asset_plan stock video queries match scene blueprints
        for i, s_plan in enumerate(asset_plan.scenes):
            self.assertEqual(s_plan.stock_video_queries, scene_blueprints[i].search_queries)

    @patch("clippilot.brain.provider.OpenAIProvider._ensure_client")
    @patch("clippilot.config.Settings.load")
    def test_json_correction_flow(self, mock_settings_load, mock_ensure_client):
        # Verify that exactly one correction request is sent to the same model on parse failure without switching
        mock_settings = MagicMock()
        mock_settings.llm_provider = "openai"
        mock_settings.llm_api_key = "fake_key"
        mock_settings.enable_reasoning_correction = True
        mock_settings.script_models = ["openai/gpt-oss-120b:free"]
        mock_settings_load.return_value = mock_settings
        
        mock_client = MagicMock()
        mock_ensure_client.return_value = mock_client
        
        # Mock completions.create to return a malformed response first, then a valid one
        resp_malformed = MagicMock()
        resp_malformed.choices = [MagicMock(message=MagicMock(content="Malformed response without JSON keys"))]
        resp_malformed.usage = MagicMock(prompt_tokens=10, completion_tokens=10)
        
        resp_corrected = MagicMock()
        resp_corrected.choices = [MagicMock(message=MagicMock(content='{"title": "Corrected title", "hook": "Corrected hook", "niche_context": "finance", "scenes": []}'))]
        resp_corrected.usage = MagicMock(prompt_tokens=20, completion_tokens=20)
        
        # First call returns malformed, second call (correction) returns corrected JSON
        mock_client.chat.completions.create.side_effect = [resp_malformed, resp_corrected]
        
        from clippilot.brain.provider import get_provider
        provider = get_provider(mock_settings, models=["openai/gpt-oss-120b:free"])
        
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "hook": {"type": "string"},
                "niche_context": {"type": "string"},
                "scenes": {"type": "array"}
            },
            "required": ["title", "hook", "niche_context", "scenes"]
        }
        
        text = provider.generate_text("Generate script", system_prompt="Sys prompt", json_schema=schema)
        
        self.assertEqual(text, '{"title": "Corrected title", "hook": "Corrected hook", "niche_context": "finance", "scenes": []}')
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        
        # Check that the second call was sent to the same model and appended the correction prompt
        calls = mock_client.chat.completions.create.call_args_list
        self.assertEqual(calls[0][1]["model"], "openai/gpt-oss-120b:free")
        self.assertEqual(calls[1][1]["model"], "openai/gpt-oss-120b:free")
        
        correction_messages = calls[1][1]["messages"]
        self.assertEqual(correction_messages[-1]["role"], "user")
        self.assertIn("You already generated the correct answer.", correction_messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
