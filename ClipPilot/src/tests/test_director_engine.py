# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import tempfile

from clippilot.brain.pipeline_orchestrator import Topic, VariationRecord, PipelineState
from clippilot.brain.strategy_engine import StrategyDecision
from clippilot.brain.creative_models import CreativeBlueprint, StoryFramework, EmotionCurve
from clippilot.brain.director_engine import make_creative_blueprint
from clippilot.brain.prompt_builder import build_blueprint_instructions
from clippilot.brain.script_generator import generate_script, Script

class TestDirectorEngine(unittest.TestCase):
    """Tests for deterministic AI Director Engine and Creative Blueprint integration."""

    def test_make_creative_blueprint_deterministic(self):
        topic = Topic(num="001", status="backlog", title="Why Credit Score Matters", niche="personal finance", angle="The mechanism of FICO calculations", guardrail="Do not give financial advice")
        variation = VariationRecord(date="2026-07-08", slug="daily_001", title="T01", fmt="F1", skin="S1", voice="V1", len="L3", pace="0%", cluster="finance", hook="question")
        decision = StrategyDecision(
            topic=topic,
            hook="question",
            voice="V1",
            skin="S1",
            format="F1",
            confidence=0.85,
            reasoning="High historical performance",
            variation=variation
        )

        blueprint_1 = make_creative_blueprint(topic, decision)
        blueprint_2 = make_creative_blueprint(topic, decision)

        # Output must be identical (deterministic)
        self.assertEqual(blueprint_1.audience, blueprint_2.audience)
        self.assertEqual(blueprint_1.target_duration, blueprint_2.target_duration)
        self.assertEqual(blueprint_1.story_framework.name, blueprint_2.story_framework.name)
        self.assertEqual(blueprint_1.emotion_curve.pattern, blueprint_2.emotion_curve.pattern)
        self.assertEqual(blueprint_1.pacing, blueprint_2.pacing)
        self.assertEqual(blueprint_1.cta_strategy, blueprint_2.cta_strategy)
        self.assertEqual(blueprint_1.caption_style, blueprint_2.caption_style)
        self.assertEqual(blueprint_1.visual_style, blueprint_2.visual_style)
        self.assertEqual(blueprint_1.target_scene_count, blueprint_2.target_scene_count)
        self.assertEqual(blueprint_1.primary_goal, blueprint_2.primary_goal)
        self.assertEqual(blueprint_1.metadata["creative_version"], "1.0")

    def test_story_framework_selection(self):
        # 1. Myth-Busting selection
        myth_topic = Topic(num="002", status="backlog", title="The Debunked Myth about Sleep", niche="health and wellness", angle="debunking the 8-hour sleep lie", guardrail="None")
        myth_blueprint = make_creative_blueprint(myth_topic)
        self.assertEqual(myth_blueprint.story_framework.name, "Myth-Busting")
        self.assertEqual(myth_blueprint.emotion_curve.pattern, "Tension-Release")
        self.assertIn("myth/debunk/lie", myth_blueprint.metadata["decision_reasoning"]["story_framework"])

        # 2. Problem-Solution selection
        prob_topic = Topic(num="003", status="backlog", title="Why Your Credit Utilization is High", niche="credit cards", angle="The mechanism of balance reports", guardrail="None")
        prob_blueprint = make_creative_blueprint(prob_topic)
        self.assertEqual(prob_blueprint.story_framework.name, "Problem-Solution")
        self.assertEqual(prob_blueprint.emotion_curve.pattern, "Hook-Dip-Rise")
        self.assertIn("problem, how-to, or cautionary warning", prob_blueprint.metadata["decision_reasoning"]["story_framework"])

        # 3. Listicle selection
        list_topic = Topic(num="004", status="backlog", title="3 Secrets of High-Yield Savings Accounts", niche="banking", angle="High yield details", guardrail="None")
        list_blueprint = make_creative_blueprint(list_topic)
        self.assertEqual(list_blueprint.story_framework.name, "Listicle/Three-Facts")
        self.assertEqual(list_blueprint.emotion_curve.pattern, "Steady-Build")
        self.assertIn("tips/secrets/facts", list_blueprint.metadata["decision_reasoning"]["story_framework"])

    def test_blueprint_serialization(self):
        topic = Topic(num="001", status="backlog", title="Why Credit Score Matters", niche="personal finance", angle="FICO mechanism", guardrail="None")
        blueprint = make_creative_blueprint(topic)
        b_dict = blueprint.to_dict()

        self.assertIsInstance(b_dict, dict)
        self.assertEqual(b_dict["audience"], blueprint.audience)
        self.assertEqual(b_dict["story_framework"]["name"], "Problem-Solution")
        self.assertEqual(b_dict["metadata"]["creative_version"], "1.0")

        # Test JSON serialization of serialized dictionary
        dumped = json.dumps(b_dict)
        loaded = json.loads(dumped)
        self.assertEqual(loaded["audience"], blueprint.audience)

    def test_learning_and_strategy_integration(self):
        topic = Topic(num="001", status="backlog", title="Testing Pacing", niche="health", angle="pacing check", guardrail="None")
        
        # Mock learning summary suggesting fast pacing
        mock_summary = MagicMock()
        mock_summary.recommendations = "We recommend fast dynamic pacing for higher retention."
        mock_summary.best_hook = "shock"

        # Variation with pacing adjustment
        variation = VariationRecord(date="2026-07-08", slug="daily_001", title="T01", fmt="F1", skin="S1", voice="V1", len="L1", pace="+10%", cluster="health", hook="shock")
        decision = StrategyDecision(
            topic=topic,
            hook="shock",
            voice="V1",
            skin="S1",
            format="F1",
            confidence=0.9,
            reasoning="Pace increase requested",
            variation=variation
        )

        blueprint = make_creative_blueprint(topic, decision, learning_summary=mock_summary)
        
        # Verify L1 (30s) maps target_duration to 30.0s and target_scene_count to 3
        self.assertEqual(blueprint.target_duration, 30.0)
        self.assertEqual(blueprint.target_scene_count, 3)

        # Verify pacing and hook strategy match parameters
        self.assertEqual(blueprint.pacing, "Fast/Dynamic")
        self.assertIn("Shocking Statistic", blueprint.hook_strategy)

    @patch("clippilot.brain.script_generator.get_provider")
    @patch("clippilot.brain.env.has_api_key")
    def test_script_generator_with_blueprint(self, mock_has_key, mock_get_provider):
        mock_has_key.return_value = True

        mock_provider = MagicMock()
        mock_provider.generate_text.return_value = '{\n  "title": "Blueprint Guided Video",\n  "hook": "Check this!",\n  "niche_context": "testing",\n  "scenes": [{"narration": "First scene", "visual_desc": "First visual"}]\n}'
        mock_provider.last_usage = {"input_tokens": 100, "output_tokens": 50, "latency": 1.0, "estimated_cost": 0.001}
        mock_get_provider.return_value = mock_provider

        topic = Topic(num="001", status="backlog", title="Why Credit Score Matters", niche="finance", angle="FICO", guardrail="None")
        blueprint = make_creative_blueprint(topic)
        variation = VariationRecord(date="2026-07-08", slug="daily_001", title="T01", fmt="F1", skin="S1", voice="V1", len="L3", pace="0%", cluster="finance", hook="question")
        state = PipelineState()

        # Call generate_script passing the blueprint
        script = generate_script(state, topic, variation, Path("."), retries=1, fallback_to_mock=False, blueprint=blueprint)

        self.assertEqual(script.title, "Blueprint Guided Video")

        # Verify that prompt builder was called and injected blueprint instructions
        last_call_args = mock_provider.generate_text.call_args[1]
        user_prompt_content = last_call_args.get("prompt", "")
        
        self.assertIn("STRICT ADHERENCE TO THE FOLLOWING CREATIVE BLUEPRINT IS REQUIRED", user_prompt_content)
        self.assertIn("Target Audience:", user_prompt_content)
        self.assertIn("Story Framework: Use the 'Problem-Solution' framework", user_prompt_content)

    @patch("clippilot.brain.pipeline_demo.load_state")
    @patch("clippilot.brain.pipeline_demo.generate_script")
    @patch("clippilot.brain.pipeline_demo.plan_scenes")
    @patch("clippilot.brain.asset_intelligence.AssetIntelligenceEngine")
    @patch("clippilot.brain.pipeline_demo.get_publisher")
    @patch("clippilot.brain.pipeline_demo.get_voice_provider")
    @patch("clippilot.brain.pipeline_demo.run_vision_qa")
    @patch("subprocess.run")
    def test_pipeline_integration(self, mock_sub_run, mock_run_qa, mock_get_voice, mock_get_publisher, mock_assets_class, mock_plan, mock_script, mock_load):
        # Mock all E2E pipeline stages to verify Stage 2B integration and reporting
        from clippilot.brain.pipeline_demo import execute_production_pipeline
        from clippilot.brain.publisher import PublishResult
        from clippilot.brain.voice_provider import VoiceAlignment, VoiceResult
        from clippilot.brain.scene_planner import ScenePlan, ScenePlanScene

        # Mock objects
        state = PipelineState()
        topic = Topic(num="006", status="unused", title="Closing Card", niche="credit", angle="Stop closing", guardrail="safe")
        state.topics = [topic]
        mock_load.return_value = state
        
        script_obj = Script(topic_num="006", title="Mock Title", hook="Hook", niche_context="niche", scenes=[], metadata={"provider": "mock"})
        mock_script.return_value = script_obj
        
        segment = ScenePlanScene(
            scene_index=1,
            narration="Line.",
            duration_seconds=1.5,
            subtitle_words=["Line."],
            subtitle_timings=[(0.0, 1.5)],
            background_id="bg_1",
            chart_id="chart_1",
        )
        mock_plan.return_value = ScenePlan(
            topic_num="006",
            title="Closing Card",
            hook="Stop",
            skin_id="S1",
            format_id="F1",
            voice_id="V1",
            scenes=[segment],
            total_duration_seconds=1.5,
        )
        
        asset_engine_mock = MagicMock()
        asset_plan_mock = MagicMock()
        asset_plan_mock.metadata = {"cost": 0.0}
        asset_engine_mock.build_asset_plan.return_value = asset_plan_mock
        mock_assets_class.return_value = asset_engine_mock
        
        publisher_mock = MagicMock()
        publisher_mock.publish.return_value = PublishResult(success=True, video_id="yt123", url="http://youtube")
        mock_get_publisher.return_value = publisher_mock

        voice_provider_mock = MagicMock()
        voice_provider_mock.generate_voice.return_value = VoiceResult(
            audio_path="mock.mp3",
            duration_seconds=1.5,
            alignments=[VoiceAlignment("Line.", 0.0, 1.5)],
            sentences_timings=[("Line.", 0.0, 1.5)]
        )
        mock_get_voice.return_value = voice_provider_mock
        
        from clippilot.brain.vision_qa import QAResult
        mock_run_qa.return_value = QAResult(passed=True, score=100.0, blank_frame_detected=False, subtitle_clipping_detected=False)
        mock_sub_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create a mock folder hierarchy
            remotion_explainer_dir = tmp_path / "ClipPilot" / "remotion_explainer"
            remotion_explainer_dir.mkdir(parents=True, exist_ok=True)
            (remotion_explainer_dir / "src").mkdir(parents=True, exist_ok=True)
            (remotion_explainer_dir / "public" / "audio" / "voice").mkdir(parents=True, exist_ok=True)

            # Create a mock Root.tsx so it can be backed up
            (remotion_explainer_dir / "src" / "Root.tsx").write_text("// dummy root", encoding="utf-8")

            # Run E2E pipeline
            report = execute_production_pipeline(
                workspace_dir=tmp_path,
                variation_dir=tmp_path / "variation",
                date_str="2026-07-08",
                slug="daily_006",
                skip_qa_publish=False
            )

            # Verify Stage 2B injected the CreativeBlueprint into the pipeline report
            self.assertTrue(report["success"])
            self.assertIn("creative_blueprint", report)
            blueprint_data = report["creative_blueprint"]
            
            self.assertEqual(blueprint_data["metadata"]["creative_version"], "1.0")
            self.assertIn("story_framework", blueprint_data)
            
            # Verify downstream stages received it (i.e. generate_script was called with the blueprint keyword arg)
            mock_script.assert_called_once()
            kwargs = mock_script.call_args[1]
            self.assertIn("blueprint", kwargs)
            self.assertIsInstance(kwargs["blueprint"], CreativeBlueprint)


if __name__ == "__main__":
    unittest.main()
