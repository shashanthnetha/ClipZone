# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot End-to-End Production Pipeline."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from clippilot.brain.pipeline_demo import execute_production_pipeline


class TestPipelineDemo(unittest.TestCase):
    """Tests for verifying the complete E2E production pipeline execution run."""

    def setUp(self) -> None:
        import os
        self.old_env = os.environ.copy()
        for key in ["LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", 
                    "PEXELS_API_KEY", "PIXABAY_API_KEY", "UNSPLASH_API_KEY"]:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self) -> None:
        import os
        os.environ.clear()
        os.environ.update(self.old_env)

    @patch("clippilot.brain.pipeline_demo.load_state")
    @patch("clippilot.brain.pipeline_demo.choose_topic")
    @patch("clippilot.brain.pipeline_demo.choose_variation")
    @patch("clippilot.brain.pipeline_demo.generate_script")
    @patch("clippilot.brain.pipeline_demo.plan_scenes")
    @patch("clippilot.brain.pipeline_demo.run_vision_qa")
    @patch("clippilot.brain.pipeline_demo.get_publisher")
    @patch("clippilot.brain.pipeline_demo.get_voice_provider")
    @patch("subprocess.run")
    def test_production_pipeline_stages(
        self,
        mock_sub_run: MagicMock,
        mock_get_voice: MagicMock,
        mock_get_publisher: MagicMock,
        mock_run_qa: MagicMock,
        mock_plan_scenes: MagicMock,
        mock_gen_script: MagicMock,
        mock_choose_var: MagicMock,
        mock_choose_topic: MagicMock,
        mock_load_state: MagicMock,
    ) -> None:
        from clippilot.brain.pipeline_orchestrator import PipelineState, Topic, VariationRecord
        from clippilot.brain.publisher import PublishResult
        from clippilot.brain.scene_planner import ScenePlan, ScenePlanScene
        from clippilot.brain.script_generator import Script, ScriptScene
        from clippilot.brain.vision_qa import QAResult
        from clippilot.brain.voice_provider import VoiceAlignment, VoiceResult

        # Setup mock states
        state = PipelineState()
        mock_load_state.return_value = state

        topic = Topic(
            num="006",
            status="unused",
            title="Closing Card",
            niche="credit",
            angle="Stop closing",
            guardrail="safe",
        )
        mock_choose_topic.return_value = topic

        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "credit", "question")
        mock_choose_var.return_value = variation

        script = Script("006", "Closing Card", "Stop", "credit", [ScriptScene("Line.", "V")])
        mock_gen_script.return_value = script

        # Scene plan segment
        segment = ScenePlanScene(
            scene_index=1,
            narration="Line.",
            duration_seconds=1.5,
            subtitle_words=["Line."],
            subtitle_timings=[(0.0, 1.5)],
            background_id="bg_1",
            chart_id="chart_1",
        )
        mock_plan_scenes.return_value = ScenePlan(
            topic_num="006",
            title="Closing Card",
            hook="Stop",
            skin_id="S1",
            format_id="F1",
            voice_id="V1",
            scenes=[segment],
            total_duration_seconds=1.5,
        )

        # Voice provider mock
        mock_voice_prov = MagicMock()
        mock_get_voice.return_value = mock_voice_prov
        mock_voice_prov.generate_voice.return_value = VoiceResult(
            audio_path="temp_voice.mp3",
            duration_seconds=1.5,
            alignments=[VoiceAlignment("Line.", 0.0, 1.5)],
            sentences_timings=[("Line.", 0.0, 1.5)],
        )

        # QA mock
        mock_run_qa.return_value = QAResult(
            passed=True, score=98, blank_frame_detected=False, subtitle_clipping_detected=False
        )

        # Publisher mock
        mock_pub = MagicMock()
        mock_pub.publish.return_value = PublishResult(
            success=True, video_id="yt_demo_abc", url="https://youtube.com/demo"
        )
        mock_get_publisher.return_value = mock_pub

        # Mock sub run to simulate ffmpeg extract and Remotion render success
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

            report = execute_production_pipeline(tmp_path, tmp_path / "variation", slug="daily_006")

            # Assert report structure and results
            self.assertTrue(report["success"])
            self.assertEqual(report["qa_score"], 98)
            self.assertEqual(report["upload_result"]["video_id"], "yt_demo_abc")
            self.assertEqual(report["providers"]["tts"], "edge-tts")
            self.assertEqual(report["providers"]["publisher"], "youtube-api")
            self.assertIn("total_time_seconds", report)
            self.assertIn("cost_estimate_usd", report)
