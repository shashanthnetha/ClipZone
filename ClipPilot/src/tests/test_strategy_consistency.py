# -*- coding: utf-8 -*-
"""Integration tests verifying end-to-end Strategy Engine topic consistency."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clippilot.brain.analytics_models import AnalyticsMetrics, VideoPerformance, GenerationMetrics, UploadMetrics
from clippilot.brain.learning_engine import LearningEngine
from clippilot.brain.performance_store import PerformanceStore
from clippilot.brain.pipeline_orchestrator import PipelineState, Topic, VariationRecord
from clippilot.brain.strategy_engine import StrategyEngine
from clippilot.brain.pipeline_demo import execute_production_pipeline


class TestStrategyConsistency(unittest.TestCase):
    """Verifies that the Strategy Engine selections propagate cleanly without static overrides."""

    def setUp(self) -> None:
        import os
        self._keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "LLM_API_KEY")
        self._saved = {k: os.environ.pop(k, None) for k in self._keys}

        from clippilot.brain import env
        self.orig_load_dotenv = env.load_dotenv
        env.load_dotenv = lambda *a, **k: None

        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        
        # Create standard directory tree
        self.clip_dir = self.root_dir / "ClipPilot"
        self.clip_dir.mkdir()
        self.explainer_dir = self.clip_dir / "remotion_explainer"
        self.explainer_dir.mkdir()
        (self.explainer_dir / "src").mkdir()
        (self.explainer_dir / "public").mkdir()
        
        self.var_dir = self.root_dir / "variation"
        self.var_dir.mkdir()
        
        # Populate variation assets
        (self.var_dir / "title_shapes.md").write_text("Titles: T01, T02\n", encoding="utf-8")
        (self.var_dir / "formats.md").write_text("Formats: F1, F2\n", encoding="utf-8")
        (self.var_dir / "visual_skins.md").write_text("Skins: S1, S2\n", encoding="utf-8")
        (self.var_dir / "voices.md").write_text("Voices: V1, V2\n", encoding="utf-8")

        # Initialize mock topics in backlog: topic 010
        # Topic fields: num, status, title, niche, angle, guardrail
        self.topics_md = (
            "| num | status | title | niche | angle | guardrail |\n"
            "|---|---|---|---|---|---|\n"
            "| 010 | unused | Buy-now-pay-later is on your credit report now | credit | buy-now-pay-later affects credit score | review report |\n"
        )
        (self.root_dir / "daily_topics.md").write_text(self.topics_md, encoding="utf-8")
        (self.root_dir / "daily_posts_ledger.md").write_text("", encoding="utf-8")
        (self.root_dir / "daily_variation_ledger.md").write_text("", encoding="utf-8")

        # Seed performance store to bypass cold start (needs 3 records)
        self.store_path = self.clip_dir / "performance_history.jsonl"
        self.store = PerformanceStore(self.store_path)
        
        # Create mock records
        for i in range(3):
            rec = VideoPerformance(
                video_id=f"daily_00{i}_xyz",
                timestamp="2026-07-03T12:00:00Z",
                topic={"num": f"00{i}", "title": "Mock", "niche": "credit"},
                variation={"title": "T01", "fmt": "F1", "voice": "V1", "skin": "S1", "hook": "question", "cluster": "credit"},
                generation_metrics=GenerationMetrics(
                    llm_provider="m", llm_model="m", script_critic_score=8.0,
                    vision_qa_score=95.0, render_time_seconds=0.1, total_cost_usd=0.0
                ),
                upload_metrics=UploadMetrics(
                    platform="youtube", upload_success=True, video_url="", video_id_on_platform="", duration_seconds=10.0
                ),
                analytics_metrics=AnalyticsMetrics(views=100, likes=10, ctr=0.05, retention_rate=0.5, avg_view_duration_seconds=5.0),
                schema_version=1,
                pipeline_version="1.0.0",
                git_commit="hash",
                strategy_metadata={}
            )
            self.store.save_record(rec)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @patch("clippilot.brain.pipeline_demo.get_voice_provider")
    @patch("clippilot.brain.pipeline_demo.render_to_remotion")
    @patch("subprocess.run")
    def test_pipeline_topic_consistency(self, mock_sub_run, mock_render, mock_voice) -> None:
        # Mock TTS voice provider
        from unittest.mock import MagicMock
        mock_provider = MagicMock()
        mock_provider.generate_voice.return_value = MagicMock(duration_seconds=5.0, alignments=[])
        mock_voice.return_value = mock_provider

        # Mock Remotion TSX compilation outputs
        mock_render_res = MagicMock()
        mock_render_res.react_component_tree = "ReactTree"
        mock_render_res.remotion_composition = "Composition"
        mock_render.return_value = mock_render_res

        # Mock Subprocess call
        mock_sub_run.return_value = MagicMock(returncode=0, stdout=b"Success", stderr=b"")

        # Run pipeline E2E
        report = execute_production_pipeline(
            workspace_dir=self.root_dir,
            variation_dir=self.var_dir,
            date_str="2026-07-01",
            slug="daily_006"  # default, should be overridden
        )

        self.assertTrue(report["success"])

        # Check recorded performance record
        records = self.store.load_records()
        # Find the newly added record (the 4th record)
        self.assertEqual(len(records), 4)
        new_record = records[-1]

        # 1. Verify Topic ID consistency
        self.assertEqual(new_record.topic["num"], "010")
        self.assertEqual(new_record.topic["title"], "Buy-now-pay-later is on your credit report now")

        # 2. Verify Output Filename consistency
        output_filename = Path(new_record.upload_metrics.output_path).name
        self.assertEqual(output_filename, "daily_010.mp4")

        # 3. Verify Video ID prefix consistency
        self.assertTrue(new_record.video_id.startswith("daily_010_"))

        # 4. Verify Strategy Metadata exists and is correct
        self.assertGreater(new_record.strategy_metadata["final_ucb_score"], 0.0)
        self.assertGreaterEqual(new_record.strategy_metadata["component_scores"]["topic"], 0.0)

        # 5. Verify derived title propagates to script
        title_lower = new_record.generation_metrics.script_metadata.get("title", "").lower().replace("-", " ")
        self.assertTrue("buy now pay later" in title_lower or "bnpl" in title_lower, f"Expected topic keywords in title: {title_lower}")
        self.assertNotIn("Stop Closing Credit Cards", new_record.generation_metrics.script_metadata.get("title", ""))

    def tearDown(self) -> None:
        from clippilot.brain import env
        env.load_dotenv = self.orig_load_dotenv
        import os
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
        self.temp_dir.cleanup()
