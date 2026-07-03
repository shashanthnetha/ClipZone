# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Learning Engine."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clippilot.brain.analytics_models import (
    AnalyticsMetrics,
    GenerationMetrics,
    UploadMetrics,
    VideoPerformance,
)
from clippilot.brain.learning_engine import LearningEngine
from clippilot.brain.performance_store import PerformanceStore


class TestLearningEngine(unittest.TestCase):
    """Verifies descriptive aggregation logic, Pearson correlations, and recommendations."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "test_perf_store.jsonl"
        self.store = PerformanceStore(self.store_path)
        self.engine = LearningEngine(self.store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_mock_record(self, video_id: str, hook: str, voice: str, skin: str,
                             views: int, ctr: float, retention: float, watch_time: float,
                             render_time: float, critic_score: float, vision_qa_score: float) -> VideoPerformance:
        return VideoPerformance(
            video_id=video_id,
            timestamp="2026-07-03T12:00:00Z",
            topic={"num": "001", "title": "Topic 1", "niche": "finance"},
            variation={"title": "T1", "fmt": "fmt_short", "voice": voice, "skin": skin, "hook": hook, "cluster": "finance"},
            generation_metrics=GenerationMetrics(
                llm_provider="mock", llm_model="mock", script_critic_score=critic_score,
                vision_qa_score=vision_qa_score, render_time_seconds=render_time, total_cost_usd=0.0
            ),
            upload_metrics=UploadMetrics(
                platform="youtube", upload_success=True, video_url="http://",
                video_id_on_platform=video_id + "_yt", duration_seconds=15.0
            ),
            analytics_metrics=AnalyticsMetrics(
                views=views, likes=int(views * 0.1), comments=int(views * 0.01),
                ctr=ctr, retention_rate=retention, avg_view_duration_seconds=watch_time
            ),
            schema_version=1,
            pipeline_version="1.0.0",
            git_commit="git123",
        )

    def test_empty_history(self) -> None:
        self.engine.load_history()
        summary = self.engine.generate_summary()
        self.assertEqual(summary.records_analyzed, 0)
        self.assertEqual(summary.best_hook, "unknown")
        self.assertEqual(summary.recommendations.confidence_score, 0.0)
        self.assertEqual(summary.correlations["critic_score_vs_retention"], 0.0)

    def test_single_record(self) -> None:
        rec = self._create_mock_record(
            "v_1", "question", "voice_a", "skin_dark",
            views=1000, ctr=0.05, retention=0.6, watch_time=9.0,
            render_time=5.0, critic_score=8.0, vision_qa_score=95.0
        )
        self.store.save_record(rec)
        
        self.engine.load_history()
        summary = self.engine.generate_summary()
        
        self.assertEqual(summary.records_analyzed, 1)
        self.assertEqual(summary.best_hook, "question")
        self.assertEqual(summary.best_voice, "voice_a")
        self.assertEqual(summary.best_skin, "skin_dark")
        self.assertEqual(summary.overall_insights.avg_ctr, 0.05)
        self.assertEqual(summary.recommendations.preferred_hook, "question")
        # Under 2 records -> correlation is always 0.0
        self.assertEqual(summary.correlations["critic_score_vs_retention"], 0.0)

    def test_multiple_records_and_ranking(self) -> None:
        # Hook 'question' (CTR: 0.05, 0.07) -> Avg CTR: 0.06
        # Hook 'claim' (CTR: 0.09) -> Avg CTR: 0.09
        rec1 = self._create_mock_record("v1", "question", "v_a", "s_dark", 1000, 0.05, 0.5, 7.5, 4.0, 8.0, 95.0)
        rec2 = self._create_mock_record("v2", "question", "v_b", "s_light", 2000, 0.07, 0.6, 9.0, 6.0, 8.5, 96.0)
        rec3 = self._create_mock_record("v3", "claim", "v_a", "s_dark", 5000, 0.09, 0.7, 10.5, 3.0, 9.0, 98.0)
        
        self.store.save_record(rec1)
        self.store.save_record(rec2)
        self.store.save_record(rec3)

        self.engine.load_history()
        summary = self.engine.generate_summary()

        self.assertEqual(summary.records_analyzed, 3)
        
        # Best Hook by CTR should be 'claim' (0.09 vs 0.06)
        self.assertEqual(summary.best_hook, "claim")
        self.assertEqual(summary.top_hooks[0]["name"], "claim")
        self.assertEqual(summary.top_hooks[1]["name"], "question")
        
        # Best Skin by Retention should be 's_dark' (Avg retention: (0.5+0.7)/2 = 0.6 vs 's_light' = 0.6.
        # Wait, s_dark retention avg = 0.6. s_light retention avg = 0.6.
        # Tie breaker: count. s_dark count is 2, s_light count is 1. So s_dark wins!
        self.assertEqual(summary.best_skin, "s_dark")

        # Fastest rendering config check (render_time: rec3 is 3.0, rec1 is 4.0, rec2 is 6.0)
        self.assertEqual(summary.fastest_render_config["video_id"], "v3")
        self.assertEqual(summary.fastest_render_config["render_time_seconds"], 3.0)

    def test_correlation_calculations(self) -> None:
        # Critic score & Retention are perfectly linear:
        # Critic: 7.0, 8.0, 9.0
        # Retention: 0.4, 0.6, 0.8
        # Should return a correlation of 1.0
        rec1 = self._create_mock_record("v1", "h1", "v1", "s1", 100, 0.01, 0.4, 6.0, 2.0, 7.0, 90.0)
        rec2 = self._create_mock_record("v2", "h1", "v1", "s1", 100, 0.01, 0.6, 6.0, 2.0, 8.0, 90.0)
        rec3 = self._create_mock_record("v3", "h1", "v1", "s1", 100, 0.01, 0.8, 6.0, 2.0, 9.0, 90.0)

        self.store.save_record(rec1)
        self.store.save_record(rec2)
        self.store.save_record(rec3)

        self.engine.load_history()
        summary = self.engine.generate_summary()

        critic_retention_corr = summary.correlations["critic_score_vs_retention"]
        self.assertAlmostEqual(critic_retention_corr, 1.0, places=4)

    def test_save_summary_writes_valid_json(self) -> None:
        rec = self._create_mock_record(
            "v_1", "question", "voice_a", "skin_dark",
            views=1000, ctr=0.05, retention=0.6, watch_time=9.0,
            render_time=5.0, critic_score=8.0, vision_qa_score=95.0
        )
        self.store.save_record(rec)

        self.engine.load_history()
        out_path_str = self.engine.save_summary()
        out_path = Path(out_path_str)

        self.assertTrue(out_path.exists())

        # Load back to verify keys
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["records_analyzed"], 1)
        self.assertEqual(data["best_hook"], "question")
        self.assertEqual(data["recommendations"]["preferred_hook"], "question")
        self.assertIn("top_hooks", data)
        self.assertIn("top_voices", data)
        self.assertIn("top_skins", data)

    def test_deterministic_output(self) -> None:
        rec1 = self._create_mock_record("v1", "h1", "v1", "s1", 100, 0.02, 0.5, 6.0, 2.0, 8.0, 90.0)
        rec2 = self._create_mock_record("v2", "h2", "v2", "s2", 200, 0.04, 0.6, 7.0, 3.0, 8.5, 92.0)
        self.store.save_record(rec1)
        self.store.save_record(rec2)

        self.engine.load_history()
        sum1 = self.engine.generate_summary()
        sum2 = self.engine.generate_summary()

        self.assertEqual(sum1, sum2)
