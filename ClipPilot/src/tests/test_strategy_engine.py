# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Strategy Engine."""
from __future__ import annotations

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
from clippilot.brain.pipeline_orchestrator import PipelineState, Topic, VariationRecord
from clippilot.brain.strategy_engine import StrategyDecision, StrategyEngine, StrategyWeights


class TestStrategyEngine(unittest.TestCase):
    """Verifies UCB selection, exploration-exploitation balancing, and constraint mapping."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "test_strategy_store.jsonl"
        self.store = PerformanceStore(self.store_path)
        self.learning_engine = LearningEngine(self.store)

        # Config files for variation_dir
        self.var_dir = Path(self.temp_dir.name) / "variation"
        self.var_dir.mkdir()
        (self.var_dir / "title_shapes.md").write_text("Titles: T01, T02\n", encoding="utf-8")
        (self.var_dir / "formats.md").write_text("Formats: F1, F2\n", encoding="utf-8")
        (self.var_dir / "visual_skins.md").write_text("Skins: S1, S2\n", encoding="utf-8")
        (self.var_dir / "voices.md").write_text("Voices: V1, V2\n", encoding="utf-8")

        # Weights config
        self.weights = StrategyWeights(
            exploitation_weight=1.0,
            exploration_weight=1.0,
            fatigue_weight=2.0
        )
        self.engine = StrategyEngine(self.store, self.learning_engine, self.weights)

        # Correct parameter ordering: num, status, title, niche, angle, guardrail
        self.topic1 = Topic("001", "unused", "Credit Cards", "credit", "angle", "guardrail")
        self.topic2 = Topic("002", "unused", "Utilization Rate", "credit", "angle", "guardrail")
        self.state = PipelineState(
            topics=[self.topic1, self.topic2],
            posts_history=[],
            variation_history=[]
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_mock_record(self, video_id: str, hook: str, views: int, ctr: float) -> VideoPerformance:
        return VideoPerformance(
            video_id=video_id,
            timestamp="2026-07-03T12:00:00Z",
            topic={"num": "001", "title": "T1", "niche": "credit"},
            variation={"title": "T01", "fmt": "F1", "voice": "V1", "skin": "S1", "hook": hook, "cluster": "credit"},
            generation_metrics=GenerationMetrics(
                llm_provider="m", llm_model="m", script_critic_score=8.0,
                vision_qa_score=95.0, render_time_seconds=2.0, total_cost_usd=0.0
            ),
            upload_metrics=UploadMetrics(
                platform="youtube", upload_success=True, video_url="", video_id_on_platform="", duration_seconds=10.0
            ),
            analytics_metrics=AnalyticsMetrics(
                views=views, likes=10, comments=1, ctr=ctr, retention_rate=0.5, avg_view_duration_seconds=5.0
            ),
            schema_version=1,
            pipeline_version="1.0.0",
            git_commit="hash",
            strategy_metadata={}
        )

    def _get_variation_history(self) -> list[VariationRecord]:
        return [
            VariationRecord(
                date="",
                slug="",
                title=r.variation["title"],
                fmt=r.variation["fmt"],
                skin=r.variation["skin"],
                voice=r.variation["voice"],
                len="L3",
                pace="0%",
                cluster=r.variation["cluster"],
                hook=r.variation["hook"]
            ) for r in self.store.load_records()
        ]

    def test_cold_start_fallback(self) -> None:
        # 0 records in history -> should fallback to cold-start
        decision = self.engine.make_decision(self.state, self.var_dir)
        self.assertEqual(decision.confidence, 0.20)
        self.assertIn("Cold-start", decision.reasoning)
        self.assertEqual(decision.topic.num, "001")
        self.assertIn(decision.hook, ["question", "whatif", "claim", "story", "number", "mythbust"])

    def test_deterministic_decisions(self) -> None:
        # Seed 3 mock records to bypass cold-start
        rec1 = self._create_mock_record("v1", "question", 100, 0.05)
        rec2 = self._create_mock_record("v2", "question", 120, 0.06)
        rec3 = self._create_mock_record("v3", "claim", 300, 0.08)
        self.store.save_record(rec1)
        self.store.save_record(rec2)
        self.store.save_record(rec3)

        self.state.variation_history = self._get_variation_history()

        dec1 = self.engine.make_decision(self.state, self.var_dir)
        dec2 = self.engine.make_decision(self.state, self.var_dir)

        self.assertEqual(dec1.topic.num, dec2.topic.num)
        self.assertEqual(dec1.hook, dec2.hook)
        self.assertEqual(dec1.voice, dec2.voice)
        self.assertEqual(dec1.metadata, dec2.metadata)

    def test_fatigue_penalty_avoidance(self) -> None:
        # Seed records
        rec1 = self._create_mock_record("v1", "question", 100, 0.05)
        rec2 = self._create_mock_record("v2", "question", 120, 0.05)
        rec3 = self._create_mock_record("v3", "claim", 300, 0.08)
        self.store.save_record(rec1)
        self.store.save_record(rec2)
        self.store.save_record(rec3)

        # Set variation_history to include the hook "claim" in the last run (highest recency penalty)
        v_rec = self._get_variation_history()[-1]
        self.state.variation_history = [v_rec]

        decision = self.engine.make_decision(self.state, self.var_dir)
        # Even though "claim" has the highest historical CTR (0.08 vs 0.05),
        # its high fatigue recency penalty should force it to select another hook (e.g. "question").
        self.assertNotEqual(decision.hook, "claim")

    def test_exploration_bonus_chooses_underexplored(self) -> None:
        """Verifies that an underexplored hook with lower performance gets chosen over a highly-exploited best hook due to the UCB exploration bonus."""
        # Hook 'question' (CTR 0.08, views 200) was used 14 times.
        # Other hooks 'whatif', 'claim', 'story', 'number' were used 10 times each.
        # Hook 'mythbust' (CTR 0.04, views 100) was used only 1 time.
        idx = 0
        for _ in range(14):
            self.store.save_record(self._create_mock_record(f"v_{idx}", "question", views=200, ctr=0.08))
            idx += 1
            
        for other_hook in ["whatif", "claim", "story", "number"]:
            for _ in range(10):
                self.store.save_record(self._create_mock_record(f"v_{idx}", other_hook, views=120, ctr=0.05))
                idx += 1
            
        self.store.save_record(self._create_mock_record(f"v_{idx}", "mythbust", views=100, ctr=0.04))

        # Build variation history list
        self.state.variation_history = self._get_variation_history()

        # Set high exploration weight to intentionally emphasize exploration
        self.engine.weights = StrategyWeights(
            exploitation_weight=1.0,
            exploration_weight=4.0,  # strong exploration bonus
            fatigue_weight=0.0       # disable fatigue to focus purely on UCB tradeoff
        )

        decision = self.engine.make_decision(self.state, self.var_dir)

        # The UCB exploration bonus for 'mythbust' (N=1) will be very high,
        # and since all other hooks are now well-explored (N >= 10),
        # 'mythbust' must be chosen due to the high exploration weight!
        self.assertEqual(decision.hook, "mythbust")
        self.assertGreater(decision.metadata["exploration_bonus"], 0.0)
