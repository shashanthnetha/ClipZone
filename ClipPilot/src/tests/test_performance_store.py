# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Performance Store."""
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
from clippilot.brain.performance_store import PerformanceStore


class TestPerformanceStore(unittest.TestCase):
    """Tests JSONL persistence, lookup, listings, updates, and schema integrity."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "test_performance_history.jsonl"
        self.store = PerformanceStore(self.store_path)

        # Build mock structures
        self.gen = GenerationMetrics(
            llm_provider="anthropic",
            llm_model="claude-test",
            script_critic_score=8.5,
            vision_qa_score=98.0,
            render_time_seconds=2.45,
            total_cost_usd=0.0135,
            script_metadata={"critic": "done"},
            stage_timings={"stage_1": 0.1, "stage_2": 0.2},
        )

        self.up = UploadMetrics(
            platform="youtube",
            upload_success=True,
            video_url="https://youtube.com/watch?v=123",
            video_id_on_platform="123",
            duration_seconds=15.0,
            resolution_width=1080,
            resolution_height=1920,
            fps=30,
            file_size_bytes=512000,
            output_path="/tmp/video.mp4",
        )

        self.an = AnalyticsMetrics(
            views=100,
            likes=10,
            shares=2,
            comments=1,
            retention_rate=0.75,
            ctr=0.05,
            impressions=2000,
            avg_view_duration_seconds=11.2,
            avg_percentage_viewed=74.6,
            watch_time_hours=0.31,
            subscribers_gained=5,
            revenue_usd=0.15,
        )

        self.record1 = VideoPerformance(
            video_id="video_001",
            timestamp="2026-07-03T12:00:00Z",
            topic={"num": "001", "title": "Topic 1"},
            variation={"title": "T01", "fmt": "F1", "voice": "V1", "skin": "S1", "hook": "H1", "cluster": "C1"},
            generation_metrics=self.gen,
            upload_metrics=self.up,
            analytics_metrics=self.an,
            schema_version=1,
            pipeline_version="1.0.0",
            git_commit="hash123",
        )

        self.record2 = VideoPerformance(
            video_id="video_002",
            timestamp="2026-07-03T13:00:00Z",
            topic={"num": "002", "title": "Topic 2"},
            variation={"title": "T02", "fmt": "F2", "voice": "V2", "skin": "S2", "hook": "H2", "cluster": "C2"},
            generation_metrics=self.gen,
            upload_metrics=self.up,
            analytics_metrics=self.an,
            schema_version=1,
            pipeline_version="1.0.0",
            git_commit="hash123",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save_and_load_records(self) -> None:
        self.store.save_record(self.record1)
        self.store.save_record(self.record2)

        loaded = self.store.load_records()
        self.assertEqual(len(loaded), 2)
        
        self.assertEqual(loaded[0].video_id, "video_001")
        self.assertEqual(loaded[0].git_commit, "hash123")
        self.assertEqual(loaded[0].schema_version, 1)
        self.assertEqual(loaded[0].variation["voice"], "V1")
        self.assertEqual(loaded[0].generation_metrics.llm_model, "claude-test")
        self.assertEqual(loaded[0].upload_metrics.duration_seconds, 15.0)
        self.assertEqual(loaded[0].analytics_metrics.views, 100)
        self.assertEqual(loaded[0].analytics_metrics.revenue_usd, 0.15)
        
        self.assertEqual(loaded[1].video_id, "video_002")

    def test_find_record(self) -> None:
        self.store.save_record(self.record1)
        self.store.save_record(self.record2)

        record = self.store.find_record("video_002")
        self.assertIsNotNone(record)
        self.assertEqual(record.video_id, "video_002")
        self.assertEqual(record.topic["title"], "Topic 2")

        not_found = self.store.find_record("video_nonexistent")
        self.assertIsNone(not_found)

    def test_list_records(self) -> None:
        self.store.save_record(self.record1)
        records = self.store.list_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].video_id, "video_001")

    def test_update_record(self) -> None:
        self.store.save_record(self.record1)
        
        # Modify and update
        updated_record = self.record1
        updated_record.generation_metrics.script_critic_score = 9.9
        self.store.update_record(updated_record)

        loaded = self.store.load_records()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].generation_metrics.script_critic_score, 9.9)

        # Update non-existing appends it
        self.store.update_record(self.record2)
        self.assertEqual(len(self.store.load_records()), 2)
