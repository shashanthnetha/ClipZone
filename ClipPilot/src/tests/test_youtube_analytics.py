# -*- coding: utf-8 -*-
"""Unit tests for the YouTube Analytics Importer."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from clippilot.brain.analytics_models import (
    AnalyticsMetrics,
    GenerationMetrics,
    UploadMetrics,
    VideoPerformance,
)
from clippilot.brain.performance_store import PerformanceStore
from clippilot.brain.youtube_analytics import YouTubeAnalyticsProvider


class TestYouTubeAnalytics(unittest.TestCase):
    """Verifies YouTube Analytics Provider API mapping, mocking, and store updates."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "test_perf.jsonl"
        self.store = PerformanceStore(self.store_path)

        self.provider = YouTubeAnalyticsProvider(
            client_id="test_client_id",
            client_secret="test_client_secret",
            refresh_token="test_refresh_token"
        )

        # Base mock records
        self.record1 = VideoPerformance(
            video_id="video_001",
            timestamp="2026-07-03T12:00:00Z",
            topic={"num": "001", "title": "Topic 1"},
            variation={"title": "T01", "fmt": "F1", "voice": "V1", "skin": "S1", "hook": "H1", "cluster": "C1"},
            generation_metrics=GenerationMetrics(
                llm_provider="anthropic", llm_model="claude", script_critic_score=8.0,
                vision_qa_score=90.0, render_time_seconds=1.0, total_cost_usd=0.0
            ),
            upload_metrics=UploadMetrics(
                platform="youtube", upload_success=True, video_url="https://youtube.com/watch?v=yt_abc",
                video_id_on_platform="yt_abc", duration_seconds=10.0
            ),
            analytics_metrics=AnalyticsMetrics(),  # starts empty
            schema_version=1,
            pipeline_version="1.0.0",
            git_commit="hash123",
        )
        self.store.save_record(self.record1)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_init_raises_on_missing_creds(self) -> None:
        with self.assertRaises(ValueError):
            YouTubeAnalyticsProvider(client_id="", client_secret="sec", refresh_token="ref")

    @patch("httpx.post")
    def test_refresh_token_failure(self, mock_post) -> None:
        # Mock token call returning error status code
        mock_post.return_value = httpx.Response(
            status_code=400,
            text="Invalid refresh token",
            request=httpx.Request("POST", YouTubeAnalyticsProvider.TOKEN_URL)
        )
        with self.assertRaises(RuntimeError) as context:
            self.provider.get_video_statistics("yt_abc")
        self.assertIn("Failed to refresh YouTube access token", str(context.exception))

    @patch("httpx.post")
    @patch("httpx.get")
    def test_get_video_statistics_success(self, mock_get, mock_post) -> None:
        # Token Mock
        mock_post.return_value = httpx.Response(
            status_code=200,
            json={"access_token": "mock_access_123"},
            request=httpx.Request("POST", YouTubeAnalyticsProvider.TOKEN_URL)
        )
        # Statistics Mock
        stats_resp = {
            "items": [{
                "statistics": {
                    "viewCount": "1250",
                    "likeCount": "95",
                    "commentCount": "12"
                }
            }]
        }
        mock_get.return_value = httpx.Response(
            status_code=200,
            json=stats_resp,
            request=httpx.Request("GET", YouTubeAnalyticsProvider.DATA_API_URL)
        )

        res = self.provider.get_video_statistics("yt_abc")
        self.assertEqual(res["views"], 1250)
        self.assertEqual(res["likes"], 95)
        self.assertEqual(res["comments"], 12)

    @patch("httpx.post")
    @patch("httpx.get")
    def test_get_video_statistics_missing_items(self, mock_get, mock_post) -> None:
        mock_post.return_value = httpx.Response(
            status_code=200, json={"access_token": "t"}, request=httpx.Request("POST", "t")
        )
        mock_get.return_value = httpx.Response(
            status_code=200, json={"items": []}, request=httpx.Request("GET", "t")
        )
        res = self.provider.get_video_statistics("yt_nonexistent")
        self.assertEqual(res, {"views": 0, "likes": 0, "comments": 0})

    @patch("httpx.post")
    @patch("httpx.get")
    def test_get_video_analytics_success(self, mock_get, mock_post) -> None:
        # Token Mock
        mock_post.return_value = httpx.Response(
            status_code=200,
            json={"access_token": "mock_access_123"},
            request=httpx.Request("POST", YouTubeAnalyticsProvider.TOKEN_URL)
        )
        # Analytics Mock
        analytics_resp = {
            "columnHeaders": [
                {"name": "views", "columnType": "METRIC"},
                {"name": "likes", "columnType": "METRIC"},
                {"name": "comments", "columnType": "METRIC"},
                {"name": "impressions", "columnType": "METRIC"},
                {"name": "annotationClickThroughRate", "columnType": "METRIC"},
                {"name": "averageViewDuration", "columnType": "METRIC"},
                {"name": "averageViewPercentage", "columnType": "METRIC"},
                {"name": "watchTime", "columnType": "METRIC"},
                {"name": "subscribersGained", "columnType": "METRIC"},
                {"name": "estimatedRevenue", "columnType": "METRIC"},
            ],
            "rows": [
                [1500, 110, 15, 20000, 0.054, 45.2, 75.0, 1125.0, 8, 1.25]
            ]
        }
        mock_get.return_value = httpx.Response(
            status_code=200,
            json=analytics_resp,
            request=httpx.Request("GET", YouTubeAnalyticsProvider.ANALYTICS_API_URL)
        )

        res = self.provider.get_video_analytics("yt_abc")
        self.assertEqual(res["views"], 1500)
        self.assertEqual(res["likes"], 110)
        self.assertEqual(res["comments"], 15)
        self.assertEqual(res["impressions"], 20000)
        self.assertEqual(res["ctr"], 0.054)
        self.assertEqual(res["average_view_duration_seconds"], 45.2)
        self.assertEqual(res["average_percentage_viewed"], 75.0)
        # 1125 mins / 60 = 18.75 hours
        self.assertEqual(res["watch_time_hours"], 18.75)
        self.assertEqual(res["subscribers_gained"], 8)
        self.assertEqual(res["estimated_revenue_usd"], 1.25)

    def test_update_analytics_by_video_id(self) -> None:
        metrics = AnalyticsMetrics(
            views=100, likes=10, comments=2, last_updated="2026-07-03T15:00:00Z"
        )
        self.store.update_analytics("video_001", metrics)

        updated = self.store.find_record("video_001")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.analytics_metrics.views, 100)
        self.assertEqual(updated.analytics_metrics.last_updated, "2026-07-03T15:00:00Z")

    def test_update_analytics_by_platform_id(self) -> None:
        metrics = AnalyticsMetrics(
            views=200, likes=20, comments=4, last_updated="2026-07-03T16:00:00Z"
        )
        self.store.update_analytics_by_platform_id("yt_abc", metrics)

        updated = self.store.find_record("video_001")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.analytics_metrics.views, 200)
        self.assertEqual(updated.analytics_metrics.last_updated, "2026-07-03T16:00:00Z")

    def test_update_missing_video_id_raises(self) -> None:
        metrics = AnalyticsMetrics()
        with self.assertRaises(ValueError):
            self.store.update_analytics("video_nonexistent", metrics)

    def test_update_missing_platform_id_raises(self) -> None:
        metrics = AnalyticsMetrics()
        with self.assertRaises(ValueError):
            self.store.update_analytics_by_platform_id("yt_nonexistent", metrics)
