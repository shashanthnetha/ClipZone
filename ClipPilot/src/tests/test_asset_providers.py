# -*- coding: utf-8 -*-
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from clippilot.config import Settings
from clippilot.brain.asset_models import VideoAssetPlan, SceneAssetPlan
from clippilot.media.asset_providers import (
    sanitize_query,
    score_video_asset,
    score_image_asset,
    download_assets_for_plan,
)

class TestAssetProviders(unittest.TestCase):
    def setUp(self):
        self.tmp_cache_dir = Path("data/cache/assets")
        self.target_dir = Path("data/test_target_graphics")
        
        # Clean up any leftover test folders
        if self.tmp_cache_dir.exists():
            shutil.rmtree(self.tmp_cache_dir)
        if self.target_dir.exists():
            shutil.rmtree(self.target_dir)

    def tearDown(self):
        # Clean up after tests
        if self.tmp_cache_dir.exists():
            shutil.rmtree(self.tmp_cache_dir)
        if self.target_dir.exists():
            shutil.rmtree(self.target_dir)

    def test_sanitize_query(self):
        self.assertEqual(sanitize_query("Credit Card Payment!"), "credit_card_payment")
        self.assertEqual(sanitize_query("smartphone-bill-payment"), "smartphone_bill_payment")
        self.assertEqual(sanitize_query("  finance  "), "finance")

    def test_score_video_asset(self):
        # Vertical asset
        v_vertical = {
            "width": 1080,
            "height": 1920,
            "duration": 5.0,
            "bitrate": 5000000.0,
            "tags": "finance",
            "creator": "John Doe",
        }
        # Landscape asset
        v_landscape = {
            "width": 1920,
            "height": 1080,
            "duration": 5.0,
            "bitrate": 5000000.0,
            "tags": "finance",
            "creator": "John Doe",
        }
        # Low res
        v_lowres = {
            "width": 320,
            "height": 240,
            "duration": 5.0,
        }
        # Watermarked
        v_watermark = {
            "width": 1080,
            "height": 1920,
            "duration": 5.0,
            "tags": "watermarked stock video",
        }

        self.assertGreater(score_video_asset(v_vertical, 5.0), score_video_asset(v_landscape, 5.0))
        self.assertEqual(score_video_asset(v_lowres, 5.0), -1000.0)
        self.assertEqual(score_video_asset(v_watermark, 5.0), -1000.0)

    def test_score_image_asset(self):
        img_vertical = {
            "width": 1080,
            "height": 1920,
            "tags": "finance",
        }
        img_landscape = {
            "width": 1920,
            "height": 1080,
            "tags": "finance",
        }
        img_lowres = {
            "width": 200,
            "height": 300,
        }

        self.assertGreater(score_image_asset(img_vertical), score_image_asset(img_landscape))
        self.assertEqual(score_image_asset(img_lowres), -1000.0)

    @patch("requests.get")
    def test_provider_priority_and_cache(self, mock_get):
        # Configure settings mock
        settings = Settings(
            pexels_api_key="pex_key",
            pixabay_api_key="pix_key",
            unsplash_api_key="uns_key",
            asset_provider_priority=["pixabay", "pexels"]
        )

        # Mock API responses
        # Pixabay returns empty, Pexels returns a match
        pixabay_response = MagicMock()
        pixabay_response.status_code = 200
        pixabay_response.json.return_value = {"hits": []}

        pexels_response = MagicMock()
        pexels_response.status_code = 200
        pexels_response.json.return_value = {
            "videos": [
                {
                    "width": 1080,
                    "height": 1920,
                    "duration": 5.0,
                    "user": {"name": "Pexels Creator"},
                    "url": "https://pexels.com/video/1",
                    "video_files": [{"width": 1080, "height": 1920, "link": "https://download.pexels.com/video.mp4"}]
                }
            ]
        }

        download_response = MagicMock()
        download_response.status_code = 200
        download_response.content = b"fake-video-bytes"

        # Mock requests.get sequence
        mock_get.side_effect = [pixabay_response, pexels_response, download_response]

        # Setup asset plan
        plan = VideoAssetPlan(scenes=[
            SceneAssetPlan(
                scene_number=1,
                background="dark_finance",
                stock_video_queries=["credit card"],
            )
        ])

        # Run E2E downloader
        download_assets_for_plan(plan, self.target_dir, "S1", settings)

        # Verify priority ordering called: first Pixabay, then Pexels
        self.assertIn("pixabay", mock_get.call_args_list[0][0][0])
        self.assertIn("pexels", mock_get.call_args_list[1][0][0])

        # Verify file downloaded and saved to cache
        cache_video = self.tmp_cache_dir / "pexels" / "credit_card" / "media.mp4"
        cache_meta = self.tmp_cache_dir / "pexels" / "credit_card" / "metadata.json"
        self.assertTrue(cache_video.exists())
        self.assertTrue(cache_meta.exists())

        # Verify copied to target directory
        target_video = self.target_dir / "asset_s1_1.mp4"
        self.assertTrue(target_video.exists())
        self.assertEqual(target_video.read_bytes(), b"fake-video-bytes")

        # Verify metadata attribution details
        with open(cache_meta, "r") as f:
            meta = json.load(f)
            self.assertEqual(meta["provider"], "pexels")
            self.assertEqual(meta["creator"], "Pexels Creator")
            self.assertEqual(meta["license"], "Pexels License")

        # Reset mock to test caching (subsequent run should hit the cache and NOT trigger any requests)
        mock_get.reset_mock()
        mock_get.side_effect = Exception("Should not query API when cached!")

        # Run again
        download_assets_for_plan(plan, self.target_dir, "S1", settings)
        # Verify target file still exists
        self.assertTrue(target_video.exists())

    @patch("requests.get")
    def test_lucide_icon_download(self, mock_get):
        settings = Settings()
        
        svg_mock = MagicMock()
        svg_mock.status_code = 200
        svg_mock.content = b"<svg>fake-icon</svg>"
        mock_get.return_value = svg_mock

        plan = VideoAssetPlan(scenes=[
            SceneAssetPlan(
                scene_number=2,
                background="blueprint_grid",
                icon_queries=["credit-card"]
            )
        ])

        download_assets_for_plan(plan, self.target_dir, "S2", settings)

        # Verify downloaded SVG exists
        target_svg = self.target_dir / "asset_s2_2.svg"
        self.assertTrue(target_svg.exists())
        self.assertEqual(target_svg.read_bytes(), b"<svg>fake-icon</svg>")

    def test_deterministic_mock_fallback(self):
        # Empty settings (no API keys)
        settings = Settings()
        plan = VideoAssetPlan(scenes=[
            SceneAssetPlan(
                scene_number=3,
                background="cream_pop",
                stock_video_queries=["finance"]
            )
        ])

        # Run download without mock exceptions; should complete silently using default fallback behavior
        try:
            download_assets_for_plan(plan, self.target_dir, "S3", settings)
        except Exception as e:
            self.fail(f"Graceful mock fallback crashed: {e}")

        # Target folder should be empty since no keys are configured
        target_video = self.target_dir / "asset_s3_3.mp4"
        self.assertFalse(target_video.exists())

if __name__ == "__main__":
    unittest.main()
