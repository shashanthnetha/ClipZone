# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Publisher Abstraction."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from clippilot.brain.publisher import (
    Metadata,
    PublishRequest,
    PublishResult,
    Thumbnail,
    UploadProgress,
    YouTubePublisher,
    get_publisher,
)


class TestPublisher(unittest.TestCase):
    """Tests for metadata verification, thumbnail paths, progress reporting, and dry runs."""

    def test_get_publisher_resolution(self) -> None:
        pub = get_publisher("youtube")
        self.assertIsInstance(pub, YouTubePublisher)

        with self.assertRaises(NotImplementedError):
            get_publisher("tiktok")

        with self.assertRaises(ValueError):
            get_publisher("unknown_platform")

    def test_validation_errors(self) -> None:
        pub = YouTubePublisher()

        # Title empty error
        req = PublishRequest(
            video_path="dummy.mp4",
            metadata=Metadata(title="", description="Desc"),
            dry_run=True,
        )
        with self.assertRaises(ValueError) as ctx:
            pub.validate_request(req)
        self.assertIn("title is required", str(ctx.exception))

        # Title too long error (>100 chars)
        req = PublishRequest(
            video_path="dummy.mp4",
            metadata=Metadata(title="A" * 101, description="Desc"),
            dry_run=True,
        )
        with self.assertRaises(ValueError) as ctx:
            pub.validate_request(req)
        self.assertIn("100 characters or less", str(ctx.exception))

        # Missing thumbnail error
        req = PublishRequest(
            video_path="dummy.mp4",
            metadata=Metadata(title="Good Title"),
            thumbnail=Thumbnail(image_path="nonexistent_thumb.png"),
            dry_run=True,
        )
        with self.assertRaises(FileNotFoundError):
            pub.validate_request(req)

    def test_publish_dry_run_success(self) -> None:
        pub = YouTubePublisher()
        req = PublishRequest(
            video_path="dummy.mp4",
            metadata=Metadata(title="Dry Run Video", privacy_status="public"),
            scheduled_publish_time="2026-07-04T12:00:00Z",
            dry_run=True,
        )

        progresses = []

        def progress_cb(p: UploadProgress) -> None:
            progresses.append(p)

        res = pub.publish(req, progress_callback=progress_cb)

        self.assertTrue(res.success)
        self.assertEqual(res.video_id, "dry_run_yt_123")
        self.assertEqual(res.published_at, "2026-07-04T12:00:00Z")
        self.assertEqual(len(progresses), 3)
        self.assertEqual(progresses[0].status, "initiating")
        self.assertEqual(progresses[2].status, "completed")

    def test_publish_cancellation_token(self) -> None:
        pub = YouTubePublisher()
        req = PublishRequest(
            video_path="dummy.mp4",
            metadata=Metadata(title="Cancelled Video"),
            dry_run=True,
        )

        res = pub.publish(req, cancel_check=lambda: True)
        self.assertFalse(res.success)
        self.assertIn("Upload cancelled", res.error)

    @patch("clippilot.publish.youtube.YouTubePublisher.upload_video")
    def test_publish_real_client_success(self, mock_upload_video: MagicMock) -> None:
        mock_upload_video.return_value = {
            "success": True,
            "video_id": "yt_real_999",
        }

        pub = YouTubePublisher(
            client_id="id",
            client_secret="sec",
            refresh_token="ref",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_vid = Path(tmp_dir) / "video.mp4"
            temp_vid.write_text("fake video bytes")

            req = PublishRequest(
                video_path=str(temp_vid),
                metadata=Metadata(title="Real Client Title"),
                dry_run=False,
            )

            res = pub.publish(req)
            self.assertTrue(res.success)
            self.assertEqual(res.video_id, "yt_real_999")
            self.assertEqual(res.url, "https://www.youtube.com/watch?v=yt_real_999")
            mock_upload_video.assert_called_once()
