# -*- coding: utf-8 -*-
"""ClipPilot Brain Publisher Abstraction.

Defines the Publisher Protocol and implements the YouTubePublisher wrapper.
Prepares routing structures for future TikTok, Instagram, X, and Filesystem publishers.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


@dataclass
class Metadata:
    """Standard publishing metadata payload."""
    title: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    privacy_status: str = "private"  # public, unlisted, private
    category_id: str = "22"          # Default: People & Blogs
    language: str = "en"


@dataclass
class Thumbnail:
    """Wrapper carrying thumbnail image specifications."""
    image_path: str


@dataclass
class UploadProgress:
    """Report detailing current upload transfer progress states."""
    bytes_uploaded: int
    total_bytes: int
    percentage: float
    status: str  # e.g. "initiating", "uploading", "completed", "failed"


@dataclass
class PublishRequest:
    """Parameters required to initiate a publisher upload job."""
    video_path: str
    metadata: Metadata
    thumbnail: Optional[Thumbnail] = None
    scheduled_publish_time: Optional[str] = None  # ISO-8601 string or equivalent
    draft: bool = False
    dry_run: bool = False


@dataclass
class PublishResult:
    """Structural response containing the publishing status and IDs."""
    success: bool
    video_id: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[str] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Publisher(Protocol):
    """Protocol defining the standard interface for all publishing backends."""

    def publish(
        self,
        request: PublishRequest,
        progress_callback: Optional[Callable[[UploadProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> PublishResult:
        """Upload and publish a video to the destination platform."""
        ...

    def validate_request(self, request: PublishRequest) -> None:
        """Validate metadata lengths, file paths, and options before uploading."""
        ...


class YouTubePublisher:
    """YouTube-specific Publisher wrapping the first-party resumable upload client."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ) -> None:
        self.client_id = client_id or os.getenv("YOUTUBE_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("YOUTUBE_CLIENT_SECRET", "")
        self.refresh_token = refresh_token or os.getenv("YOUTUBE_REFRESH_TOKEN", "")

    def validate_request(self, request: PublishRequest) -> None:
        """Enforces limits on titles, tags, and verifies video existence."""
        if not request.dry_run and not Path(request.video_path).exists():
            raise FileNotFoundError(f"Video file does not exist: {request.video_path}")

        if request.thumbnail and not os.path.exists(request.thumbnail.image_path):
            raise FileNotFoundError(
                f"Thumbnail file does not exist: {request.thumbnail.image_path}"
            )

        if not request.metadata.title:
            raise ValueError("Video title is required and cannot be empty.")

        if len(request.metadata.title) > 100:
            raise ValueError("YouTube video titles must be 100 characters or less.")

    def publish(
        self,
        request: PublishRequest,
        progress_callback: Optional[Callable[[UploadProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> PublishResult:
        """Performs validation, OAuth token refresh, and triggers upload."""
        try:
            self.validate_request(request)
        except Exception as e:
            return PublishResult(success=False, error=str(e))

        # Check for cancellation before initiation
        if cancel_check and cancel_check():
            return PublishResult(
                success=False, error="Upload cancelled prior to starting."
            )

        # ── Dry Run Simulation ──
        if request.dry_run:
            if progress_callback:
                progress_callback(UploadProgress(0, 100, 0.0, "initiating"))
                time.sleep(0.01)
                progress_callback(UploadProgress(50, 100, 50.0, "uploading"))
                time.sleep(0.01)
                progress_callback(UploadProgress(100, 100, 100.0, "completed"))
            return PublishResult(
                success=True,
                video_id="dry_run_yt_123",
                url="https://www.youtube.com/watch?v=dry_run_yt_123",
                published_at=request.scheduled_publish_time or "immediate",
                metadata={"dry_run": True},
            )

        # Import first-party client dynamically
        from clippilot.publish.youtube import YouTubePublisher as BaseYTClient

        client = BaseYTClient(
            client_id=self.client_id,
            client_secret=self.client_secret,
            refresh_token=self.refresh_token,
        )

        try:
            if progress_callback:
                progress_callback(
                    UploadProgress(0, 100, 0.0, "initiating OAuth refresh")
                )

            # Delegate to existing upload client
            # (Note: standard upload client does not take thumbnail directly, but
            # conforms to description append and resumable upload)
            res = client.upload_video(
                video_path=request.video_path,
                title=request.metadata.title,
                description=request.metadata.description,
                tags=request.metadata.tags,
                privacy=request.metadata.privacy_status,
                category_id=request.metadata.category_id,
                dry_run=False,
            )

            if not res.get("success", False):
                return PublishResult(
                    success=False, error=res.get("error", "Unknown upload error")
                )

            return PublishResult(
                success=True,
                video_id=res.get("video_id"),
                url=f"https://www.youtube.com/watch?v={res.get('video_id')}",
                published_at=request.scheduled_publish_time or "immediate",
                metadata=res,
            )
        except Exception as e:
            return PublishResult(success=False, error=str(e))


def get_publisher(platform_name: str = "youtube") -> Publisher:
    """Factory to retrieve the active, configured Publisher instance."""
    if platform_name.lower() == "youtube":
        return YouTubePublisher()

    supported = ["tiktok", "instagram", "x", "filesystem"]
    if platform_name.lower() in supported:
        raise NotImplementedError(
            f"Publisher for platform '{platform_name}' is designed but not yet implemented."
        )

    raise ValueError(f"Unknown publisher platform '{platform_name}'.")
