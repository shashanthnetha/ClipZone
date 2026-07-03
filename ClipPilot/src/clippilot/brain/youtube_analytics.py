# -*- coding: utf-8 -*-
"""ClipPilot Brain Analytics Import Provider.

Defines the AnalyticsProvider protocol and a concrete YouTubeAnalyticsProvider implementation
using YouTube Data API v3 and YouTube Analytics API v2.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, Optional, Protocol


class AnalyticsProvider(Protocol):
    """Protocol defining the interface for an Analytics provider."""

    def get_video_statistics(self, video_id_on_platform: str) -> Dict[str, Any]:
        """Gets basic statistics (views, likes, comments) for a specific video."""
        ...

    def get_video_analytics(self, video_id_on_platform: str) -> Dict[str, Any]:
        """Gets detailed retention/performance metrics for a specific video."""
        ...


class YouTubeAnalyticsProvider:
    """YouTube-specific analytics provider calling v3 Data API and v2 Analytics API."""

    TOKEN_URL = "https://oauth2.googleapis.com/token"
    DATA_API_URL = "https://www.googleapis.com/youtube/v3/videos"
    ANALYTICS_API_URL = "https://youtubeanalytics.googleapis.com/v2/reports"

    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        if not client_id or not client_secret or not refresh_token:
            raise ValueError("client_id, client_secret, and refresh_token are all required.")
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token

    def _refresh_access_token(self) -> str:
        """Exchanges the refresh token for a short-lived access token."""
        import httpx
        resp = httpx.post(self.TOKEN_URL, data={
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
        }, timeout=30)
        
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to refresh YouTube access token: {resp.text}")
        
        token = resp.json().get("access_token")
        if not token:
            raise RuntimeError("Access token missing from YouTube token response.")
        return token

    def get_video_statistics(self, video_id_on_platform: str) -> Dict[str, Any]:
        """Fetches viewCount, likeCount, and commentCount from YouTube Data API v3."""
        import httpx
        access_token = self._refresh_access_token()
        
        resp = httpx.get(
            self.DATA_API_URL,
            params={
                "id": video_id_on_platform,
                "part": "statistics",
            },
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"YouTube Data API error: {resp.text}")
            
        items = resp.json().get("items", [])
        if not items:
            return {"views": 0, "likes": 0, "comments": 0}
            
        stats = items[0].get("statistics", {})
        return {
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            "comments": int(stats.get("commentCount", 0)),
        }

    def get_video_analytics(self, video_id_on_platform: str) -> Dict[str, Any]:
        """Fetches detailed metrics from YouTube Analytics API v2."""
        import httpx
        access_token = self._refresh_access_token()
        
        # Query from inception to today to catch stats for this video
        start_date = "2006-01-01"
        end_date = datetime.date.today().isoformat()
        
        metrics = (
            "views,likes,comments,impressions,annotationClickThroughRate,"
            "averageViewDuration,averageViewPercentage,watchTime,subscribersGained,estimatedRevenue"
        )
        
        resp = httpx.get(
            self.ANALYTICS_API_URL,
            params={
                "ids": "channel==MINE",
                "startDate": start_date,
                "endDate": end_date,
                "metrics": metrics,
                "filters": f"video=={video_id_on_platform}",
            },
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"YouTube Analytics API error: {resp.text}")
            
        data = resp.json()
        column_headers = data.get("columnHeaders", [])
        rows = data.get("rows", [])
        
        results: Dict[str, Any] = {}
        if rows and len(rows) > 0:
            row = rows[0]
            for idx, header in enumerate(column_headers):
                results[header["name"]] = row[idx]
                
        # YouTube Analytics returns watchTime in minutes; convert to hours
        watch_time_mins = float(results.get("watchTime", 0.0))
        watch_time_hours = round(watch_time_mins / 60.0, 4)
        
        return {
            "views": int(results.get("views", 0)),
            "likes": int(results.get("likes", 0)),
            "comments": int(results.get("comments", 0)),
            "impressions": int(results.get("impressions", 0)),
            "ctr": float(results.get("annotationClickThroughRate", 0.0)),
            "average_view_duration_seconds": float(results.get("averageViewDuration", 0.0)),
            "average_percentage_viewed": float(results.get("averageViewPercentage", 0.0)),
            "watch_time_hours": watch_time_hours,
            "subscribers_gained": int(results.get("subscribersGained", 0)),
            "estimated_revenue_usd": float(results.get("estimatedRevenue", 0.0)),
        }
