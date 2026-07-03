# -*- coding: utf-8 -*-
"""ClipPilot Brain Analytics Data Models.

Defines the structured data classes representing generation metrics, upload states,
viewer engagement analytics, and aggregate learning memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationMetrics:
    """Detailed performance and resource utilization metrics from script generation and rendering."""
    llm_provider: str
    llm_model: str
    script_critic_score: float
    vision_qa_score: float
    render_time_seconds: float
    total_cost_usd: float
    script_metadata: Dict[str, Any] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class UploadMetrics:
    """Target platform publishing status and output media metadata."""
    platform: str
    upload_success: bool
    video_url: str
    video_id_on_platform: str
    duration_seconds: float = 0.0
    resolution_width: int = 1080
    resolution_height: int = 1920
    fps: int = 30
    file_size_bytes: int = 0
    output_path: str = ""


@dataclass
class AnalyticsMetrics:
    """Viewer engagement, watch time, and conversion outcomes from published analytics."""
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    retention_rate: float = 0.0
    ctr: float = 0.0
    impressions: int = 0
    avg_view_duration_seconds: float = 0.0
    avg_percentage_viewed: float = 0.0
    watch_time_hours: float = 0.0
    subscribers_gained: int = 0
    revenue_usd: float = 0.0
    last_updated: str = ""


@dataclass
class VideoPerformance:
    """Complete history of a generated video, linking design inputs with performance outputs."""
    video_id: str
    timestamp: str
    topic: Dict[str, Any]
    variation: Dict[str, Any]
    generation_metrics: GenerationMetrics
    upload_metrics: UploadMetrics
    analytics_metrics: AnalyticsMetrics = field(default_factory=AnalyticsMetrics)
    schema_version: int = 1
    pipeline_version: str = "1.0.0"
    git_commit: str = "unknown"


@dataclass
class LearningRecord:
    """Aggregate rules, adjustments, and producer notes derived from video performance outcomes."""
    video_performance: VideoPerformance
    notes: str = ""
    adjusted_rules: Dict[str, Any] = field(default_factory=dict)
