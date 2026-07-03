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


@dataclass
class HookPerformance:
    """Aggregated metrics for a specific hook type."""
    hook: str
    count: int
    avg_views: float
    avg_ctr: float
    avg_retention: float


@dataclass
class VoicePerformance:
    """Aggregated metrics for a specific voice."""
    voice: str
    count: int
    avg_views: float
    avg_watch_time_seconds: float


@dataclass
class SkinPerformance:
    """Aggregated metrics for a specific skin."""
    skin: str
    count: int
    avg_views: float
    avg_retention: float


@dataclass
class TopicPerformance:
    """Aggregated metrics for a specific topic number."""
    topic_num: str
    count: int
    avg_views: float
    avg_ctr: float


@dataclass
class OverallInsights:
    """Average metrics across all analyzed runs."""
    avg_ctr: float = 0.0
    avg_retention: float = 0.0
    avg_watch_time: float = 0.0
    avg_critic_score: float = 0.0
    avg_vision_qa_score: float = 0.0


@dataclass
class Recommendations:
    """Deterministic, rule-based recommendations for future content runs."""
    preferred_hook: str = ""
    preferred_voice: str = ""
    preferred_skin: str = ""
    preferred_format: str = ""
    preferred_topic: str = ""
    preferred_niche: str = ""
    confidence_score: float = 0.0
    notes: str = ""


@dataclass
class LearningSummary:
    """Complete summary of historical performance analysis and insights."""
    records_analyzed: int
    overall_insights: OverallInsights
    best_hook: str
    best_voice: str
    best_skin: str
    best_format: str
    best_niche: str
    fastest_render_config: Dict[str, Any]
    correlations: Dict[str, float]
    top_hooks: List[Dict[str, Any]] = field(default_factory=list)
    top_voices: List[Dict[str, Any]] = field(default_factory=list)
    top_skins: List[Dict[str, Any]] = field(default_factory=list)
    top_formats: List[Dict[str, Any]] = field(default_factory=list)
    top_topics: List[Dict[str, Any]] = field(default_factory=list)
    top_niches: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: Recommendations = field(default_factory=Recommendations)

