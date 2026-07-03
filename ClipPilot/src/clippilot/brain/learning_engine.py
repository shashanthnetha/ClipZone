# -*- coding: utf-8 -*-
"""ClipPilot Brain Learning Engine.

Analyzes historical VideoPerformance logs to compute statistics, correlations,
ranked performance lists, and deterministic content generation recommendations.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from clippilot.brain.analytics_models import (
    LearningSummary,
    OverallInsights,
    Recommendations,
    VideoPerformance,
)
from clippilot.brain.performance_store import PerformanceStore


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Computes the Pearson correlation coefficient between two lists of numbers."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    
    if var_x == 0.0 or var_y == 0.0:
        return 0.0
        
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    corr = cov / ((var_x * var_y) ** 0.5)
    return round(corr, 4)


def _aggregate_category(records: List[VideoPerformance], key_fn) -> Dict[str, Dict[str, Any]]:
    """Groups records by a key function and aggregates performance metrics."""
    groups: Dict[str, Dict[str, Any]] = {}
    for r in records:
        key = key_fn(r) or "unknown"
        if key not in groups:
            groups[key] = {
                "count": 0,
                "views": [],
                "likes": [],
                "comments": [],
                "ctr": [],
                "retention": [],
                "watch_time": [],
                "render_time": [],
            }
        g = groups[key]
        g["count"] += 1
        g["views"].append(r.analytics_metrics.views)
        g["likes"].append(r.analytics_metrics.likes)
        g["comments"].append(r.analytics_metrics.comments)
        g["ctr"].append(r.analytics_metrics.ctr)
        g["retention"].append(r.analytics_metrics.retention_rate)
        g["watch_time"].append(r.analytics_metrics.avg_view_duration_seconds)
        g["render_time"].append(r.generation_metrics.render_time_seconds)
        
    results = {}
    for key, g in groups.items():
        count = g["count"]
        results[key] = {
            "name": key,
            "count": count,
            "avg_views": round(sum(g["views"]) / count, 2),
            "avg_likes": round(sum(g["likes"]) / count, 2),
            "avg_comments": round(sum(g["comments"]) / count, 2),
            "avg_ctr": round(sum(g["ctr"]) / count, 4),
            "avg_retention": round(sum(g["retention"]) / count, 4),
            "avg_watch_time_seconds": round(sum(g["watch_time"]) / count, 2),
            "avg_render_time_seconds": round(sum(g["render_time"]) / count, 4),
        }
    return results


class LearningEngine:
    """Analyzes performance metrics to produce summaries and generation advice."""

    def __init__(self, store: PerformanceStore):
        self.store = store
        self.history: List[VideoPerformance] = []

    def load_history(self) -> List[VideoPerformance]:
        """Loads historical records from the store."""
        self.history = self.store.load_records()
        return self.history

    def compute_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Groups and aggregates metrics for key categories."""
        if not self.history:
            return {}

        return {
            "hook": _aggregate_category(self.history, lambda r: r.variation.get("hook")),
            "voice": _aggregate_category(self.history, lambda r: r.variation.get("voice")),
            "skin": _aggregate_category(self.history, lambda r: r.variation.get("skin")),
            "format": _aggregate_category(self.history, lambda r: r.variation.get("fmt")),
            "topic": _aggregate_category(self.history, lambda r: r.topic.get("num")),
            "niche": _aggregate_category(self.history, lambda r: r.topic.get("niche")),
        }

    def compute_correlations(self) -> Dict[str, float]:
        """Computes deterministic Pearson correlations for descriptive insights."""
        if len(self.history) < 2:
            return {
                "critic_score_vs_retention": 0.0,
                "vision_qa_vs_ctr": 0.0,
                "duration_vs_retention": 0.0,
                "hook_vs_ctr": 0.0,
                "voice_vs_watch_time": 0.0,
                "skin_vs_retention": 0.0,
            }

        # Numeric correlations
        critic_scores = [r.generation_metrics.script_critic_score for r in self.history]
        retention_rates = [r.analytics_metrics.retention_rate for r in self.history]
        vision_qa_scores = [r.generation_metrics.vision_qa_score for r in self.history]
        ctrs = [r.analytics_metrics.ctr for r in self.history]
        durations = [r.upload_metrics.duration_seconds for r in self.history]

        # Categorical maps
        unique_hooks = sorted(list(set(r.variation.get("hook", "unknown") for r in self.history)))
        hook_map = {h: idx for idx, h in enumerate(unique_hooks)}
        hook_indices = [hook_map.get(r.variation.get("hook", "unknown"), 0) for r in self.history]

        unique_voices = sorted(list(set(r.variation.get("voice", "unknown") for r in self.history)))
        voice_map = {v: idx for idx, v in enumerate(unique_voices)}
        voice_indices = [voice_map.get(r.variation.get("voice", "unknown"), 0) for r in self.history]

        unique_skins = sorted(list(set(r.variation.get("skin", "unknown") for r in self.history)))
        skin_map = {s: idx for idx, s in enumerate(unique_skins)}
        skin_indices = [skin_map.get(r.variation.get("skin", "unknown"), 0) for r in self.history]

        watch_times = [r.analytics_metrics.avg_view_duration_seconds for r in self.history]

        return {
            "critic_score_vs_retention": _pearson_correlation(critic_scores, retention_rates),
            "vision_qa_vs_ctr": _pearson_correlation(vision_qa_scores, ctrs),
            "duration_vs_retention": _pearson_correlation(durations, retention_rates),
            "hook_vs_ctr": _pearson_correlation(hook_indices, ctrs),
            "voice_vs_watch_time": _pearson_correlation(voice_indices, watch_times),
            "skin_vs_retention": _pearson_correlation(skin_indices, retention_rates),
        }

    def generate_summary(self) -> LearningSummary:
        """Processes historical metrics to compile ranked lists and recommendations."""
        records = self.history
        count = len(records)

        if count == 0:
            return LearningSummary(
                records_analyzed=0,
                overall_insights=OverallInsights(),
                best_hook="unknown",
                best_voice="unknown",
                best_skin="unknown",
                best_format="unknown",
                best_niche="unknown",
                fastest_render_config={},
                correlations=self.compute_correlations(),
                recommendations=Recommendations(),
            )

        # 1. Compute overall metrics
        total_ctr = sum(r.analytics_metrics.ctr for r in records)
        total_retention = sum(r.analytics_metrics.retention_rate for r in records)
        total_watch_time = sum(r.analytics_metrics.avg_view_duration_seconds for r in records)
        total_critic = sum(r.generation_metrics.script_critic_score for r in records)
        total_qa = sum(r.generation_metrics.vision_qa_score for r in records)

        insights = OverallInsights(
            avg_ctr=round(total_ctr / count, 4),
            avg_retention=round(total_retention / count, 4),
            avg_watch_time=round(total_watch_time / count, 2),
            avg_critic_score=round(total_critic / count, 2),
            avg_vision_qa_score=round(total_qa / count, 2),
        )

        # 2. Get category-level stats
        stats = self.compute_statistics()

        # Helper to sort and rank Top-5
        def get_ranked_list(category_stats: Dict[str, Dict[str, Any]], sort_key: str) -> List[Dict[str, Any]]:
            items = list(category_stats.values())
            # Sort descending by the key, secondarily descending by count to break ties deterministically
            items.sort(key=lambda x: (x.get(sort_key, 0), x.get("count", 0)), reverse=True)
            return items[:5]

        top_hooks = get_ranked_list(stats["hook"], "avg_ctr")
        top_voices = get_ranked_list(stats["voice"], "avg_watch_time_seconds")
        top_skins = get_ranked_list(stats["skin"], "avg_retention")
        top_formats = get_ranked_list(stats["format"], "avg_views")
        top_topics = get_ranked_list(stats["topic"], "avg_views")
        top_niches = get_ranked_list(stats["niche"], "avg_views")

        best_hook = top_hooks[0]["name"] if top_hooks else "unknown"
        best_voice = top_voices[0]["name"] if top_voices else "unknown"
        best_skin = top_skins[0]["name"] if top_skins else "unknown"
        best_format = top_formats[0]["name"] if top_formats else "unknown"
        best_niche = top_niches[0]["name"] if top_niches else "unknown"

        # 3. Find fastest render configuration
        fastest_record = min(records, key=lambda r: r.generation_metrics.render_time_seconds)
        fastest_config = {
            "video_id": fastest_record.video_id,
            "render_time_seconds": fastest_record.generation_metrics.render_time_seconds,
            "skin": fastest_record.variation.get("skin", "unknown"),
            "format": fastest_record.variation.get("fmt", "unknown"),
        }

        # 4. Generate recommendations
        confidence = min(1.0, round(0.1 + (count / 15.0) * 0.9, 2))
        
        # Topic num & Niche fallback
        best_topic = top_topics[0]["name"] if top_topics else "unknown"
        
        recs = Recommendations(
            preferred_hook=best_hook,
            preferred_voice=best_voice,
            preferred_skin=best_skin,
            preferred_format=best_format,
            preferred_topic=best_topic,
            preferred_niche=best_niche,
            confidence_score=confidence,
            notes=(
                f"Preferred hook '{best_hook}' selected by highest CTR. "
                f"Preferred voice '{best_voice}' selected by watch duration. "
                f"Preferred skin '{best_skin}' selected by user retention. "
                f"Generated deterministically from {count} performance logs."
            ),
        )

        return LearningSummary(
            records_analyzed=count,
            overall_insights=insights,
            best_hook=best_hook,
            best_voice=best_voice,
            best_skin=best_skin,
            best_format=best_format,
            best_niche=best_niche,
            fastest_render_config=fastest_config,
            correlations=self.compute_correlations(),
            top_hooks=top_hooks,
            top_voices=top_voices,
            top_skins=top_skins,
            top_formats=top_formats,
            top_topics=top_topics,
            top_niches=top_niches,
            recommendations=recs,
        )

    def save_summary(self) -> str:
        """Compiles the summary and saves it as JSON to ClipPilot/learning_summary.json."""
        summary = self.generate_summary()
        summary_dict = asdict(summary)
        
        # Save path relative to PerformanceStore path location
        parent_dir = self.store.store_path.parent
        output_path = parent_dir / "learning_summary.json"
        
        parent_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
            
        return str(output_path)
