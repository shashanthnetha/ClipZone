# -*- coding: utf-8 -*-
"""ClipPilot Brain Performance Store.

Manages persistent logging of pipeline performance records in JSONL format,
enabling video execution metadata and engagement analytics tracking.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from clippilot.brain.analytics_models import (
    AnalyticsMetrics,
    GenerationMetrics,
    UploadMetrics,
    VideoPerformance,
)


def _get_git_commit(workspace_dir: Path) -> str:
    """Safely fetch current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workspace_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class PerformanceStore:
    """Manages the reading and writing of VideoPerformance records in JSONL format."""

    def __init__(self, store_path: Path):
        self.store_path = store_path

    def save_record(self, record: VideoPerformance) -> None:
        """Appends a new performance record to the JSONL file."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize dataclass
        data = asdict(record)
        
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def load_records(self) -> List[VideoPerformance]:
        """Loads and returns all performance records from the store."""
        if not self.store_path.exists():
            return []

        records = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(self._deserialize_record(data))
                except Exception as e:
                    # Skip corrupt lines for backward/forward compatibility
                    print(f"⚠️ Warning: Failed to parse performance record line: {e}")
                    continue
        return records

    def find_record(self, video_id: str) -> Optional[VideoPerformance]:
        """Finds and returns a single record matching the given video_id."""
        for record in self.load_records():
            if record.video_id == video_id:
                return record
        return None

    def find_record_by_platform_id(self, platform_video_id: str) -> Optional[VideoPerformance]:
        """Finds and returns a single record matching the given platform-specific video identifier."""
        for record in self.load_records():
            if record.upload_metrics.video_id_on_platform == platform_video_id:
                return record
        return None

    def list_records(self) -> List[VideoPerformance]:
        """Alias for load_records to retrieve all logged executions."""
        return self.load_records()

    def update_record(self, record: VideoPerformance) -> None:
        """Rewrites the record file, replacing the record with matching video_id."""
        records = self.load_records()
        updated = False
        
        for idx, r in enumerate(records):
            if r.video_id == record.video_id:
                records[idx] = record
                updated = True
                break

        if not updated:
            records.append(record)

        # Rewrite the entire store file
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.store_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(asdict(r), default=str) + "\n")
        
        os.replace(temp_path, self.store_path)

    def update_analytics(self, video_id: str, analytics_metrics: AnalyticsMetrics) -> None:
        """Finds record by video_id, updates its analytics_metrics, and saves it."""
        record = self.find_record(video_id)
        if record:
            record.analytics_metrics = analytics_metrics
            self.update_record(record)
        else:
            raise ValueError(f"No performance record found with video_id: {video_id}")

    def update_analytics_by_platform_id(self, platform_video_id: str, analytics_metrics: AnalyticsMetrics) -> None:
        """Finds record by platform_video_id, updates its analytics_metrics, and saves it."""
        record = self.find_record_by_platform_id(platform_video_id)
        if record:
            record.analytics_metrics = analytics_metrics
            self.update_record(record)
        else:
            raise ValueError(f"No performance record found with platform video_id: {platform_video_id}")

    def _deserialize_record(self, data: Dict[str, Any]) -> VideoPerformance:
        """Reconstruct a strongly typed VideoPerformance instance from a dictionary."""
        # Extract Nested Dataclasses
        gen_data = data.get("generation_metrics", {})
        gen = GenerationMetrics(
            llm_provider=gen_data.get("llm_provider", ""),
            llm_model=gen_data.get("llm_model", ""),
            script_critic_score=float(gen_data.get("script_critic_score", 0.0)),
            vision_qa_score=float(gen_data.get("vision_qa_score", 0.0)),
            render_time_seconds=float(gen_data.get("render_time_seconds", 0.0)),
            total_cost_usd=float(gen_data.get("total_cost_usd", 0.0)),
            script_metadata=gen_data.get("script_metadata", {}),
            stage_timings=gen_data.get("stage_timings", {}),
        )

        up_data = data.get("upload_metrics", {})
        up = UploadMetrics(
            platform=up_data.get("platform", ""),
            upload_success=bool(up_data.get("upload_success", False)),
            video_url=up_data.get("video_url", ""),
            video_id_on_platform=up_data.get("video_id_on_platform", ""),
            duration_seconds=float(up_data.get("duration_seconds", 0.0)),
            resolution_width=int(up_data.get("resolution_width", 1080)),
            resolution_height=int(up_data.get("resolution_height", 1920)),
            fps=int(up_data.get("fps", 30)),
            file_size_bytes=int(up_data.get("file_size_bytes", 0)),
            output_path=up_data.get("output_path", ""),
        )

        an_data = data.get("analytics_metrics", {})
        an = AnalyticsMetrics(
            views=int(an_data.get("views", 0)),
            likes=int(an_data.get("likes", 0)),
            shares=int(an_data.get("shares", 0)),
            comments=int(an_data.get("comments", 0)),
            retention_rate=float(an_data.get("retention_rate", 0.0)),
            ctr=float(an_data.get("ctr", 0.0)),
            impressions=int(an_data.get("impressions", 0)),
            avg_view_duration_seconds=float(an_data.get("avg_view_duration_seconds", 0.0)),
            avg_percentage_viewed=float(an_data.get("avg_percentage_viewed", 0.0)),
            watch_time_hours=float(an_data.get("watch_time_hours", 0.0)),
            subscribers_gained=int(an_data.get("subscribers_gained", 0)),
            revenue_usd=float(an_data.get("revenue_usd", 0.0)),
            last_updated=str(an_data.get("last_updated", "")),
        )

        return VideoPerformance(
            video_id=data.get("video_id", ""),
            timestamp=data.get("timestamp", ""),
            topic=data.get("topic", {}),
            variation=data.get("variation", {}),
            generation_metrics=gen,
            upload_metrics=up,
            analytics_metrics=an,
            schema_version=int(data.get("schema_version", 1)),
            pipeline_version=data.get("pipeline_version", "1.0.0"),
            git_commit=data.get("git_commit", "unknown"),
            strategy_metadata=data.get("strategy_metadata", {}),
            video_asset_plan=data.get("video_asset_plan", {}),
        )
