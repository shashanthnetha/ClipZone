# -*- coding: utf-8 -*-
"""Asset Provider Integration for ClipPilot.

Provides automated visual assets fetching from Pexels, Pixabay, Unsplash, and Lucide SVG.
Uses deterministic scoring to prioritize vertical (9:16) format, high resolution, and duration compatibility.
"""
from __future__ import annotations

import os
import json
import re
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from clippilot.config import Settings
from clippilot.brain.asset_models import VideoAssetPlan, SceneAssetPlan
from clippilot.logger import get_logger

logger = get_logger("clippilot.assets")

def sanitize_query(query: str) -> str:
    """Sanitize queries to keep alphanumeric and spaces, replacing spaces with underscores."""
    clean = re.sub(r'[^a-zA-Z0-9\s-]', '', query.lower())
    return re.sub(r'[\s-]+', '_', clean.strip())

def score_video_asset(asset: Dict[str, Any], target_duration: float) -> float:
    """Deterministically score a video asset. Higher is better.
    
    Skips if resolution < 480px, duration is out of range, or watermark tags present.
    """
    width = int(asset.get("width") or 0)
    height = int(asset.get("height") or 0)
    if width < 480 or height < 480:
        return -1000.0

    tags = str(asset.get("tags", "")).lower()
    creator = str(asset.get("creator", "")).lower()
    if "watermark" in tags or "watermark" in creator:
        return -1000.0

    # 1. Aspect Ratio Score: Prefer vertical (9:16 ratio ~0.56)
    ratio = width / height
    is_vertical = 0.5 <= ratio <= 0.65
    ratio_score = 50.0 if is_vertical else 0.0

    # 2. Resolution Score: Pixels in millions
    resolution_score = (width * height) / 1_000_000.0

    # 3. Duration Score: Close to target scene duration (penalize if < 2s or > 60s)
    duration = float(asset.get("duration", 0.0))
    if duration > 0.0:
        if duration < 2.0 or duration > 60.0:
            return -1000.0
        diff = abs(duration - target_duration)
        duration_score = max(0.0, 10.0 - diff)
    else:
        duration_score = 0.0

    # 4. Bitrate Score (if available)
    bitrate = float(asset.get("bitrate", 0.0))
    bitrate_score = min(5.0, bitrate / 1_000_000.0) if bitrate > 0.0 else 0.0

    return ratio_score + resolution_score + duration_score + bitrate_score

def score_image_asset(asset: Dict[str, Any]) -> float:
    """Deterministically score an image asset. Higher is better.
    
    Skips if resolution < 480px or watermark tags present.
    """
    width = int(asset.get("width") or 0)
    height = int(asset.get("height") or 0)
    if width < 480 or height < 480:
        return -1000.0

    tags = str(asset.get("tags", "")).lower()
    creator = str(asset.get("creator", "")).lower()
    if "watermark" in tags or "watermark" in creator:
        return -1000.0

    # 1. Aspect Ratio Score: Prefer vertical (9:16 ratio ~0.56)
    ratio = width / height
    is_vertical = 0.5 <= ratio <= 0.65
    ratio_score = 50.0 if is_vertical else 0.0

    # 2. Resolution Score: Pixels in millions
    resolution_score = (width * height) / 1_000_000.0

    return ratio_score + resolution_score

# ── API Integration Providers ──

def query_pexels(query: str, api_key: str, target_duration: float) -> Optional[Dict[str, Any]]:
    """Search and score top 5-10 videos from Pexels API."""
    if not api_key:
        return None
    headers = {"Authorization": api_key}
    url = f"https://api.pexels.com/videos/search?query={requests.utils.quote(query)}&per_page=10"
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            logger.warning(f"Pexels query failed with code {res.status_code}")
            return None
        data = res.json()
        candidates = []
        for video in data.get("videos", []):
            files = video.get("video_files", [])
            if not files:
                continue
            best_file = max(files, key=lambda f: (int(f.get("width") or 0) * int(f.get("height") or 0)))
            candidates.append({
                "width": video.get("width"),
                "height": video.get("height"),
                "duration": video.get("duration"),
                "creator": video.get("user", {}).get("name", "Unknown"),
                "tags": "",
                "download_url": best_file.get("link"),
                "source_url": video.get("url"),
                "license": "Pexels License",
                "provider": "pexels"
            })
        
        # Deterministically select highest scoring candidate
        scored = []
        for c in candidates:
            score = score_video_asset(c, target_duration)
            if score >= 0:
                scored.append((score, c))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]
    except Exception as e:
        logger.warning(f"Error querying Pexels: {e}")
    return None

def query_pixabay(query: str, api_key: str, target_duration: float) -> Optional[Dict[str, Any]]:
    """Search and score top 5-10 videos from Pixabay API."""
    if not api_key:
        return None
    url = f"https://pixabay.com/api/videos/?key={api_key}&q={requests.utils.quote(query)}&per_page=10"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code != 200:
            logger.warning(f"Pixabay query failed with code {res.status_code}")
            return None
        data = res.json()
        candidates = []
        for hit in data.get("hits", []):
            v_dict = hit.get("videos", {})
            best_file = None
            for size in ["large", "medium", "small", "tiny"]:
                if size in v_dict and v_dict[size].get("url"):
                    best_file = v_dict[size]
                    break
            if not best_file:
                continue
            candidates.append({
                "width": best_file.get("width"),
                "height": best_file.get("height"),
                "duration": hit.get("duration"),
                "creator": hit.get("user", "Unknown"),
                "tags": hit.get("tags", ""),
                "download_url": best_file.get("url"),
                "source_url": hit.get("pageURL"),
                "license": "Pixabay License",
                "provider": "pixabay"
            })
        
        scored = []
        for c in candidates:
            score = score_video_asset(c, target_duration)
            if score >= 0:
                scored.append((score, c))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]
    except Exception as e:
        logger.warning(f"Error querying Pixabay: {e}")
    return None

def query_unsplash(query: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Search and score top 5-10 images from Unsplash API."""
    if not api_key:
        return None
    headers = {"Authorization": f"Client-ID {api_key}"}
    url = f"https://api.unsplash.com/search/photos?query={requests.utils.quote(query)}&per_page=10"
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            logger.warning(f"Unsplash query failed with code {res.status_code}")
            return None
        data = res.json()
        candidates = []
        for item in data.get("results", []):
            urls = item.get("urls", {})
            download_url = urls.get("regular") or urls.get("full")
            if not download_url:
                continue
            candidates.append({
                "width": item.get("width"),
                "height": item.get("height"),
                "creator": item.get("user", {}).get("name", "Unknown"),
                "tags": "",
                "download_url": download_url,
                "source_url": item.get("links", {}).get("html", ""),
                "license": "Unsplash License",
                "provider": "unsplash"
            })
        
        scored = []
        for c in candidates:
            score = score_image_asset(c)
            if score >= 0:
                scored.append((score, c))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]
    except Exception as e:
        logger.warning(f"Error querying Unsplash: {e}")
    return None

def download_lucide_icon(icon_name: str) -> Optional[bytes]:
    """Fetch direct SVG file bytes from Lucide Icons GitHub repository CDN."""
    url = f"https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/{icon_name}.svg"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.content
    except Exception as e:
        logger.warning(f"Failed to fetch Lucide icon {icon_name}: {e}")
    return None

# ── Pipeline Asset Downloads Manager ──

def download_assets_for_plan(
    asset_plan: VideoAssetPlan,
    target_dir: Path,
    skin_id: str,
    settings: Optional[Settings] = None
) -> None:
    """Processes a VideoAssetPlan and downloads stock visuals/icons, writing to target_dir.
    
    Caches media files in `data/cache/assets/<provider>/<sanitized_query>/`.
    """
    if settings is None:
        settings = Settings.load()

    target_dir.mkdir(parents=True, exist_ok=True)
    cache_base = Path("data/cache/assets")
    cache_base.mkdir(parents=True, exist_ok=True)

    providers = settings.asset_provider_priority

    for scene in asset_plan.scenes:
        scene_num = scene.scene_number
        target_duration = getattr(scene, "duration_seconds", 5.0)

        # 1. Download Icons if requested
        for icon in scene.icon_queries:
            clean_icon = icon.lower().strip()
            san_query = sanitize_query(clean_icon)
            cache_dir = cache_base / "lucide" / san_query
            cache_dir.mkdir(parents=True, exist_ok=True)

            media_file = cache_dir / "media.svg"
            meta_file = cache_dir / "metadata.json"

            # Cache hit check
            if media_file.exists():
                shutil.copy(media_file, target_dir / f"asset_{skin_id.lower()}_{scene_num}.svg")
                logger.info(f"✔️ Icon cached hit for scene {scene_num}: '{icon}'")
                continue

            # Cache miss - download
            svg_bytes = download_lucide_icon(clean_icon)
            if svg_bytes:
                media_file.write_bytes(svg_bytes)
                meta_file.write_text(json.dumps({
                    "query": icon,
                    "provider": "lucide",
                    "source_url": f"https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/{clean_icon}.svg",
                    "creator": "Lucide Contributors",
                    "license": "ISC License",
                    "download_timestamp": datetime.now(timezone.utc).isoformat()
                }, indent=2), encoding="utf-8")
                
                shutil.copy(media_file, target_dir / f"asset_{skin_id.lower()}_{scene_num}.svg")
                logger.info(f"✔️ Downloaded Lucide icon for scene {scene_num}: '{icon}'")

        # 2. Download Stock footage / Images
        # Visual media queries priority: video queries -> image queries
        queries = []
        is_video_mode = True
        if scene.stock_video_queries:
            queries = scene.stock_video_queries
        elif scene.image_queries:
            queries = scene.image_queries
            is_video_mode = False

        if not queries:
            continue

        query = queries[0]
        san_query = sanitize_query(query)
        downloaded = False

        for provider in providers:
            if downloaded:
                break

            cache_dir = cache_base / provider / san_query
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Try checking the cache directory first
            # Look for media.mp4, media.png or media.jpg
            media_path = None
            for ext in [".mp4", ".png", ".jpg"]:
                p = cache_dir / f"media{ext}"
                if p.exists():
                    media_path = p
                    break

            if media_path:
                ext = media_path.suffix
                dest = target_dir / f"asset_{skin_id.lower()}_{scene_num}{ext}"
                shutil.copy(media_path, dest)
                logger.info(f"✔️ Cache hit for scene {scene_num} from {provider} -> {dest.name}")
                downloaded = True
                break

            # Cache miss - search API
            asset = None
            if provider == "pexels" and settings.pexels_api_key and is_video_mode:
                asset = query_pexels(query, settings.pexels_api_key, target_duration)
            elif provider == "pixabay" and settings.pixabay_api_key and is_video_mode:
                asset = query_pixabay(query, settings.pixabay_api_key, target_duration)
            elif provider == "unsplash" and settings.unsplash_api_key and not is_video_mode:
                asset = query_unsplash(query, settings.unsplash_api_key)

            if asset:
                download_url = asset.get("download_url")
                if not download_url:
                    continue
                try:
                    res = requests.get(download_url, timeout=30)
                    if res.status_code == 200:
                        # Determine file extension based on provider and video mode
                        ext = ".mp4" if is_video_mode else ".png"
                        if provider == "unsplash":
                            ext = ".png"
                        
                        media_file = cache_dir / f"media{ext}"
                        media_file.write_bytes(res.content)

                        meta_file = cache_dir / "metadata.json"
                        meta_file.write_text(json.dumps({
                            "query": query,
                            "provider": provider,
                            "source_url": asset.get("source_url"),
                            "creator": asset.get("creator"),
                            "license": asset.get("license"),
                            "download_timestamp": datetime.now(timezone.utc).isoformat()
                        }, indent=2), encoding="utf-8")

                        dest = target_dir / f"asset_{skin_id.lower()}_{scene_num}{ext}"
                        shutil.copy(media_file, dest)
                        logger.info(f"✔️ Successfully downloaded and cached asset for scene {scene_num} from {provider} -> {dest.name}")
                        downloaded = True
                except Exception as e:
                    logger.warning(f"Failed to download asset from {provider}: {e}")

        if not downloaded:
            logger.info(f"ℹ️ No keys or no matches for '{query}' on {providers}. Falling back to default mock asset.")
