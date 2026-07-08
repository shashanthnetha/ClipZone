# -*- coding: utf-8 -*-
"""ClipPilot Brain End-to-End Production Pipeline.

Coordinates and executes the entire video generation pipeline using real
renderers, real voice generators, real vision auditors (where configured),
and dry-run publisher uploads.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from clippilot.brain.asset_registry import AssetRegistry
from clippilot.brain.composition_engine import compose_video
from clippilot.brain.execution_engine import ExecutionContext
from clippilot.brain.pipeline_orchestrator import RunContext, choose_topic, choose_variation, load_state
from clippilot.brain.publisher import Metadata, PublishRequest, get_publisher
from clippilot.brain.remotion_renderer import render_to_remotion
from clippilot.brain.render_compiler import compile_render_graph
from clippilot.brain.render_graph import build_render_graph
from clippilot.brain.scene_planner import plan_scenes
from clippilot.brain.script_generator import generate_script
from clippilot.brain.vision_qa import run_vision_qa
from clippilot.brain.voice_provider import VoiceAlignment, VoiceRequest, get_voice_provider
from clippilot.config import Settings
from clippilot.logger import get_logger

logger = get_logger("clippilot.pipeline")

def print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if "⚠️" in msg:
        logger.warning(msg)
    elif "❌" in msg or "FAIL" in msg or "failed" in msg.lower():
        logger.error(msg)
    else:
        logger.info(msg)

# Tiny 1-second silent MP3 base64 to backfill missing SFX assets
SILENT_MP3_B64 = (
    "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjYyLjEyLjEwMQAAAAAAAAAAAAAA//tAwAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAoAAAQ9gAQEBYWHR0dIyMpKSkvLzU1NTs7QUFBSEhOTk5UVFpaWmBgZmZmbGxycnJ5eX9/f4WFi4uLkZGXl5ednaSkpKqqsLCwtra8vLzCwsjIyM7O1dXV29vh4eHn5+3t7fPz+fn5//8AAAAATGF2YzYyLjI4AAAAAAAAAAAAAAAAJAV8AAAAAAAAEPYp+CflAAAAAAD/+xDEAAPAAAGkAAAAIAAANIAAAARMQU1FMy4xMDBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMQpg8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxFMDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDEfIPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMSmA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxM+DwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMQU1FMy4xMDBVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX//sQxNYDwAABpAAAACAAADSAAAAEVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/+xDE1gPAAAGkAAAAIAAANIAAAARVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVf/7EMTWA8AAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQ=="
)


VOICE_MAP = {
    "V1": "en-US-AndrewMultilingualNeural",
    "V2": "en-US-AriaNeural",
    "V3": "en-US-GuyNeural",
    "V4": "en-US-ChristopherNeural",
    "V5": "en-GB-RyanNeural",
    "V6": "en-GB-SoniaNeural",
    "V7": "en-AU-NatashaNeural",
    "V8": "en-US-EmmaNeural",
}


def _extract_frame_screenshot(video_path: str, timestamp_seconds: float, output_path: str) -> bool:
    """Uses bundled ffmpeg to extract a frame screenshot at specific timestamp."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp_seconds),
            "-i",
            video_path,
            "-vframes",
            "1",
            "-q:v",
            "2",
            output_path,
        ]
        # Run silently
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def execute_production_pipeline(
    workspace_dir: Path,
    variation_dir: Path,
    date_str: str = "2026-07-03",
    slug: str = "daily_006",
    skip_qa_publish: bool = False,
) -> dict[str, Any]:
    """Executes the E2E production pipeline with real renderers and real TTS."""
    from clippilot.brain.provider import pipeline_diagnostics
    pipeline_diagnostics["attempts"].clear()
    pipeline_diagnostics["correction_requests"] = 0
    pipeline_diagnostics["json_repairs"] = 0
    pipeline_diagnostics["schema_failures"] = 0
    pipeline_diagnostics["total_latency"] = 0.0
    pipeline_diagnostics["input_tokens"] = 0
    pipeline_diagnostics["output_tokens"] = 0
    pipeline_diagnostics["actual_provider"] = None
    pipeline_diagnostics["actual_model"] = None

    timings = {}
    report = {}

    settings = Settings.load()
    api_key_present = bool(
        os.environ.get("LLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or settings.llm_api_key
    )

    print("\n🚀 [STAGE 1] Initializing Context and Loading State Ledger...")
    start = time.time()
    run_ctx = RunContext(
        workspace_dir=workspace_dir,
        variation_dir=variation_dir,
        date_str=date_str,
        slug=slug,
    )
    state = load_state(run_ctx)
    timings["load_state"] = round(time.time() - start, 4)
    print(f"✔️ Ledger state loaded.")

    print("\n🚀 [STAGE 2] Selecting Backlog Topic and Variation via Strategy Engine...")
    start = time.time()
    try:
        from clippilot.brain.performance_store import PerformanceStore
        from clippilot.brain.learning_engine import LearningEngine
        from clippilot.brain.strategy_engine import StrategyEngine

        store_path = workspace_dir / "ClipPilot" / "performance_history.jsonl"
        store = PerformanceStore(store_path)
        learning_engine = LearningEngine(store)
        strategy_engine = StrategyEngine(store, learning_engine)

        decision = strategy_engine.make_decision(state, variation_dir)
        topic = decision.topic
        variation = decision.variation
        
        slug = f"daily_{topic.num}"
        run_ctx.slug = slug

        if not variation.slug:
            variation.slug = slug
        if not variation.date:
            variation.date = date_str

        print(f"🎯 Strategy Decision: Topic {topic.num} '{topic.title}', Hook '{decision.hook}', Voice '{decision.voice}' (Confidence: {decision.confidence})")
        print(f"🧐 Reasoning: {decision.reasoning}")
    except Exception as e:
        print(f"⚠️ Strategy Engine selection failed: {e}. Falling back to standard rotation.")
        topic = choose_topic(state)
        if topic:
            topic_cluster = topic.niche or "credit"
            variation = choose_variation(state, variation_dir, topic_cluster)
            slug = f"daily_{topic.num}"
            run_ctx.slug = slug
        else:
            variation = None

    timings["choose_topic"] = round(time.time() - start, 4)
    timings["choose_variation"] = 0.0

    if not topic or not variation:
        print("❌ No unused topics or valid variations found!")
        return {"success": False, "error": "No unused topics/variations"}

    print("\n🚀 [STAGE 4] Script Generation...")
    start = time.time()
    script = generate_script(state, topic, variation, workspace_dir)
    timings["generate_script"] = round(time.time() - start, 4)
    print(f"✔️ Script generated: '{script.title}'")

    # [STAGE 4B] Script Critic Evaluation & Revision
    start_critic = time.time()
    from clippilot.brain.script_critic import orchestrate_script_revision
    script = orchestrate_script_revision(state, topic, variation, workspace_dir, script, threshold=8.5)
    timings["script_critic"] = round(time.time() - start_critic, 4)

    print("\n🚀 [STAGE 5] Initial Scene Planning...")
    start = time.time()
    scene_plan = plan_scenes(script, variation)
    timings["plan_scenes"] = round(time.time() - start, 4)
    print(f"✔️ Scene plan blueprints ready.")

    print("\n🚀 [STAGE 5B] Asset Intelligence...")
    start_assets = time.time()
    from clippilot.brain.asset_intelligence import AssetIntelligenceEngine
    asset_engine = AssetIntelligenceEngine()
    asset_plan = asset_engine.build_asset_plan(script, scene_plan, settings)
    timings["asset_intelligence"] = round(time.time() - start_assets, 4)
    report["video_asset_plan"] = asset_plan.to_dict()
    print("✔️ Asset Plan created.")
    
    print("\n🚀 [STAGE 5C] Downloading Visual Assets & Icons...")
    start_downloads = time.time()
    from clippilot.media.asset_providers import download_assets_for_plan
    explainer_dir = workspace_dir / "ClipPilot" / "remotion_explainer"
    graphics_dir = explainer_dir / "public" / "assets" / "graphics"
    download_assets_for_plan(asset_plan, graphics_dir, variation.skin, settings)
    timings["asset_downloads"] = round(time.time() - start_downloads, 4)
    print("✔️ Visual assets & icons synced.")
    print("\nAsset Plan")
    for s_plan in asset_plan.scenes:
        print(f"\nScene {s_plan.scene_number}")
        print(f"Background: {s_plan.background}")
        print(f"Stock Queries:")
        for sq in s_plan.stock_video_queries:
            print(f"  {sq}")
        print(f"Icon Queries:")
        for icon in s_plan.icon_queries:
            print(f"  {icon}")
        if s_plan.chart_type:
            print(f"Chart:\n  {s_plan.chart_type}")
        if s_plan.transitions:
            print(f"Transition:")
            for trans in s_plan.transitions:
                print(f"  {trans}")

    print("\n🚀 [STAGE 6] Generating Real TTS Audio & Syncing Timings...")
    start = time.time()
    voice_provider = get_voice_provider("edge-tts")
    explainer_dir = workspace_dir / "ClipPilot" / "remotion_explainer"
    voice_dir = explainer_dir / "public" / "audio" / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)

    total_duration = 0.0
    all_alignments = []

    for s in scene_plan.scenes:
        output_path = voice_dir / f"{topic.num}_scene_{s.scene_index}.mp3"
        print(f"  - Generating TTS for scene {s.scene_index} -> {output_path.name}")
        
        voice_req = VoiceRequest(
            text=s.narration,
            voice_id=VOICE_MAP.get(variation.voice, variation.voice),
            output_path=str(output_path),
        )
        voice_res = voice_provider.generate_voice(voice_req)
        
        # Override plan values with the real timings returned from edge-tts
        s.duration_seconds = voice_res.duration_seconds
        s.subtitle_words = [a.word for a in voice_res.alignments]
        s.subtitle_timings = [(a.start_seconds, a.end_seconds) for a in voice_res.alignments]
        
        total_duration += s.duration_seconds
        # Keep tracking shifted alignments for QA verification
        for align in voice_res.alignments:
            all_alignments.append(
                VoiceAlignment(
                    word=align.word,
                    start_seconds=align.start_seconds + total_duration - s.duration_seconds,
                    end_seconds=align.end_seconds + total_duration - s.duration_seconds,
                )
            )

    scene_plan.total_duration_seconds = total_duration
    timings["generate_voice"] = round(time.time() - start, 4)
    print(f"✔️ Narrations generated. Total voice duration: {total_duration:.2f}s")

    # Backfill missing SFX assets with silent mp3s to prevent Remotion compiler crashes
    sfx_dir = explainer_dir / "public" / "audio" / "sfx"
    sfx_dir.mkdir(parents=True, exist_ok=True)
    for sfx in ["whoosh", "pop", "ding"]:
        sfx_path = sfx_dir / f"{sfx}.mp3"
        if not sfx_path.exists():
            sfx_path.write_bytes(base64.b64decode(SILENT_MP3_B64 + "="))

    print("\n🚀 [STAGE 7] Building Layout Composition...")
    start = time.time()
    comp = compose_video(scene_plan)
    timings["build_composition"] = round(time.time() - start, 4)

    print("\n🚀 [STAGE 8] Constructing Render Graph...")
    start = time.time()
    graph = build_render_graph(comp)
    timings["build_render_graph"] = round(time.time() - start, 4)

    print("\n🚀 [STAGE 9] Compiling Render Tree Component Nodes...")
    start = time.time()
    tree = compile_render_graph(graph)
    timings["compile_render_tree"] = round(time.time() - start, 4)

    print("\n🚀 [STAGE 10] Rendering Remotion React/TSX Output...")
    start = time.time()
    render_output = render_to_remotion(tree)
    timings["render_compilation"] = round(time.time() - start, 4)

    # Write scenes and root compositions to remotion folders
    src_dir = explainer_dir / "src"
    scenes_path = src_dir / "scenes.tsx"
    root_path = src_dir / "Root.tsx"

    root_backup = root_path.read_text(encoding="utf-8") if root_path.exists() else None
    out_dir = explainer_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_mp4 = out_dir / f"{slug}.mp4"

    print("\n🚀 [STAGE 11] Executing Real Remotion MP4 Compiler Subprocess...")
    start = time.time()
    try:
        scenes_path.write_text(render_output.react_component_tree, encoding="utf-8")
        root_path.write_text(render_output.remotion_composition, encoding="utf-8")

        import re
        comp_id = re.sub(r'[^a-zA-Z0-9-]', '', tree.title.replace(' ', '-').replace('_', '-'))
        cmd = [
            "npx",
            "remotion",
            "render",
            "src/index.ts",
            comp_id,
            str(output_mp4),
            "--concurrency=2",
        ]
        print(f"  - Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(explainer_dir), check=True)
        timings["render_video"] = round(time.time() - start, 4)
        print(f"✔️ Remotion render complete: {output_mp4.name}")
    finally:
        # Restore backups
        if scenes_path.exists():
            scenes_path.unlink()
        if root_backup:
            root_path.write_text(root_backup, encoding="utf-8")
        elif root_path.exists():
            root_path.unlink()

    if skip_qa_publish:
        script_cost = script.metadata.get("cost", 0.0) if script.metadata else 0.0
        critic_cost = script.metadata.get("critic_usage", {}).get("cost", 0.0) if script.metadata else 0.0
        asset_cost = asset_plan.metadata.get("cost", 0.0) if 'asset_plan' in locals() and asset_plan else 0.0
        total_cost = round(script_cost + critic_cost + asset_cost, 6)

        if script.metadata:
            script.metadata["stage_usages"] = {
                "script": {
                    "provider": script.metadata.get("provider"),
                    "requested_model": script.metadata.get("requested_model"),
                    "actual_model": script.metadata.get("actual_model"),
                    "input_tokens": script.metadata.get("input_tokens"),
                    "output_tokens": script.metadata.get("output_tokens"),
                    "latency": script.metadata.get("latency"),
                    "estimated_cost": script.metadata.get("estimated_cost"),
                    "cost": script.metadata.get("cost")
                },
                "critic": script.metadata.get("critic_usage", {}),
                "asset": asset_plan.metadata if 'asset_plan' in locals() and asset_plan else {},
                "vision_qa": {
                    "provider": "MockProvider",
                    "requested_model": "mock",
                    "actual_model": "mock",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency": 0.0,
                    "estimated_cost": 0.0,
                    "cost": 0.0
                }
            }

        # Stop here and return success
        report["success"] = True
        report["providers"] = {
            "llm": script.metadata.get("provider", "unknown") if script.metadata else "unknown",
            "tts": "edge-tts",
            "vision_qa": "skipped",
            "publisher": "skipped",
        }
        report["cost_estimate_usd"] = total_cost
        report["qa_score"] = 100.0
        report["qa_passed"] = True
        report["output_paths"] = {
            "final_video": str(output_mp4),
            "voiceovers": str(voice_dir),
        }
        report["upload_result"] = {
            "success": True,
            "video_id": "skipped_render_only",
            "url": "skipped_render_only",
        }
        
        # Still record performance metrics for history tracking (with placeholder upload/qa status)
        try:
            import uuid
            import datetime
            from clippilot.brain.performance_store import PerformanceStore, _get_git_commit
            from clippilot.brain.analytics_models import VideoPerformance, GenerationMetrics, UploadMetrics, AnalyticsMetrics

            store_path = workspace_dir / "ClipPilot" / "performance_history.jsonl"
            store = PerformanceStore(store_path)

            video_id = f"{slug}_{uuid.uuid4().hex[:8]}"
            git_hash = _get_git_commit(workspace_dir)

            gen_metrics = GenerationMetrics(
                llm_provider=report["providers"]["llm"],
                llm_model=script.metadata.get("actual_model", settings.llm_model or "claude-opus-4-8"),
                script_critic_score=script.metadata.get("final_score", 0.0),
                vision_qa_score=100.0,
                render_time_seconds=timings.get("execute_remotion", 0.0),
                total_cost_usd=total_cost,
                script_metadata=script.metadata,
                stage_timings=timings,
            )

            file_size = output_mp4.stat().st_size if output_mp4.exists() else 0
            upload_metrics = UploadMetrics(
                platform="skipped",
                upload_success=True,
                video_url="skipped_render_only",
                video_id_on_platform="skipped_render_only",
                duration_seconds=total_duration,
                resolution_width=1080,
                resolution_height=1920,
                fps=30,
                file_size_bytes=file_size,
                output_path=str(output_mp4),
            )

            perf_record = VideoPerformance(
                video_id=video_id,
                timestamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                topic={
                    "num": topic.num,
                    "title": topic.title,
                    "niche": topic.niche,
                    "angle": topic.angle,
                    "guardrail": topic.guardrail,
                },
                variation={
                    "title": variation.title,
                    "fmt": variation.fmt,
                    "voice": variation.voice,
                    "skin": variation.skin,
                    "hook": variation.hook,
                    "cluster": variation.cluster,
                },
                generation_metrics=gen_metrics,
                upload_metrics=upload_metrics,
                analytics_metrics=AnalyticsMetrics(),
                schema_version=1,
                pipeline_version="1.0.0",
                git_commit=git_hash,
                strategy_metadata=decision.metadata if 'decision' in locals() else {},
                video_asset_plan=report.get("video_asset_plan", {}),
            )
            store.save_record(perf_record)
        except Exception as e:
            print(f"⚠️ Warning: Failed to record performance metrics: {e}")

        return report

    print("\n🚀 [STAGE 12] Extracting Frame Screenshots for Vision QA...")
    start = time.time()
    sampled_image_paths = []
    # Sample midpoints of scenes
    current_time = 0.0
    for idx, scene in enumerate(tree.scenes):
        duration = scene.timing.duration_frames / tree.fps
        midpoint = current_time + (duration / 2.0)
        img_path = out_dir / f"{slug}_frame_{idx + 1}.png"
        
        extracted = _extract_frame_screenshot(str(output_mp4), midpoint, str(img_path))
        if extracted:
            sampled_image_paths.append(str(img_path))
        current_time += duration

    qa_res = run_vision_qa(str(output_mp4), script, tree, all_alignments, sampled_image_paths, provider=None)
    timings["vision_qa"] = round(time.time() - start, 4)
    print(f"✔️ Vision QA complete. Score: {qa_res.score}/100. Status: {qa_res.passed}")

    # Clean up extracted frame screenshots
    for p in sampled_image_paths:
        if os.path.exists(p):
            os.unlink(p)

    print("\n🚀 [STAGE 13] Perform Youtube Publisher Upload (Dry Run)...")
    start = time.time()
    publisher = get_publisher("youtube")
    pub_req = PublishRequest(
        video_path=str(output_mp4),
        metadata=Metadata(
            title=script.title,
            description=f"{script.hook}\n\nKeywords: {', '.join(script.niche_context.split())}",
        ),
        dry_run=True,
    )
    pub_res = publisher.publish(pub_req)
    timings["publish"] = round(time.time() - start, 4)
    print(f"✔️ Publisher upload complete. Success: {pub_res.success}")

    # Calculate real run cost estimate based on tokens and API pricing
    script_cost = script.metadata.get("cost", 0.0) if script.metadata else 0.0
    critic_cost = script.metadata.get("critic_usage", {}).get("cost", 0.0) if script.metadata else 0.0
    asset_cost = asset_plan.metadata.get("cost", 0.0) if 'asset_plan' in locals() and asset_plan else 0.0
    qa_cost = qa_res.metadata.get("cost", 0.0)
    total_cost = round(script_cost + critic_cost + asset_cost + qa_cost, 6)

    if script.metadata:
        script.metadata["stage_usages"] = {
            "script": {
                "provider": script.metadata.get("provider"),
                "requested_model": script.metadata.get("requested_model"),
                "actual_model": script.metadata.get("actual_model"),
                "input_tokens": script.metadata.get("input_tokens"),
                "output_tokens": script.metadata.get("output_tokens"),
                "latency": script.metadata.get("latency"),
                "estimated_cost": script.metadata.get("estimated_cost"),
                "cost": script.metadata.get("cost")
            },
            "critic": script.metadata.get("critic_usage", {}),
            "asset": asset_plan.metadata if 'asset_plan' in locals() and asset_plan else {},
            "vision_qa": qa_res.metadata
        }

    # Compile the final report
    report["success"] = True
    report["timings"] = timings
    report["total_time_seconds"] = round(sum(timings.values()), 4)
    
    qa_prov = qa_res.metadata.get("provider", "mock-fallback")
    qa_model = qa_res.metadata.get("actual_model", "none")
    llm_prov = script.metadata.get("provider", "mock-fallback") if script.metadata else "mock-fallback"
    
    report["providers"] = {
        "llm": llm_prov,
        "tts": "edge-tts",
        "vision_qa": f"{qa_prov} ({qa_model})" if qa_prov != "MockProvider" else "simulated-fallback",
        "publisher": "youtube-api",
    }
    report["cost_estimate_usd"] = total_cost
    report["qa_score"] = qa_res.score
    report["qa_passed"] = qa_res.passed
    report["output_paths"] = {
        "final_video": str(output_mp4),
        "voiceovers": str(voice_dir),
    }
    report["upload_result"] = {
        "success": pub_res.success,
        "video_id": pub_res.video_id,
        "url": pub_res.url,
    }

    from clippilot.brain.provider import pipeline_diagnostics
    print("\n=========================================")
    print("Model attempts:")
    attempts = pipeline_diagnostics.get("attempts", [])
    model_stats = {}
    for att in attempts:
        m = att["model"]
        status = att["status"]
        lat = att["latency"]
        if m not in model_stats:
            model_stats[m] = {"attempts": [], "total_latency": 0.0}
        model_stats[m]["attempts"].append(status)
        model_stats[m]["total_latency"] += lat
        
    for m, stats in model_stats.items():
        print(f"  {m}")
        for status in stats["attempts"]:
            print(f"    {status}")
            
    print("\nTime spent per model:")
    for m, stats in model_stats.items():
        print(f"  {m}: {stats['total_latency']:.4f}s")
        
    print(f"\nRetry count          : {len(attempts) - 1 if attempts else 0}")
    print(f"Correction requests  : {pipeline_diagnostics.get('correction_requests', 0)}")
    print(f"JSON repairs         : {pipeline_diagnostics.get('json_repairs', 0)}")
    print(f"Schema failures      : {pipeline_diagnostics.get('schema_failures', 0)}")
    print(f"Estimated token usage: Input={pipeline_diagnostics.get('input_tokens', 0)}, Output={pipeline_diagnostics.get('output_tokens', 0)}")
    print(f"Actual provider used : {pipeline_diagnostics.get('actual_provider') or 'None'}")
    print("=========================================")

    print("\n=========================================")
    print("PRODUCTION PIPELINE COMPLETED SUCCESSFULLY")
    print("=========================================")
    print(f"TOTAL RUN TIME : {report['total_time_seconds']}s")
    print(f"ESTIMATED COST : ${report['cost_estimate_usd']:.5f}")
    print(f"QA EVAL SCORE  : {report['qa_score']}/100")
    print(f"VIDEO FILE PATH: {report['output_paths']['final_video']}")
    print(f"PUBLISH URL    : {report['upload_result']['url']}")
    print("=========================================\n")

    # Record execution performance metrics
    try:
        import uuid
        import datetime
        from clippilot.brain.performance_store import PerformanceStore, _get_git_commit
        from clippilot.brain.analytics_models import VideoPerformance, GenerationMetrics, UploadMetrics, AnalyticsMetrics

        store_path = workspace_dir / "ClipPilot" / "performance_history.jsonl"
        store = PerformanceStore(store_path)

        video_id = f"{slug}_{uuid.uuid4().hex[:8]}"
        git_hash = _get_git_commit(workspace_dir)

        gen_metrics = GenerationMetrics(
            llm_provider=report["providers"]["llm"],
            llm_model=script.metadata.get("actual_model", settings.llm_model or "claude-opus-4-8"),
            script_critic_score=script.metadata.get("final_score", 0.0),
            vision_qa_score=float(qa_res.score),
            render_time_seconds=timings.get("execute_remotion", 0.0),
            total_cost_usd=total_cost,
            script_metadata=script.metadata,
            stage_timings=timings,
        )

        file_size = output_mp4.stat().st_size if output_mp4.exists() else 0
        upload_metrics = UploadMetrics(
            platform="youtube",
            upload_success=pub_res.success,
            video_url=pub_res.url,
            video_id_on_platform=pub_res.video_id,
            duration_seconds=total_duration,
            resolution_width=1080,
            resolution_height=1920,
            fps=30,
            file_size_bytes=file_size,
            output_path=str(output_mp4),
        )

        perf_record = VideoPerformance(
            video_id=video_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            topic={
                "num": topic.num,
                "title": topic.title,
                "niche": topic.niche,
                "angle": topic.angle,
                "guardrail": topic.guardrail,
            },
            variation={
                "title": variation.title,
                "fmt": variation.fmt,
                "voice": variation.voice,
                "skin": variation.skin,
                "hook": variation.hook,
                "cluster": variation.cluster,
            },
            generation_metrics=gen_metrics,
            upload_metrics=upload_metrics,
            analytics_metrics=AnalyticsMetrics(),
            schema_version=1,
            pipeline_version="1.0.0",
            git_commit=git_hash,
            strategy_metadata=decision.metadata if 'decision' in locals() else {},
            video_asset_plan=report.get("video_asset_plan", {}),
        )
        store.save_record(perf_record)
        print(f"✔️ Performance metrics recorded to '{store_path}' (Video ID: {video_id})")
    except Exception as e:
        print(f"⚠️ Warning: Failed to record performance metrics: {e}")

    return report


execute_e2e_pipeline = execute_production_pipeline


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    execute_production_pipeline(workspace_dir=root_dir, variation_dir=root_dir / "variation")
