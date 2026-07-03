# -*- coding: utf-8 -*-
"""ClipPilot Brain Pipeline Orchestrator — Phase 1.

Deterministic startup, state loading, variation rotation, topic selection,
and ledger updating layers.
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from clippilot.brain.parsers import (
    parse_learned_rules,
    parse_markdown_table,
    update_topic_status,
)


@dataclass
class Topic:
    """Represents a row in the daily_topics.md backlog."""
    num: str
    status: str
    title: str
    niche: str
    angle: str
    guardrail: str


@dataclass
class PostRecord:
    """Represents a row in the daily_posts_ledger.md posted log."""
    date: str
    slug: str
    title: str
    topic_num: str
    scheduled: str
    final_mp4: str
    gate: str


@dataclass
class VariationRecord:
    """Represents a row in the variation_ledger.md configuration log."""
    date: str
    slug: str
    title: str  # T-id (e.g., T01)
    fmt: str    # F-id (e.g., F1)
    skin: str   # S-id (e.g., S1)
    voice: str  # V-id (e.g., V1)
    len: str    # L-id (e.g., L3)
    pace: str   # (e.g., -4%)
    cluster: str # (e.g., credit)
    hook: str   # (e.g., question)


@dataclass
class PipelineState:
    """Carries the loaded lists and settings representing the pipeline state."""
    topics: list[Topic] = field(default_factory=list)
    posts_history: list[PostRecord] = field(default_factory=list)
    variation_history: list[VariationRecord] = field(default_factory=list)
    learned_rules: str = ""


@dataclass
class RunContext:
    """Represents the context parameters for a single pipeline execution."""
    workspace_dir: Path
    variation_dir: Path
    date_str: str
    slug: str


def load_state(context: RunContext) -> PipelineState:
    """Load and parse the state files from the workspace directory."""
    topics_path = context.workspace_dir / "daily_topics.md"
    posts_path = context.workspace_dir / "daily_posts_ledger.md"
    variation_path = context.workspace_dir / "variation_ledger.md"
    skill_path = context.workspace_dir / ".claude" / "skills" / "ultimate-short" / "SKILL.md"

    state = PipelineState()

    if topics_path.exists():
        raw_topics = parse_markdown_table(topics_path.read_text(encoding="utf-8"))
        for t in raw_topics:
            state.topics.append(
                Topic(
                    num=t.get("num", ""),
                    status=t.get("status", ""),
                    title=t.get("working title (viral pattern)", t.get("title", "")),
                    niche=t.get("niche / cpm", t.get("niche", "")),
                    angle=t.get("angle (the true mechanism)", t.get("angle", "")),
                    guardrail=t.get("brand-safety guardrail", t.get("guardrail", "")),
                )
            )

    if posts_path.exists():
        raw_posts = parse_markdown_table(posts_path.read_text(encoding="utf-8"))
        for p in raw_posts:
            state.posts_history.append(
                PostRecord(
                    date=p.get("date", ""),
                    slug=p.get("slug", ""),
                    title=p.get("title", ""),
                    topic_num=p.get("topic num", p.get("topic", "")),
                    scheduled=p.get("scheduled (local)", p.get("scheduled", "")),
                    final_mp4=p.get("final mp4", ""),
                    gate=p.get("gate", ""),
                )
            )

    if variation_path.exists():
        raw_variations = parse_markdown_table(variation_path.read_text(encoding="utf-8"))
        for v in raw_variations:
            state.variation_history.append(
                VariationRecord(
                    date=v.get("date", ""),
                    slug=v.get("slug", ""),
                    title=v.get("title", ""),
                    fmt=v.get("fmt", ""),
                    skin=v.get("skin", ""),
                    voice=v.get("voice", ""),
                    len=v.get("len", ""),
                    pace=v.get("pace", ""),
                    cluster=v.get("cluster", ""),
                    hook=v.get("hook", ""),
                )
            )

    if skill_path.exists():
        state.learned_rules = parse_learned_rules(skill_path.read_text(encoding="utf-8"))

    return state


def choose_topic(state: PipelineState) -> Optional[Topic]:
    """Find and return the first topic marked 'unused' in the backlog."""
    for topic in state.topics:
        if topic.status.lower().strip() == "unused":
            return topic
    return None


def _parse_ids_from_catalog(catalog_path: Path, pattern: str) -> list[str]:
    """Helper to extract distinct IDs from catalog files matching a regex pattern."""
    if not catalog_path.exists():
        return []
    content = catalog_path.read_text(encoding="utf-8")
    return sorted(list(set(re.findall(pattern, content))))


def choose_variation(
    state: PipelineState,
    variation_dir: Path,
    topic_cluster: str,
    seed: Optional[int] = None,
) -> VariationRecord:
    """Select a new distinct variation combo complying with rotation and diversity rules.

    Raises ValueError if no valid combinations can satisfy all constraints.
    """
    # 1. Parse catalog options
    titles = _parse_ids_from_catalog(variation_dir / "title_shapes.md", r"\bT\d+\b")
    formats = _parse_ids_from_catalog(variation_dir / "formats.md", r"\bF\d+\b")
    skins = _parse_ids_from_catalog(variation_dir / "visual_skins.md", r"\bS\d+\b")
    voices = _parse_ids_from_catalog(variation_dir / "voices.md", r"\bV\d+\b")

    lengths = ["L1", "L2", "L3", "L4"]
    paces = ["-8%", "-4%", "0%", "+4%", "+6%"]
    hooks = ["question", "whatif", "claim", "story", "number", "mythbust"]

    history = state.variation_history
    last_6 = history[-6:] if history else []

    # 2. Extract rotation sets
    recent_titles = {v.title for v in history[-6:]} if len(history) >= 6 else {v.title for v in history}
    recent_formats = {v.fmt for v in history[-5:]} if len(history) >= 5 else {v.fmt for v in history}
    recent_skins = {v.skin for v in history[-4:]} if len(history) >= 4 else {v.skin for v in history}
    recent_voices = {v.voice for v in history[-4:]} if len(history) >= 4 else {v.voice for v in history}

    # 3. Generate and filter candidate options combinatorially
    candidates = []
    for title in titles:
        if title in recent_titles:
            continue
        for fmt in formats:
            if fmt in recent_formats:
                continue
            for skin in skins:
                if skin in recent_skins:
                    continue
                for voice in voices:
                    if voice in recent_voices:
                        continue
                    for length in lengths:
                        for pace in paces:
                            for hook in hooks:
                                candidates.append((title, fmt, skin, voice, length, pace, hook))

    # Determine last cluster for back-to-back constraint
    last_cluster = history[-1].cluster if history else None

    # Stable sort to ensure reproducibility
    candidates.sort()

    # If seed is provided, shuffle deterministically using seed
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(candidates)

    # 4. Find the first combo satisfying the diversity gate and cluster constraints
    for title, fmt, skin, voice, length, pace, hook in candidates:
        # Niche cluster cannot match back-to-back
        if last_cluster and topic_cluster == last_cluster:
            continue

        # Check diversity against EACH of the last 6 runs (must differ on >= 3 axes)
        passed_diversity = True
        for old in last_6:
            diffs = 0
            if title != old.title:
                diffs += 1
            if fmt != old.fmt:
                diffs += 1
            if skin != old.skin:
                diffs += 1
            if voice != old.voice:
                diffs += 1
            if length != old.len:
                diffs += 1
            if topic_cluster != old.cluster:
                diffs += 1
            if hook != old.hook:
                diffs += 1

            if diffs < 3:
                passed_diversity = False
                break

        if passed_diversity:
            return VariationRecord(
                date="",  # Filled on write
                slug="",  # Filled on write
                title=title,
                fmt=fmt,
                skin=skin,
                voice=voice,
                len=length,
                pace=pace,
                cluster=topic_cluster,
                hook=hook,
            )

    raise ValueError("Impossible variation constraints: No combination satisfies rotation and diversity rules.")


def update_ledgers(
    context: RunContext,
    state: PipelineState,
    selected_topic: Topic,
    selected_variation: VariationRecord,
) -> None:
    """Update topic status, posts ledger, and variation ledger files in-place."""
    # 1. Update daily_topics.md status
    topics_path = context.workspace_dir / "daily_topics.md"
    if topics_path.exists():
        content = topics_path.read_text(encoding="utf-8")
        updated_content = update_topic_status(content, selected_topic.num, f"USED {context.date_str}")
        topics_path.write_text(updated_content, encoding="utf-8")

    # 2. Append to daily_posts_ledger.md
    posts_path = context.workspace_dir / "daily_posts_ledger.md"
    if posts_path.exists():
        content = posts_path.read_text(encoding="utf-8")
        # Format matching the original markdown style
        # e.g. | 2026-07-01 | daily006_tonguemap | Why everything you learned about your tongue is wrong | 006 | ...
        new_row = (
            f"| {context.date_str} | {context.slug} | {selected_topic.title} | {selected_topic.num} | "
            f"✅ LIVE (staged) | ClipPilot/remotion_explainer/out/{context.slug}_final.mp4 | "
            f"greenlit ({selected_topic.angle[:30]}...) |\n"
        )
        # Append before terminal newline or at the end
        if content.endswith("\n"):
            content += new_row
        else:
            content += "\n" + new_row
        posts_path.write_text(content, encoding="utf-8")

    # 3. Append to variation_ledger.md
    variation_path = context.workspace_dir / "variation_ledger.md"
    if variation_path.exists():
        content = variation_path.read_text(encoding="utf-8")
        # Format matching: | Date | Slug | title | fmt | skin | voice | len | pace | cluster | hook |
        new_row = (
            f"| {context.date_str} | {context.slug} | {selected_variation.title} | {selected_variation.fmt} | "
            f"{selected_variation.skin} | {selected_variation.voice} | {selected_variation.len} | "
            f"{selected_variation.pace} | {selected_variation.cluster} | {selected_variation.hook} |\n"
        )
        if content.endswith("\n"):
            content += new_row
        else:
            content += "\n" + new_row
        variation_path.write_text(content, encoding="utf-8")
