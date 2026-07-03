# -*- coding: utf-8 -*-
"""ClipPilot Brain Strategy Engine.

Selects the next video topic and variation configuration intelligently based on
historical performance metrics and exploration/exploitation balancing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from clippilot.brain.analytics_models import TopicPerformance, VideoPerformance
from clippilot.brain.learning_engine import LearningEngine
from clippilot.brain.performance_store import PerformanceStore
from clippilot.brain.pipeline_orchestrator import choose_topic, choose_variation, Topic, PipelineState, VariationRecord


@dataclass
class StrategyWeights:
    """Configurable weights for Strategy Engine decision parameters."""
    exploitation_weight: float = 1.0
    exploration_weight: float = 0.5
    fatigue_weight: float = 2.0


@dataclass
class StrategyDecision:
    """Intelligent next-video configuration chosen by the Strategy Engine."""
    topic: Topic
    hook: str
    voice: str
    skin: str
    format: str
    confidence: float
    reasoning: str
    variation: VariationRecord
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyEngine:
    """Uses UCB utility functions to select next topic/variation configurations."""

    def __init__(self, store: PerformanceStore, learning_engine: LearningEngine, weights: Optional[StrategyWeights] = None):
        self.store = store
        self.learning_engine = learning_engine
        self.weights = weights or StrategyWeights()

    def make_decision(self, state: PipelineState, variation_dir: Path) -> StrategyDecision:
        """Selects the optimal topic and variation candidate based on UCB scoring or cold-start fallback."""
        records = self.store.load_records()
        self.learning_engine.history = records

        # 1. Cold-Start Fallback (fewer than 3 records)
        if len(records) < 3:
            topic = choose_topic(state)
            if not topic:
                raise ValueError("No unused topics found in backlog!")
            topic_cluster = topic.niche or "credit"
            
            try:
                variation = choose_variation(state, variation_dir, topic_cluster)
            except ValueError:
                # If constraints are impossible during cold start, bypass diversity checks
                import random
                from clippilot.brain.pipeline_orchestrator import _parse_ids_from_catalog
                titles = _parse_ids_from_catalog(variation_dir / "title_shapes.md", r"\bT\d+\b") or ["T01"]
                formats = _parse_ids_from_catalog(variation_dir / "formats.md", r"\bF\d+\b") or ["F1"]
                skins = _parse_ids_from_catalog(variation_dir / "visual_skins.md", r"\bS\d+\b") or ["S1"]
                voices = _parse_ids_from_catalog(variation_dir / "voices.md", r"\bV\d+\b") or ["V1"]
                variation = VariationRecord(
                    date="", slug="", title=titles[0], fmt=formats[0], skin=skins[0], voice=voices[0],
                    len="L3", pace="0%", cluster=topic_cluster, hook="question"
                )

            return StrategyDecision(
                topic=topic,
                hook=variation.hook,
                voice=variation.voice,
                skin=variation.skin,
                format=variation.fmt,
                confidence=0.20,
                reasoning="Cold-start: fallback to rotation due to sparse performance history.",
                variation=variation,
                metadata={
                    "exploitation_score": 0.0,
                    "exploration_bonus": 0.0,
                    "fatigue_penalty": 0.0,
                    "final_ucb_score": 0.0,
                    "component_scores": {
                        "topic": 0.0, "hook": 0.0, "voice": 0.0, "skin": 0.0, "format": 0.0
                    }
                }
            )

        # 2. Setup historical metrics
        summary = self.learning_engine.generate_summary()
        total_records = len(records)
        ln_total = math.log(total_records)

        # Extract stats mapping
        def get_stat_map(top_list: List[Dict[str, Any]], metric_name: str) -> Dict[str, float]:
            return {item["name"]: float(item.get(metric_name, 0.0)) for item in top_list}

        hook_ctrs = get_stat_map(summary.top_hooks, "avg_ctr")
        voice_times = get_stat_map(summary.top_voices, "avg_watch_time_seconds")
        skin_retentions = get_stat_map(summary.top_skins, "avg_retention")
        format_views = get_stat_map(summary.top_formats, "avg_views")
        niche_views = get_stat_map(summary.top_niches, "avg_views")

        # Find maximums for normalisation (guarding division by zero)
        max_hook_ctr = max(hook_ctrs.values()) if hook_ctrs else 0.1
        if max_hook_ctr == 0.0: max_hook_ctr = 0.1
        
        max_voice_time = max(voice_times.values()) if voice_times else 10.0
        if max_voice_time == 0.0: max_voice_time = 10.0
        
        max_skin_ret = max(skin_retentions.values()) if skin_retentions else 0.5
        if max_skin_ret == 0.0: max_skin_ret = 0.5
        
        max_format_views = max(format_views.values()) if format_views else 100.0
        if max_format_views == 0.0: max_format_views = 100.0
        
        max_niche_views = max(niche_views.values()) if niche_views else 100.0
        if max_niche_views == 0.0: max_niche_views = 100.0

        # Frequency maps for UCB exploration calculation
        freq_hook: Dict[str, int] = {}
        freq_voice: Dict[str, int] = {}
        freq_skin: Dict[str, int] = {}
        freq_format: Dict[str, int] = {}
        freq_niche: Dict[str, int] = {}

        for r in records:
            h = r.variation.get("hook", "unknown")
            freq_hook[h] = freq_hook.get(h, 0) + 1
            v = r.variation.get("voice", "unknown")
            freq_voice[v] = freq_voice.get(v, 0) + 1
            sk = r.variation.get("skin", "unknown")
            freq_skin[sk] = freq_skin.get(sk, 0) + 1
            fmt = r.variation.get("fmt", "unknown")
            freq_format[fmt] = freq_format.get(fmt, 0) + 1
            nc = r.topic.get("niche", "unknown")
            freq_niche[nc] = freq_niche.get(nc, 0) + 1

        # 3. Parse catalog axes options
        from clippilot.brain.pipeline_orchestrator import _parse_ids_from_catalog
        titles = _parse_ids_from_catalog(variation_dir / "title_shapes.md", r"\bT\d+\b") or ["T01"]
        formats = _parse_ids_from_catalog(variation_dir / "formats.md", r"\bF\d+\b") or ["F1"]
        skins = _parse_ids_from_catalog(variation_dir / "visual_skins.md", r"\bS\d+\b") or ["S1"]
        voices = _parse_ids_from_catalog(variation_dir / "voices.md", r"\bV\d+\b") or ["V1"]
        lengths = ["L1", "L2", "L3", "L4"]
        paces = ["-8%", "-4%", "0%", "+4%", "+6%"]
        hooks = ["question", "whatif", "claim", "story", "number", "mythbust"]

        # Recency lists for fatigue calculations
        recent_runs = state.variation_history[-6:] if state.variation_history else []
        last_niche = state.variation_history[-1].cluster if state.variation_history else None

        recent_titles = {v.title for v in state.variation_history[-6:]} if len(state.variation_history) >= 6 else {v.title for v in state.variation_history}
        recent_formats = {v.fmt for v in state.variation_history[-5:]} if len(state.variation_history) >= 5 else {v.fmt for v in state.variation_history}
        recent_skins = {v.skin for v in state.variation_history[-4:]} if len(state.variation_history) >= 4 else {v.skin for v in state.variation_history}
        recent_voices = {v.voice for v in state.variation_history[-4:]} if len(state.variation_history) >= 4 else {v.voice for v in state.variation_history}

        filtered_titles = [t for t in titles if t not in recent_titles] or titles
        filtered_formats = [f for f in formats if f not in recent_formats] or formats
        filtered_skins = [s for s in skins if s not in recent_skins] or skins
        filtered_voices = [v for v in voices if v not in recent_voices] or voices

        # 4. Generate & Score candidates
        candidates_scored = []
        unused_topics = [t for t in state.topics if t.status.lower().strip() == "unused"]
        if not unused_topics:
            raise ValueError("No unused topics found in backlog!")

        for topic in unused_topics:
            topic_cluster = topic.niche or "credit"
            for title in filtered_titles:
                for fmt in filtered_formats:
                    for skin in filtered_skins:
                        for voice in filtered_voices:
                            for length in lengths:
                                for pace in paces:
                                    for hook in hooks:
                                        # Compute exploitation component scores
                                        h_score = hook_ctrs.get(hook, max_hook_ctr * 0.5) / max_hook_ctr
                                        v_score = voice_times.get(voice, max_voice_time * 0.5) / max_voice_time
                                        s_score = skin_retentions.get(skin, max_skin_ret * 0.5) / max_skin_ret
                                        f_score = format_views.get(fmt, max_format_views * 0.5) / max_format_views
                                        n_score = niche_views.get(topic_cluster, max_niche_views * 0.5) / max_niche_views

                                        exploitation_score = round(
                                            h_score * 0.3 + s_score * 0.3 + v_score * 0.2 + f_score * 0.1 + n_score * 0.1, 4
                                        )

                                        # Compute exploration UCB bonuses
                                        b_hook = math.sqrt(2.0 * ln_total / (freq_hook.get(hook, 0) + 1))
                                        b_voice = math.sqrt(2.0 * ln_total / (freq_voice.get(voice, 0) + 1))
                                        b_skin = math.sqrt(2.0 * ln_total / (freq_skin.get(skin, 0) + 1))
                                        b_format = math.sqrt(2.0 * ln_total / (freq_format.get(fmt, 0) + 1))
                                        b_niche = math.sqrt(2.0 * ln_total / (freq_niche.get(topic_cluster, 0) + 1))

                                        exploration_bonus = round(
                                            (b_hook + b_voice + b_skin + b_format + b_niche) / 5.0, 4
                                        )

                                        # Compute fatigue penalties
                                        recency_penalty = 0.0
                                        for idx, old in enumerate(reversed(recent_runs)):
                                            recency_factor = 1.0 / (idx + 1)
                                            if hook == old.hook: recency_penalty += recency_factor * 0.5
                                            if voice == old.voice: recency_penalty += recency_factor * 0.3
                                            if skin == old.skin: recency_penalty += recency_factor * 0.5
                                            if fmt == old.fmt: recency_penalty += recency_factor * 0.2

                                        niche_match_penalty = 2.0 if last_niche and topic_cluster == last_niche else 0.0

                                        # Diversity check
                                        diversity_penalty = 0.0
                                        for old in recent_runs:
                                            diffs = 0
                                            if title != old.title: diffs += 1
                                            if fmt != old.fmt: diffs += 1
                                            if skin != old.skin: diffs += 1
                                            if voice != old.voice: diffs += 1
                                            if length != old.len: diffs += 1
                                            if topic_cluster != old.cluster: diffs += 1
                                            if hook != old.hook: diffs += 1
                                            if diffs < 3:
                                                diversity_penalty = 5.0
                                                break

                                        fatigue_penalty = round(recency_penalty + niche_match_penalty + diversity_penalty, 4)

                                        final_score = round(
                                            self.weights.exploitation_weight * exploitation_score +
                                            self.weights.exploration_weight * exploration_bonus -
                                            self.weights.fatigue_weight * fatigue_penalty,
                                            4
                                        )

                                        candidates_scored.append({
                                            "topic": topic,
                                            "title": title,
                                            "fmt": fmt,
                                            "skin": skin,
                                            "voice": voice,
                                            "len": length,
                                            "pace": pace,
                                            "hook": hook,
                                            "final_score": final_score,
                                            "exploitation": exploitation_score,
                                            "exploration": exploration_bonus,
                                            "fatigue": fatigue_penalty,
                                            "component_scores": {
                                                "topic": round(n_score, 4),
                                                "hook": round(h_score, 4),
                                                "voice": round(v_score, 4),
                                                "skin": round(s_score, 4),
                                                "format": round(f_score, 4),
                                            }
                                        })

        # Find best candidate
        # Secondary sorting checks key configurations to break ties deterministically
        best = max(candidates_scored, key=lambda c: (c["final_score"], c["title"], c["hook"]))

        variation = VariationRecord(
            date="",
            slug="",
            title=best["title"],
            fmt=best["fmt"],
            skin=best["skin"],
            voice=best["voice"],
            len=best["len"],
            pace=best["pace"],
            cluster=best["topic"].niche or "credit",
            hook=best["hook"]
        )

        confidence = min(1.0, round(0.3 + (total_records / 15.0) * 0.7, 2))

        reasoning = (
            f"Selected topic '{best['topic'].title}' and variation hook '{best['hook']}' "
            f"complying with exploitation (score: {best['exploitation']}) and exploration "
            f"(bonus: {best['exploration']}) weights. Fatigue penalty applied: {best['fatigue']}."
        )

        return StrategyDecision(
            topic=best["topic"],
            hook=best["hook"],
            voice=best["voice"],
            skin=best["skin"],
            format=best["fmt"],
            confidence=confidence,
            reasoning=reasoning,
            variation=variation,
            metadata={
                "exploitation_score": best["exploitation"],
                "exploration_bonus": best["exploration"],
                "fatigue_penalty": best["fatigue"],
                "final_ucb_score": best["final_score"],
                "component_scores": best["component_scores"]
            }
        )
