# -*- coding: utf-8 -*-
"""ClipPilot Brain Director Rules.

Defines deterministic rules and heuristics for mapping input parameters
(topic, strategy decision, learning engine summary) to creative choices.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from clippilot.brain.creative_models import StoryFramework, EmotionCurve

def get_audience(niche: Optional[str]) -> Tuple[str, str]:
    """Deterministically map niche to target audience."""
    n = (niche or "credit").lower()
    if any(k in n for k in ["credit", "finance", "money", "debt", "bnpl", "card"]):
        return (
            "Young adults looking to build credit history, optimize financial cards, and avoid debt traps.",
            f"Selected finance-specific audience because niche contains financial terms ('{n}')."
        )
    if any(k in n for k in ["health", "fitness", "wellness", "diet", "sleep"]):
        return (
            "Health-conscious individuals seeking actionable daily wellness hacks and research-backed physical advice.",
            f"Selected health-specific audience because niche contains wellness terms ('{n}')."
        )
    if any(k in n for k in ["tech", "coding", "software", "ai", "gadget"]):
        return (
            "Tech enthusiasts, developers, and early adopters wanting to keep up with industry trends.",
            f"Selected tech audience because niche contains technology terms ('{n}')."
        )
    return (
        "General social media viewers interested in self-improvement and life optimization tips.",
        f"Selected general audience fallback for niche '{n}'."
    )

def get_story_framework(title: str, angle: str) -> Tuple[StoryFramework, str]:
    """Select the story framework based on title and mechanism angle keywords."""
    t_low = title.lower()
    a_low = angle.lower()
    
    if "myth" in t_low or "myth" in a_low or "debunk" in t_low or "debunk" in a_low or "lie" in t_low or "lie" in a_low:
        fw = StoryFramework(
            name="Myth-Busting",
            description="Exposes a common misconception first, explains why it's false, and delivers the correct/surprising truth."
        )
        reason = "Selected Myth-Busting because title or angle contains myth/debunk/lie keywords."
        return fw, reason
        
    if "why" in t_low or "how" in t_low or "problem" in t_low or "problem" in a_low or "stop" in t_low or "danger" in t_low or "mistake" in t_low or "mistake" in a_low:
        fw = StoryFramework(
            name="Problem-Solution",
            description="Presents a severe, relatable pain point, builds interest around the mechanism, and introduces the solution."
        )
        reason = "Selected Problem-Solution because title or angle indicates a problem, how-to, or cautionary warning."
        return fw, reason

    if "fact" in t_low or "fact" in a_low or "tip" in t_low or "tip" in a_low or "secret" in t_low or "secret" in a_low or "3" in t_low or "three" in t_low or "list" in t_low:
        fw = StoryFramework(
            name="Listicle/Three-Facts",
            description="Presents information as a fast-paced sequence of discrete, high-impact facts or recommendations."
        )
        reason = "Selected Listicle/Three-Facts because title or angle suggests a collection of tips/secrets/facts."
        return fw, reason

    # Default to AIDA marketing funnel
    fw = StoryFramework(
        name="AIDA",
        description="Attention (shock/hook), Interest (curiosity build-up), Desire (relatable value), and Action (CTA)."
    )
    reason = "Selected AIDA framework as the default general-purpose creative structure."
    return fw, reason

def get_emotion_curve(framework_name: str) -> Tuple[EmotionCurve, str]:
    """Select the emotion curve pattern corresponding to the story framework."""
    if framework_name == "Problem-Solution":
        curve = EmotionCurve(
            pattern="Hook-Dip-Rise",
            description="Grab attention at maximum energy, dip into empathy/seriousness when outlining the pain point, and rise to high energy/hope during the solution and final CTA."
        )
        reason = "Selected Hook-Dip-Rise to build empathy during the problem phase of the Problem-Solution framework."
        return curve, reason
        
    if framework_name == "Myth-Busting":
        curve = EmotionCurve(
            pattern="Tension-Release",
            description="Build tension by stating a widely-believed myth/error, release tension with the surprising revelation, and finish with a strong call-to-action."
        )
        reason = "Selected Tension-Release to enhance the contrast between myth and truth in the Myth-Busting framework."
        return curve, reason

    if framework_name == "Listicle/Three-Facts":
        curve = EmotionCurve(
            pattern="Steady-Build",
            description="Start at high interest and continuously increase delivery pace and energy value with each consecutive point, ending at peak energy."
        )
        reason = "Selected Steady-Build to prevent viewer fatigue and maintain high retention across discrete facts."
        return curve, reason

    curve = EmotionCurve(
        pattern="Hook-Rise",
        description="Starts with high hook energy, maintains a steady professional delivery level, and rises to peak excitement during action call."
    )
    reason = "Selected Hook-Rise as a standard, high-retention default emotion curve."
    return curve, reason

def get_pacing_strategy(variation_pace: Optional[str], learning_summary: Optional[Any]) -> Tuple[str, str]:
    """Determine pacing based on variation configuration and historical pacing metrics."""
    p = (variation_pace or "").strip()
    if p and p != "0%":
        reason = f"Selected Fast/Dynamic pacing because variation config requested pace adjustment '{p}'."
        return "Fast/Dynamic", reason
        
    # Check if fast formats perform better historically
    if learning_summary and hasattr(learning_summary, "recommendations") and learning_summary.recommendations:
        rec = learning_summary.recommendations
        # If any learning recommendation mentions fast or dynamic pacing
        if "fast" in str(rec).lower() or "dynamic" in str(rec).lower():
            reason = "Selected Fast/Dynamic pacing based on learning engine recommendations."
            return "Fast/Dynamic", reason

    reason = "Selected Steady/Informative pacing as default for clear concept delivery."
    return "Steady/Informative", reason

def get_cta_strategy(format_str: Optional[str]) -> Tuple[str, str]:
    """Select the Call-To-Action strategy based on format constraints."""
    fmt = (format_str or "F1").upper()
    if "F2" in fmt or "SUB" in fmt or "SHARE" in fmt:
        reason = f"Selected Subscriber Growth CTA because format '{fmt}' prioritizes long-term brand building."
        return "Ask the viewer to subscribe, share the short with a friend, and drop a comment below.", reason
    
    reason = f"Selected High-Value Retention CTA because format '{fmt}' prioritizes immediate user saving and liking."
    return "Instruct the viewer to hit like, save this video for future reference, and drop a comment with questions.", reason

def get_hook_strategy(selected_hook: Optional[str], learning_summary: Optional[Any]) -> Tuple[str, str]:
    """Determine the hook strategy based on chosen hook type or learning statistics."""
    h = (selected_hook or "question").lower()

    if "question" in h:
        reason = f"Selected Question Hook because strategy selected '{h}'."
        return "Question Hook: Ask a direct, highly relatable question confronting the viewer's current state.", reason
    if "curiosity" in h:
        reason = f"Selected Curiosity Gap because strategy selected '{h}'."
        return "Curiosity Gap: State a shocking or counter-intuitive premise without immediately giving the explanation.", reason
    if "shock" in h:
        reason = f"Selected Shocking Statistic because strategy selected '{h}'."
        return "Shocking Statistic: Begin with a hard, high-impact statistic or numerical fact that challenges normal beliefs.", reason
        
    reason = "Selected Direct Challenge: Openly challenge a common practice or status quo choice made by the target audience."
    return "Direct Challenge Hook", reason

def get_visual_and_caption_style(skin_str: Optional[str]) -> Tuple[str, str, str]:
    """Map visual skin style to descriptive rendering and caption design instructions."""
    sk = (skin_str or "S1").upper()
    if sk == "S1":
        return (
            "Modern dark mode tech theme with high-contrast neon accents, glowing outlines, and premium stock overlays.",
            "Bold, neon-colored centered word-by-word karaoke style captions with scale pop micro-animations on active words.",
            f"Selected Neon Tech visual and caption styles to match skin '{sk}'."
        )
    if sk == "S2":
        return (
            "Clean minimalist layout with pastel/light backgrounds, simple elegant geometric grids, and high-quality photography.",
            "Minimalist, classic white subtitles centered in the lower-third with a subtle dark backdrop bar.",
            f"Selected Minimalist visual and caption styles to match skin '{sk}'."
        )
    return (
        "Professional documentary style with clean split-screen compositions, cinematic overlays, and editorial graphics.",
        "Yellow-and-white bold impact fonts in uppercase with thin black outlines.",
        f"Selected Editorial fallback visual and caption styles for skin '{sk}'."
    )
