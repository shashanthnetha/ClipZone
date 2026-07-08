# -*- coding: utf-8 -*-
"""ClipPilot Brain Director Rules.

Defines deterministic rules and heuristics for mapping input parameters
(topic, strategy decision, learning engine summary) to creative choices.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from clippilot.brain.creative_models import StoryFramework, EmotionCurve, SceneBlueprint, CreativeBlueprint

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

def get_transition_style(skin_str: Optional[str]) -> str:
    """Map skin style to transition style."""
    sk = (skin_str or "S1").upper()
    if sk == "S2":
        return "reveal_wipe"
    if sk == "S5":
        return "hardcut"
    if sk in ["S3", "S6"]:
        return "crossfade"
    return "swipe"

def get_animation_style(skin_str: Optional[str]) -> str:
    """Map skin style to primary animation style."""
    sk = (skin_str or "S1").upper()
    if sk == "S1":
        return "spring"
    if sk == "S2":
        return "draw_path"
    if sk == "S3":
        return "bounce"
    if sk == "S4":
        return "counter_roll"
    if sk == "S5":
        return "snappy_drop"
    return "glow_pulse"

def get_camera_language(skin_str: Optional[str]) -> str:
    """Map skin style to camera language movements."""
    sk = (skin_str or "S1").upper()
    if sk in ["S1", "S5"]:
        return "slow_zoom"
    if sk == "S3":
        return "pan_left"
    if sk == "S4":
        return "pan_right"
    return "static"

def get_emphasis_color(skin_str: Optional[str]) -> str:
    """Map skin style to highlight/emphasis color."""
    sk = (skin_str or "S1").upper()
    colors = {
        "S1": "#00ffcc",  # Neon cyan
        "S2": "#ff5577",  # Rose/coral
        "S3": "#fca311",  # Warm yellow
        "S4": "#5588ff",  # Soft blue
        "S5": "#e67e22",  # Retro orange
        "S6": "#ff00ff",  # Neon magenta
    }
    return colors.get(sk, "#fca311")

def make_scene_blueprints(blueprint: CreativeBlueprint) -> List[SceneBlueprint]:
    """Deterministically generates SceneBlueprint instances for the entire video timeline."""
    total_scenes = blueprint.target_scene_count
    duration_per_scene = round(blueprint.target_duration / total_scenes, 2)
    
    framework = blueprint.story_framework.name
    visual_style = blueprint.visual_style
    caption_style = blueprint.caption_style
    
    # Extract settings from the blueprint
    metadata = blueprint.metadata
    decision_reasoning = metadata.get("decision_reasoning", {})
    skin_str = "S1"
    for r in decision_reasoning.values():
        if "skin" in r:
            parts = r.split("'")
            if len(parts) >= 2:
                skin_str = parts[1]
                break
                
    transition = get_transition_style(skin_str)
    animation = get_animation_style(skin_str)
    camera = get_camera_language(skin_str)
    emphasis_color = get_emphasis_color(skin_str)

    blueprints = []
    
    for i in range(total_scenes):
        scene_num = i + 1
        
        # 1. Determine retention stage
        if scene_num == 1:
            stage = "Hook"
        elif scene_num == total_scenes:
            stage = "CTA"
        elif scene_num == 2:
            stage = "Build"
        elif scene_num == total_scenes - 1:
            stage = "Reveal"
        else:
            stage = "Rehook" if scene_num % 2 == 1 else "Payoff"
            
        # 2. Determine emotional intensity
        if stage == "Hook":
            intensity = "High"
        elif stage == "CTA":
            intensity = "Medium"
        elif stage == "Reveal" or stage == "Payoff":
            intensity = "High"
        else:
            intensity = "Medium" if framework == "Listicle/Three-Facts" else "Low"

        # 3. Determine visual priority to ensure variation across scenes
        priorities = ["stock_video", "stock_image", "motion_graphic", "icon", "text_only"]
        priority = priorities[i % len(priorities)]
        if stage == "CTA":
            priority = "text_only"
        elif stage == "Hook":
            priority = "stock_video"
            
        # 4. Map goals and visual intents based on framework and position
        if framework == "Problem-Solution":
            if stage == "Hook":
                goal = "Confront the viewer with the painful problem/question"
                intent = "Frustrated person encountering a financial or tech roadblock"
                narration = "Establish the main pain point/question immediately"
                queries = ["frustrated person looking at screen", "problem warning message"]
                primary_visual = "warning_sign"
            elif stage == "Build":
                goal = "Detail the root cause or mechanism behind the problem"
                intent = "Complex data analytical chart showing mechanism details"
                narration = "Explain why this roadblock happens under the hood"
                queries = ["financial transaction graph", "analysis process workflow"]
                primary_visual = "analytical_chart"
            elif stage == "Reveal":
                goal = "Reveal the shocking solution or secret breakthrough"
                intent = "Exciting positive trend or glowing interface reveal"
                narration = "Present the key solution mechanism to resolve the pain"
                queries = ["success growth chart", "man smiling happy screen"]
                primary_visual = "success_graph"
            elif stage == "CTA":
                goal = "Instruct the viewer to take action (like, save, comment)"
                intent = "Clean graphic with call to action overlay text"
                narration = "Encourage retention engagement steps"
                queries = ["subscribe arrow click button", "thumbs up like animation"]
                primary_visual = "cta_card"
            else:
                goal = "Rehook the audience with a secondary problem nuance"
                intent = "Intriguing outline or warning graphic"
                narration = "Add extra tension to avoid dropping off"
                queries = ["declining report dashboard", "warning screen"]
                primary_visual = "secondary_warning"
                
        elif framework == "Myth-Busting":
            if stage == "Hook":
                goal = "State the widely believed myth or misconception"
                intent = "Skeptical face or error symbol representing false belief"
                narration = "Expose the myth as a shocking lie"
                queries = ["skeptical look expression", "error incorrect wrong sign"]
                primary_visual = "myth_alert"
            elif stage == "Reveal":
                goal = "Expose the correct surprising truth with facts"
                intent = "Shocking revelation glowing fact outline"
                narration = "Disprove the myth with real scientific/data facts"
                queries = ["shocked face reaction", "scientific data research concept"]
                primary_visual = "truth_reveal"
            elif stage == "CTA":
                goal = "Final summary action request"
                intent = "Clear recommendation summary bullet card"
                narration = "Final CTA wrap-up"
                queries = ["checklist summary points", "checkmark list icons"]
                primary_visual = "final_checkmark"
            else:
                goal = "Build supporting evidence for debunking"
                intent = "Evidence grid showing true comparison"
                narration = "Deep dive details debunking the myth"
                queries = ["comparison table layout", "credit score rating dials"]
                primary_visual = "comparison_graph"
                
        elif framework == "Listicle/Three-Facts":
            if stage == "Hook":
                goal = "Introduce the catalog list of tips/secrets"
                intent = "Eye-catching collection catalog listing intro card"
                narration = "Hook with valuable secrets proposal"
                queries = ["secrets box reveal icon", "top tips checklist"]
                primary_visual = "secrets_box"
            elif stage == "CTA":
                goal = "Remind the viewer to save the list"
                intent = "Final checklist card summary with save reminder"
                narration = "Ask to save for future reference"
                queries = ["save bookmark icon click", "like share button click"]
                primary_visual = "save_icon"
            else:
                point_num = scene_num - 1
                goal = f"Present Point #{point_num} of the high value list"
                intent = f"Point {point_num} illustration visual"
                narration = f"Explain recommendation point #{point_num}"
                queries = [f"key idea point {point_num}", "info visual map"]
                primary_visual = f"point_{point_num}_badge"
                
        else:  # AIDA / Default
            if stage == "Hook":
                goal = "Attention: grab eyes with high-contrast energy"
                intent = "Vibrant glowing outline or warning overlay"
                narration = "Open with maximum energy hook sentence"
                queries = ["neon warning graphic", "alert flashing lights"]
                primary_visual = "attention_icon"
            elif stage == "Build":
                goal = "Interest: build curiosity with stats or questions"
                intent = "Question marks or curiosity gap diagram"
                narration = "Develop the core concept details"
                queries = ["confused person questioning", "curiosity map"]
                primary_visual = "interest_map"
            elif stage == "Reveal":
                goal = "Desire: outline the valuable benefits"
                intent = "Positive progress bar or success indicators"
                narration = "Make them want the solution"
                queries = ["rising finance business trend", "successful dashboard key"]
                primary_visual = "desire_chart"
            elif stage == "CTA":
                goal = "Action: prompt for immediate interaction"
                intent = "Social media save/like icons list"
                narration = "CTA instruction"
                queries = ["click save video animation", "thumbs up icon"]
                primary_visual = "action_dials"
            else:
                goal = "Maintain engagement through transition"
                intent = "Steady narrative stock footage"
                narration = "Keep delivery pacing steady"
                queries = ["abstract dark digital waves", "background digital grid"]
                primary_visual = "pacing_grid"

        blueprints.append(
            SceneBlueprint(
                scene_number=scene_num,
                duration=duration_per_scene,
                scene_goal=goal,
                visual_intent=intent,
                visual_priority=priority,
                narration_purpose=narration,
                emotional_intensity=intensity,
                retention_stage=stage,
                visual_style=visual_style,
                animation_style=animation,
                transition=transition,
                camera_language=camera,
                caption_behavior=f"emphasize active words with {emphasis_color} in center karaoke style",
                search_queries=queries,
                fallback_icon="default_icon.png",
                primary_visual=primary_visual
            )
        )
        
    return blueprints
