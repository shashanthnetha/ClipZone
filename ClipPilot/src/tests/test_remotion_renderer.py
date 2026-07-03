# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Remotion Renderer."""
from __future__ import annotations

import unittest

from clippilot.brain.composition_engine import compose_video
from clippilot.brain.pipeline_orchestrator import VariationRecord
from clippilot.brain.remotion_renderer import render_to_remotion
from clippilot.brain.render_compiler import compile_render_graph
from clippilot.brain.render_graph import build_render_graph
from clippilot.brain.scene_planner import plan_scenes
from clippilot.brain.script_generator import Script, ScriptScene


class TestRemotionRenderer(unittest.TestCase):
    """Tests for React component generation, TSX mappings, and Remotion composition boundaries."""

    def test_render_to_remotion_basic(self) -> None:
        script = Script(
            topic_num="006",
            title="Closing card hurts score",
            hook="Stop closing your cards",
            niche_context="credit",
            scenes=[
                ScriptScene("Sentence number one.", "visual 1"),
                ScriptScene("Sentence number two.", "visual 2"),
            ]
        )
        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "credit", "question")
        scene_plan = plan_scenes(script, variation)
        comp = compose_video(scene_plan, width=1080, height=1920, fps=30)
        graph = build_render_graph(comp)
        tree = compile_render_graph(graph)

        output = render_to_remotion(tree)

        # 1. Verify component file structure
        react_tree = output.react_component_tree
        self.assertIn("export const Scene1: React.FC", react_tree)
        self.assertIn("export const Scene2: React.FC", react_tree)

        # 2. Verify subcomponents mapping (elements placement)
        self.assertIn("<Background bgType='dark_radial'", react_tree)
        self.assertIn("<Chart chartType='ScoreDial'", react_tree)
        self.assertIn("<Subtitle words={", react_tree)
        self.assertIn("<Audio src=", react_tree)

        # 3. Verify styling, absolute layout properties and timings
        self.assertIn("position: 'absolute'", react_tree)
        self.assertIn("top: 1440", react_tree)  # subtitle position Y
        self.assertIn("left: 100", react_tree)  # subtitle position X
        self.assertIn('"Sentence", "number", "one."', react_tree)

        # 4. Verify transitions and animations bindings
        self.assertIn("animations={[{ type: 'spring'", react_tree)
        self.assertIn("transitions={[{ type: 'crossfade'", react_tree)

        # 5. Verify root composition registry file boundaries
        root_registry = output.remotion_composition
        self.assertIn("import { Composition, Sequence } from 'remotion';", root_registry)
        self.assertIn("import { Scene1 } from './scenes';", root_registry)
        self.assertIn("<Composition", root_registry)
        self.assertIn("id='Closing-card-hurts-score'", root_registry)
        self.assertIn("<Sequence from={0}", root_registry)
