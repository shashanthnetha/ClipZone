# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Vision QA Layer."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from clippilot.brain.composition_engine import compose_video
from clippilot.brain.pipeline_orchestrator import VariationRecord
from clippilot.brain.render_compiler import compile_render_graph
from clippilot.brain.render_graph import build_render_graph
from clippilot.brain.scene_planner import plan_scenes
from clippilot.brain.script_generator import Script, ScriptScene
from clippilot.brain.vision_provider import (
    OpenAICompatibleVisionProvider,
    VisionRequest,
    VisionResult,
    get_vision_provider,
)
from clippilot.brain.vision_qa import (
    FrameAnalysis,
    QAResult,
    _parse_qa_vision_response,
    run_vision_qa,
    sample_frames_timestamps,
)


class TestVisionQA(unittest.TestCase):
    """Tests for frame sampling rules, Vision responses parsing, and QA score generations."""

    def test_get_vision_provider_resolution(self) -> None:
        provider = get_vision_provider("openai-compatible")
        self.assertIsInstance(provider, OpenAICompatibleVisionProvider)

        with self.assertRaises(ValueError):
            get_vision_provider("unknown_vision")

    def test_sample_frames_timestamps(self) -> None:
        script = Script(
            topic_num="006",
            title="Closing card hurts score",
            hook="Stop closing",
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

        # "Sentence number one." = 3 words. Duration = max(1.5, 3/2.5) = 1.5s
        # Scene 1 midpoint = 0.75s
        # Scene 2 midpoint = 1.5 + 0.75 = 2.25s
        timestamps = sample_frames_timestamps(tree)
        self.assertEqual(len(timestamps), 2)
        self.assertAlmostEqual(timestamps[0], 0.75)
        self.assertAlmostEqual(timestamps[1], 2.25)

    def test_parse_qa_vision_response(self) -> None:
        valid_response = (
            "```json\n"
            "{\n"
            "  \"score\": 95,\n"
            "  \"blank_frame_detected\": false,\n"
            "  \"subtitle_clipping_detected\": false,\n"
            "  \"ocr_mismatches\": [],\n"
            "  \"summary\": \"Looks excellent.\",\n"
            "  \"analyses\": [\n"
            "    {\"frame_seconds\": 0.75, \"subtitle_visible\": true, \"subtitle_clipped\": false, \"blank_frame\": false, \"ocr_text\": \"Sentence\"}\n"
            "  ]\n"
            "}\n"
            "```"
        )
        parsed = _parse_qa_vision_response(valid_response)
        self.assertEqual(parsed["score"], 95)
        self.assertEqual(len(parsed["analyses"]), 1)

    @patch("clippilot.brain.vision_qa.get_vision_provider")
    def test_run_vision_qa_simulated_fallback(self, mock_get_vision: MagicMock) -> None:
        # Non-existent image paths should cause simulated fallback (100 score)
        script = Script("006", "Closing", "Stop", "credit", [ScriptScene("Line A.", "V")])
        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "credit", "question")
        scene_plan = plan_scenes(script, variation)
        comp = compose_video(scene_plan, width=1080, height=1920, fps=30)
        graph = build_render_graph(comp)
        tree = compile_render_graph(graph)

        res = run_vision_qa("out.mp4", script, tree, [], ["fake_path.png"])

        self.assertTrue(res.passed)
        self.assertEqual(res.score, 100)
        self.assertEqual(res.summary, "QA completed successfully (simulated fallback).")
        mock_get_vision.assert_not_called()

    def test_run_vision_qa_real_mocked(self) -> None:
        mock_provider = MagicMock()
        mock_provider.supports_vision.return_value = True

        response_dict = {
            "score": 92,
            "blank_frame_detected": False,
            "subtitle_clipping_detected": False,
            "ocr_mismatches": [],
            "summary": "Looks great.",
            "analyses": [
                {"frame_seconds": 0.75, "subtitle_visible": True, "subtitle_clipped": False, "blank_frame": False, "ocr_text": "Line"}
            ]
        }
        mock_provider.generate_vision.return_value = response_dict

        script = Script("006", "Closing", "Stop", "credit", [ScriptScene("Line A.", "V")])
        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "credit", "question")
        scene_plan = plan_scenes(script, variation)
        comp = compose_video(scene_plan, width=1080, height=1920, fps=30)
        graph = build_render_graph(comp)
        tree = compile_render_graph(graph)

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_img = Path(tmp_dir) / "frame1.png"
            temp_img.write_bytes(b"dummy image data")

            res = run_vision_qa("out.mp4", script, tree, [], [str(temp_img)], provider=mock_provider)

            self.assertTrue(res.passed)
            self.assertEqual(res.score, 92)
            self.assertEqual(len(res.frame_analyses), 1)
            self.assertEqual(res.frame_analyses[0].ocr_text, "Line")
            mock_provider.generate_vision.assert_called_once()
window = None
