# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Script Critic and Revision Orchestration."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from clippilot.brain.pipeline_orchestrator import PipelineState, Topic, VariationRecord
from clippilot.brain.script_generator import Script, ScriptScene
from clippilot.brain.script_critic import (
    CategoryFeedback,
    CriticResult,
    run_script_critic,
    run_script_rewrite,
    orchestrate_script_revision,
)


class TestScriptCritic(unittest.TestCase):
    """Tests for script quality scoring, mock/real evaluation, and revision loop mechanics."""

    def setUp(self) -> None:
        import os
        self.old_env = os.environ.copy()
        os.environ["LLM_API_KEY"] = "mock_key"

        self.state = PipelineState(learned_rules="Rule A")
        self.topic = Topic("001", "unused", "Title Proposal", "niche", "angle", "guardrail")
        self.variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "cluster", "hook")
        self.script = Script(
            topic_num="001",
            title="Old Title",
            hook="Old Hook",
            niche_context="Context",
            scenes=[ScriptScene(narration="Old Line", visual_desc="Old View")],
            metadata={}
        )

    def tearDown(self) -> None:
        import os
        os.environ.clear()
        os.environ.update(self.old_env)

    def test_critic_mock_mode(self) -> None:
        import os
        if "LLM_API_KEY" in os.environ:
            del os.environ["LLM_API_KEY"]
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        from clippilot.config import Settings
        with patch("clippilot.config.Settings.load") as mock_settings_load:
            mock_settings = MagicMock()
            mock_settings.llm_api_key = None
            mock_settings_load.return_value = mock_settings

            # First critique of original script (returns overall = 7.8)
            critique = run_script_critic(self.script, mock_settings)
            self.assertEqual(critique.overall.score, 7.8)
            self.assertEqual(critique.hook.score, 7.8)

            # Critique of rewritten script (returns overall = 8.8)
            rewritten_script = Script(
                topic_num="001",
                title="Why You MUST Stop Closing Credit Cards",
                hook="Stop closing credit cards immediately!",
                niche_context="Context",
                scenes=[ScriptScene(narration="Line", visual_desc="View")]
            )
            rewritten_critique = run_script_critic(rewritten_script, mock_settings)
            self.assertEqual(rewritten_critique.overall.score, 8.8)

    @patch("clippilot.brain.script_critic.get_provider")
    def test_run_script_critic_success(self, mock_get_provider: MagicMock) -> None:
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        valid_response = (
            "{\n"
            "  \"hook\": { \"score\": 9.0, \"reason\": \"Good\", \"improvement\": \"None\" },\n"
            "  \"curiosity\": { \"score\": 8.5, \"reason\": \"Good\", \"improvement\": \"None\" },\n"
            "  \"clarity\": { \"score\": 9.5, \"reason\": \"Clear\", \"improvement\": \"None\" },\n"
            "  \"retention\": { \"score\": 8.0, \"reason\": \"Okay\", \"improvement\": \"More visual\" },\n"
            "  \"cta\": { \"score\": 9.0, \"reason\": \"Good\", \"improvement\": \"None\" },\n"
            "  \"overall\": { \"score\": 8.9, \"reason\": \"Strong\", \"improvement\": \"None\" }\n"
            "}"
        )
        mock_provider.generate_text.return_value = valid_response

        from clippilot.config import Settings
        settings = Settings.load()
        result = run_script_critic(self.script, settings)

        self.assertEqual(result.overall.score, 8.9)
        self.assertEqual(result.overall.reason, "Strong")
        self.assertEqual(result.overall.improvement, "None")
        self.assertEqual(result.hook.score, 9.0)
        self.assertEqual(result.retention.score, 8.0)
        self.assertEqual(result.retention.improvement, "More visual")

    @patch("clippilot.brain.script_critic.get_provider")
    def test_run_script_rewrite_success(self, mock_get_provider: MagicMock) -> None:
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        valid_script_response = (
            "{\n"
            "  \"title\": \"Improved Title\",\n"
            "  \"hook\": \"Improved Hook\",\n"
            "  \"niche_context\": \"Improved Context\",\n"
            "  \"scenes\": [\n"
            "    { \"narration\": \"Improved Line\", \"visual_desc\": \"Improved View\" }\n"
            "  ]\n"
            "}"
        )
        mock_provider.generate_text.return_value = valid_script_response

        from clippilot.config import Settings
        settings = Settings.load()

        feedback = CriticResult(
            hook=CategoryFeedback(score=6.0, reason="Weak", improvement="Punchier"),
            curiosity=CategoryFeedback(score=6.0, reason="Weak", improvement="Punchier"),
            clarity=CategoryFeedback(score=6.0, reason="Weak", improvement="Punchier"),
            retention=CategoryFeedback(score=6.0, reason="Weak", improvement="Punchier"),
            cta=CategoryFeedback(score=6.0, reason="Weak", improvement="Punchier"),
            overall=CategoryFeedback(score=6.0, reason="Weak", improvement="Punchier")
        )

        improved = run_script_rewrite(self.script, feedback, settings)
        self.assertEqual(improved.title, "Improved Title")
        self.assertEqual(improved.hook, "Improved Hook")
        self.assertEqual(improved.scenes[0].narration, "Improved Line")

    @patch("clippilot.brain.script_critic.run_script_critic")
    @patch("clippilot.brain.script_critic.run_script_rewrite")
    def test_orchestration_no_revision_needed(
        self, mock_rewrite: MagicMock, mock_critic: MagicMock
    ) -> None:
        # High score -> skip revision
        mock_critic.return_value = CriticResult(
            hook=CategoryFeedback(score=9.0, reason="Good", improvement="None"),
            curiosity=CategoryFeedback(score=9.0, reason="Good", improvement="None"),
            clarity=CategoryFeedback(score=9.0, reason="Good", improvement="None"),
            retention=CategoryFeedback(score=9.0, reason="Good", improvement="None"),
            cta=CategoryFeedback(score=9.0, reason="Good", improvement="None"),
            overall=CategoryFeedback(score=9.0, reason="Good", improvement="None")
        )

        final_script = orchestrate_script_revision(
            self.state, self.topic, self.variation, Path("."), self.script, threshold=8.5
        )

        self.assertEqual(final_script.metadata["initial_score"], 9.0)
        self.assertEqual(final_script.metadata["final_score"], 9.0)
        self.assertEqual(final_script.metadata["revision_count"], 0)
        self.assertFalse(final_script.metadata["accepted_revision"])
        mock_rewrite.assert_not_called()

    @patch("clippilot.brain.script_critic.run_script_critic")
    @patch("clippilot.brain.script_critic.run_script_rewrite")
    def test_orchestration_revision_accepted(
        self, mock_rewrite: MagicMock, mock_critic: MagicMock
    ) -> None:
        # Low score (7.0), then rewrite, then high score (9.0)
        initial_critique = CriticResult(
            hook=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier"),
            curiosity=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier"),
            clarity=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier"),
            retention=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier"),
            cta=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier"),
            overall=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier")
        )
        final_critique = CriticResult(
            hook=CategoryFeedback(score=9.0, reason="Good", improvement="None"),
            curiosity=CategoryFeedback(score=9.0, reason="Good", improvement="None"),
            clarity=CategoryFeedback(score=9.0, reason="Good", improvement="None"),
            retention=CategoryFeedback(score=9.0, reason="Good", improvement="None"),
            cta=CategoryFeedback(score=9.0, reason="Good", improvement="None"),
            overall=CategoryFeedback(score=9.0, reason="Good", improvement="None")
        )
        mock_critic.side_effect = [initial_critique, final_critique]

        rewritten_script = Script(
            topic_num="001",
            title="New Title",
            hook="New Hook",
            niche_context="Context",
            scenes=[ScriptScene(narration="New Line", visual_desc="New View")],
            metadata={}
        )
        mock_rewrite.return_value = rewritten_script

        final_script = orchestrate_script_revision(
            self.state, self.topic, self.variation, Path("."), self.script, threshold=8.5
        )

        self.assertEqual(final_script.title, "New Title")
        self.assertEqual(final_script.metadata["initial_score"], 7.0)
        self.assertEqual(final_script.metadata["final_score"], 9.0)
        self.assertEqual(final_script.metadata["revision_count"], 1)
        self.assertTrue(final_script.metadata["accepted_revision"])
        mock_rewrite.assert_called_once()

    @patch("clippilot.brain.script_critic.run_script_critic")
    @patch("clippilot.brain.script_critic.run_script_rewrite")
    def test_orchestration_revision_rejected(
        self, mock_rewrite: MagicMock, mock_critic: MagicMock
    ) -> None:
        # Low score (7.0), then rewrite, but revised score is worse (6.5) -> keep original
        initial_critique = CriticResult(
            hook=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier"),
            curiosity=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier"),
            clarity=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier"),
            retention=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier"),
            cta=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier"),
            overall=CategoryFeedback(score=7.0, reason="Weak", improvement="Punchier")
        )
        final_critique = CriticResult(
            hook=CategoryFeedback(score=6.5, reason="Worse", improvement="None"),
            curiosity=CategoryFeedback(score=6.5, reason="Worse", improvement="None"),
            clarity=CategoryFeedback(score=6.5, reason="Worse", improvement="None"),
            retention=CategoryFeedback(score=6.5, reason="Worse", improvement="None"),
            cta=CategoryFeedback(score=6.5, reason="Worse", improvement="None"),
            overall=CategoryFeedback(score=6.5, reason="Worse", improvement="None")
        )
        mock_critic.side_effect = [initial_critique, final_critique]

        rewritten_script = Script(
            topic_num="001",
            title="New Title",
            hook="New Hook",
            niche_context="Context",
            scenes=[ScriptScene(narration="New Line", visual_desc="New View")],
            metadata={}
        )
        mock_rewrite.return_value = rewritten_script

        final_script = orchestrate_script_revision(
            self.state, self.topic, self.variation, Path("."), self.script, threshold=8.5
        )

        self.assertEqual(final_script.title, "Old Title")
        self.assertEqual(final_script.metadata["initial_score"], 7.0)
        self.assertEqual(final_script.metadata["final_score"], 7.0)
        self.assertEqual(final_script.metadata["revision_count"], 1)
        self.assertFalse(final_script.metadata["accepted_revision"])
        mock_rewrite.assert_called_once()
