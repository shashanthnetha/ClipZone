# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Script Generator."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from clippilot.brain.pipeline_orchestrator import PipelineState, Topic, VariationRecord
from clippilot.brain.script_generator import (
    Script,
    ScriptValidationError,
    _clean_json_text,
    _validate_and_parse_json,
    generate_script,
)


class TestScriptGenerator(unittest.TestCase):
    """Tests for prompt builder, JSON validation, and LLM retry mechanics."""

    def test_clean_json_text(self) -> None:
        # Case A: Markdown code fence
        fenced = "Some introduction text.\n```json\n{\n  \"title\": \"Hello\"\n}\n```\nSome footer."
        self.assertEqual(_clean_json_text(fenced), '{\n  "title": "Hello"\n}')

        # Case B: Plain JSON wrapped in garbage text
        garbage = "Here is the response: {\"test\": 123} - hope you like it."
        self.assertEqual(_clean_json_text(garbage), '{"test": 123}')

    def test_validate_and_parse_json_valid(self) -> None:
        valid_json = (
            "{\n"
            "  \"title\": \"My Title\",\n"
            "  \"hook\": \"Un-swipeable Hook\",\n"
            "  \"niche_context\": \"CPM Credit\",\n"
            "  \"scenes\": [\n"
            "    {\"narration\": \"Scene 1\", \"visual_desc\": \"Icon dials\"}\n"
            "  ]\n"
            "}"
        )
        parsed = _validate_and_parse_json(valid_json)
        self.assertEqual(parsed["title"], "My Title")
        self.assertEqual(parsed["scenes"][0]["narration"], "Scene 1")

    def test_validate_and_parse_json_invalid(self) -> None:
        # Missing key
        invalid_json = "{\"title\": \"No scenes or hook\"}"
        with self.assertRaises(ScriptValidationError):
            _validate_and_parse_json(invalid_json)

        # Scenes is not a list
        bad_scenes = "{\"title\": \"A\", \"hook\": \"B\", \"niche_context\": \"C\", \"scenes\": \"not a list\"}"
        with self.assertRaises(ScriptValidationError):
            _validate_and_parse_json(bad_scenes)

    def setUp(self) -> None:
        import os
        self.old_env = os.environ.copy()
        os.environ["LLM_API_KEY"] = "mock_key"

    def tearDown(self) -> None:
        import os
        os.environ.clear()
        os.environ.update(self.old_env)

    @patch("clippilot.brain.script_generator.get_provider")
    def test_generate_script_success(self, mock_get_provider: unittest.mock.MagicMock) -> None:
        mock_provider = unittest.mock.MagicMock()
        mock_provider.model = "claude-test"
        mock_get_provider.return_value = mock_provider

        valid_response = (
            "{\n"
            "  \"title\": \"Title 1\",\n"
            "  \"hook\": \"Hook 1\",\n"
            "  \"niche_context\": \"Context 1\",\n"
            "  \"scenes\": [\n"
            "    {\"narration\": \"Line 1\", \"visual_desc\": \"View 1\"}\n"
            "  ]\n"
            "}"
        )
        mock_provider.generate_text.return_value = valid_response

        state = PipelineState(learned_rules="Rule A")
        topic = Topic("001", "unused", "Title Proposal", "niche", "angle", "guardrail")
        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "cluster", "hook")

        script = generate_script(state, topic, variation, Path("."), retries=3)

        self.assertEqual(script.title, "Title 1")
        self.assertEqual(script.hook, "Hook 1")
        self.assertEqual(len(script.scenes), 1)
        self.assertEqual(script.scenes[0].narration, "Line 1")
        self.assertFalse(script.metadata["fallback_flag"])
        self.assertEqual(script.metadata["provider"], "MagicMock")
        self.assertEqual(script.metadata["actual_model"], "claude-test")
        mock_provider.generate_text.assert_called_once()

    @patch("clippilot.brain.script_generator.get_provider")
    def test_generate_script_retry_success(self, mock_get_provider: unittest.mock.MagicMock) -> None:
        mock_provider = unittest.mock.MagicMock()
        mock_provider.model = "claude-test"
        mock_get_provider.return_value = mock_provider

        # First call: garbage response (JSON parse fails)
        # Second call: valid JSON
        valid_response = (
            "{\n"
            "  \"title\": \"Title 1\",\n"
            "  \"hook\": \"Hook 1\",\n"
            "  \"niche_context\": \"Context 1\",\n"
            "  \"scenes\": [\n"
            "    {\"narration\": \"Line 1\", \"visual_desc\": \"View 1\"}\n"
            "  ]\n"
            "}"
        )
        mock_provider.generate_text.side_effect = [RuntimeError("API error"), valid_response]

        state = PipelineState(learned_rules="Rule A")
        topic = Topic("001", "unused", "Title Proposal", "niche", "angle", "guardrail")
        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "cluster", "hook")

        script = generate_script(state, topic, variation, Path("."), retries=3)

        self.assertEqual(script.title, "Title 1")
        self.assertEqual(mock_provider.generate_text.call_count, 2)
        self.assertFalse(script.metadata["fallback_flag"])
        self.assertEqual(script.metadata["retries"], 2)

    @patch("clippilot.brain.script_generator.get_provider")
    def test_generate_script_exhausted_retries(self, mock_get_provider: unittest.mock.MagicMock) -> None:
        mock_provider = unittest.mock.MagicMock()
        mock_get_provider.return_value = mock_provider

        # Fails persistently on all attempts with API error
        mock_provider.generate_text.side_effect = RuntimeError("API error")

        state = PipelineState(learned_rules="Rule A")
        topic = Topic("001", "unused", "Title Proposal", "niche", "angle", "guardrail")
        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "cluster", "hook")

        with self.assertRaises(ScriptValidationError):
            generate_script(state, topic, variation, Path("."), retries=3, fallback_to_mock=False)

        self.assertEqual(mock_provider.generate_text.call_count, 3)

    @patch("clippilot.brain.script_generator.get_provider")
    def test_generate_script_provider_failure_fallback(self, mock_get_provider: unittest.mock.MagicMock) -> None:
        mock_provider = unittest.mock.MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_provider.generate_text.side_effect = RuntimeError("API error")

        state = PipelineState(learned_rules="Rule A")
        topic = Topic("001", "unused", "Title Proposal", "niche", "angle", "guardrail")
        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "cluster", "hook")

        script = generate_script(state, topic, variation, Path("."), retries=3, fallback_to_mock=True)

        self.assertEqual(script.title, "Why Title Proposal")
        self.assertTrue(script.metadata["fallback_flag"])
        self.assertEqual(script.metadata["retries"], 3)

    def test_generate_script_missing_key_fallback(self) -> None:
        import os
        # Temporarily clear all possible API keys
        for k in ["LLM_API_KEY", "ANTHROPIC_API_KEY"]:
            if k in os.environ:
                del os.environ[k]

        state = PipelineState(learned_rules="Rule A")
        topic = Topic("001", "unused", "Title Proposal", "niche", "angle", "guardrail")
        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "cluster", "hook")

        # Mock settings to make sure llm_api_key is also not configured
        with patch("clippilot.config.Settings.load") as mock_settings_load:
            mock_settings = unittest.mock.MagicMock()
            mock_settings.llm_api_key = None
            mock_settings.llm_provider = "anthropic"
            mock_settings.llm_model = "claude-test"
            mock_settings_load.return_value = mock_settings

            script = generate_script(state, topic, variation, Path("."), retries=3, fallback_to_mock=True)

        self.assertEqual(script.title, "Why Title Proposal")
        self.assertTrue(script.metadata["fallback_flag"])
        self.assertEqual(script.metadata["provider"], "MockProvider")
