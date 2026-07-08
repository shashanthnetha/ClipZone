# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import json

from clippilot.brain.provider import tolerant_json_loads, OpenAIProvider, ProviderUsage
from clippilot.brain.script_generator import generate_script, ScriptValidationError
from clippilot.config import Settings
from clippilot.brain.pipeline_orchestrator import PipelineState


class TestJSONRobustness(unittest.TestCase):
    """Unit tests for ClipPilot LLM JSON robustness and provider fallback isolation."""

    def test_json_wrapped_in_markdown_fences(self):
        text = '```json\n{\n  "title": "BNPL Report",\n  "hook": "Check this!",\n  "niche_context": "finance",\n  "scenes": []\n}\n```'
        parsed = tolerant_json_loads(text)
        self.assertEqual(parsed["title"], "BNPL Report")
        self.assertEqual(parsed["hook"], "Check this!")

    def test_leading_explanations_before_json(self):
        text = 'Here is the response:\n{\n  "status": "success",\n  "code": 200\n}'
        parsed = tolerant_json_loads(text)
        self.assertEqual(parsed["status"], "success")
        self.assertEqual(parsed["code"], 200)

    def test_trailing_explanations_after_json(self):
        text = '{\n  "status": "success"\n}\nHope this helps!'
        parsed = tolerant_json_loads(text)
        self.assertEqual(parsed["status"], "success")

    def test_trailing_commas_in_objects(self):
        text = '{\n  "a": 1,\n  "b": "two",\n}'
        parsed = tolerant_json_loads(text)
        self.assertEqual(parsed["a"], 1)
        self.assertEqual(parsed["b"], "two")

    def test_trailing_commas_in_arrays(self):
        text = '{\n  "list": [1, 2, 3,],\n}'
        parsed = tolerant_json_loads(text)
        self.assertEqual(parsed["list"], [1, 2, 3])

    def test_multiple_json_objects_selection(self):
        # First block is invalid JSON, second block is valid JSON
        text = 'First incomplete block: {\n  "a": 1\nAnd here is a valid one: {\n  "title": "Valid",\n  "score": 10\n}'
        parsed = tolerant_json_loads(text)
        self.assertEqual(parsed["title"], "Valid")
        self.assertEqual(parsed["score"], 10)

    def test_completely_invalid_json_raises(self):
        text = 'This is just some text with no valid json block.'
        with self.assertRaises(Exception):
            tolerant_json_loads(text)

    @patch("clippilot.brain.provider.execute_with_retries")
    def test_provider_success_repaired_json_does_not_switch_models_or_retry(self, mock_execute):
        # Mock API response returning malformed JSON with leading text and trailing commas
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Here is the plan:\n{\n  "scenes": [\n    {\n      "scene_number": 1,\n      "background": "dark",\n    }\n  ],\n}'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        
        mock_execute.return_value = mock_response

        # Instantiate provider with multiple candidate models to test fallback loop isolation
        provider = OpenAIProvider(model=["model-a", "model-b"], api_key="test-key")
        
        # Call generate_vision
        parsed = provider.generate_vision(
            messages=[],
            json_schema={},
            system_prompt=None
        )

        # Assert that parsing succeeded after repairs
        self.assertEqual(len(parsed["scenes"]), 1)
        self.assertEqual(parsed["scenes"][0]["background"], "dark")

        # Ensure we only called the API once (no retries) and stayed on the first model ("model-a")
        self.assertEqual(mock_execute.call_count, 1)
        self.assertEqual(provider.model, "model-a")
        self.assertEqual(provider.last_usage["actual_model"], "model-a")

    @patch("clippilot.brain.provider.execute_with_retries")
    def test_provider_success_unrecoverable_json_propagates_immediately(self, mock_execute):
        # Mock API response returning completely unrecoverable junk text
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is completely unrecoverable junk."
        mock_response.usage = MagicMock()
        
        mock_execute.return_value = mock_response

        provider = OpenAIProvider(model=["model-a", "model-b"], api_key="test-key")

        # generate_vision should raise Exception immediately instead of advancing to "model-b"
        with self.assertRaises(Exception):
            provider.generate_vision(
                messages=[],
                json_schema={},
                system_prompt=None
            )

        # Confirm we only called the API once and did NOT fallback/switch to model-b
        self.assertEqual(mock_execute.call_count, 1)
        self.assertEqual(provider.model, "model-a")

    def test_safe_extract_response_content_formats(self):
        from clippilot.brain.provider import safe_extract_response_content

        # 1. Standard OpenAI response object
        resp_obj = MagicMock()
        choice = MagicMock()
        msg = MagicMock()
        msg.content = "Standard response text"
        choice.message = msg
        resp_obj.choices = [choice]
        self.assertEqual(safe_extract_response_content(resp_obj), "Standard response text")

        # 2. Legacy choices[0].text format
        resp_legacy = MagicMock()
        choice_legacy = MagicMock()
        choice_legacy.message = None
        choice_legacy.text = "Legacy text"
        resp_legacy.choices = [choice_legacy]
        self.assertEqual(safe_extract_response_content(resp_legacy), "Legacy text")

        # 3. Multimodal list of blocks
        resp_multimodal = MagicMock()
        choice_mm = MagicMock()
        msg_mm = MagicMock()
        msg_mm.content = [{"type": "text", "text": "Hello "}, {"type": "text", "text": "World!"}]
        choice_mm.message = msg_mm
        resp_multimodal.choices = [choice_mm]
        self.assertEqual(safe_extract_response_content(resp_multimodal), "Hello World!")

        # 4. Dictionary structure
        resp_dict = {
            "choices": [
                {
                    "message": {
                        "content": "Dict text"
                    }
                }
            ]
        }
        self.assertEqual(safe_extract_response_content(resp_dict), "Dict text")

    def test_failed_response_saving(self):
        import os
        from pathlib import Path
        from clippilot.brain.provider import save_failed_response

        test_content = "This is raw failed response content."
        save_failed_response("test_stage", test_content, "test reason")

        # Verify file creation inside logs/failed_llm_responses/
        logs_dir = Path("logs/failed_llm_responses")
        self.assertTrue(logs_dir.exists())

        files = list(logs_dir.glob("test_stage_*.txt"))
        self.assertGreater(len(files), 0)

        # Verify file content
        newest_file = max(files, key=os.path.getctime)
        content = newest_file.read_text(encoding="utf-8")
        self.assertIn("Reason: test reason", content)
        self.assertIn(test_content, content)

        # Cleanup the test file
        newest_file.unlink()

    @patch("clippilot.brain.env.has_api_key")
    @patch("clippilot.brain.script_generator.get_provider")
    def test_metadata_preservation_on_parsing_failure(self, mock_get_provider, mock_has_key):
        mock_has_key.return_value = True
        from clippilot.brain.pipeline_orchestrator import Topic, VariationRecord
        from pathlib import Path
        # Setup mock provider that succeeds at the API level (returns text) but returns invalid JSON
        mock_provider = MagicMock()
        mock_provider.model = "nemotron-test"
        mock_provider.requested_model = "nemotron-test"
        mock_provider.generate_text.return_value = "This is NOT valid JSON."
        mock_provider.last_usage = {
            "input_tokens": 123,
            "output_tokens": 45,
            "latency": 0.987,
            "estimated_cost": 0.00012,
            "cost": 0.00012
        }
        mock_get_provider.return_value = mock_provider

        state = PipelineState(learned_rules="Rule A")
        topic = Topic("001", "unused", "Title Proposal", "niche", "angle", "guardrail")
        variation = VariationRecord("2026-07-03", "slug", "T01", "F1", "S1", "V1", "L3", "0%", "cluster", "hook")

        script = generate_script(state, topic, variation, Path("."), retries=1, fallback_to_mock=True)

        # Verify that mock script has the REAL provider metadata preserved!
        self.assertEqual(script.metadata["provider"], "MagicMock")
        self.assertEqual(script.metadata["actual_model"], "nemotron-test")
        self.assertEqual(script.metadata["input_tokens"], 123)
        self.assertEqual(script.metadata["output_tokens"], 45)
        self.assertEqual(script.metadata["estimated_cost"], 0.00012)
        self.assertTrue(script.metadata["fallback_flag"])

    def test_detects_reasoning_leakage(self):
        from clippilot.brain.provider import detects_reasoning_leakage

        # Pure JSON
        self.assertFalse(detects_reasoning_leakage('{"title": "Valid JSON"}'))
        self.assertFalse(detects_reasoning_leakage('[{"scenes": []}]'))
        # Markdown Code Block JSON
        self.assertFalse(detects_reasoning_leakage('```json\n{"title": "Valid Markdown"}\n```'))
        
        # Reasoning Leakage
        self.assertTrue(detects_reasoning_leakage("Let me think... Here is the JSON: {\"title\": \"Prose\"}"))
        self.assertTrue(detects_reasoning_leakage("The user wants me to do this. {\"title\": \"Prose\"}"))
        self.assertTrue(detects_reasoning_leakage("Sure! Here's the JSON payload: {\"title\": \"Prose\"}"))

    @patch("clippilot.config.Settings")
    @patch("clippilot.brain.provider.execute_with_retries")
    def test_correction_logic_succeeds(self, mock_execute, mock_settings_cls):
        # Configure settings to enable correction
        mock_settings = MagicMock()
        mock_settings.enable_reasoning_correction = True
        mock_settings_cls.load.return_value = mock_settings

        from clippilot.brain.provider import OpenAIProvider
        provider = OpenAIProvider(model="gpt-4o", api_key="dummy-key")

        # Mock first response with reasoning leakage and invalid JSON (no closing brace)
        first_resp = MagicMock()
        first_resp.choices = [MagicMock()]
        first_resp.choices[0].message.content = "Let me think... Here is the JSON: {\"title\": \"Leakage\""
        first_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        second_resp = MagicMock()
        second_resp.choices = [MagicMock()]
        second_resp.choices[0].message.content = '{"title": "Corrected JSON"}'
        second_resp.usage = MagicMock(prompt_tokens=15, completion_tokens=25)

        mock_execute.side_effect = [first_resp, second_resp]

        result = provider.generate_text(
            prompt="Write a script",
            system_prompt=None,
            json_schema={"type": "object"}
        )

        self.assertEqual(result, '{"title": "Corrected JSON"}')
        # Total execute calls = 2 (first call + 1 correction retry)
        self.assertEqual(mock_execute.call_count, 2)
        # Usage must be accumulated
        self.assertEqual(provider.last_usage["input_tokens"], 25) # 10 + 15
        self.assertEqual(provider.last_usage["output_tokens"], 45) # 20 + 25

    @patch("clippilot.config.Settings")
    @patch("clippilot.brain.provider.execute_with_retries")
    def test_correction_logic_disabled_by_default(self, mock_execute, mock_settings_cls):
        # Configure settings to disable correction
        mock_settings = MagicMock()
        mock_settings.enable_reasoning_correction = False
        mock_settings_cls.load.return_value = mock_settings

        from clippilot.brain.provider import OpenAIProvider
        provider = OpenAIProvider(model="gpt-4o", api_key="dummy-key")

        # Mock first response with reasoning leakage
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "Let me think... Here is the JSON: {\"title\": \"Leakage\"}"
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        mock_execute.return_value = resp

        result = provider.generate_text(
            prompt="Write a script",
            system_prompt=None,
            json_schema={"type": "object"}
        )

        # Should return original leaked response without sending a second call
        self.assertEqual(result, "Let me think... Here is the JSON: {\"title\": \"Leakage\"}")
        self.assertEqual(mock_execute.call_count, 1)

    def test_token_estimation_when_zero(self):
        from clippilot.brain.provider import make_provider_usage
        usage = make_provider_usage(
            provider="TestProvider",
            requested_model="dummy-model",
            actual_model="dummy-model",
            input_tokens=0,
            output_tokens=0,
            latency=1.5,
            prompt_or_messages="Hello world prompt",
            response_str="A short response",
            system_prompt="System"
        )
        self.assertTrue(usage["estimated_usage"])
        # 'Hello world prompt' (18 chars) + 'System' (6 chars) = 24 chars. 24 / 4 = 6 prompt tokens.
        # 'A short response' (16 chars) = 16 / 4 = 4 completion tokens.
        self.assertEqual(usage["input_tokens"], 6)
        self.assertEqual(usage["output_tokens"], 4)

    def test_is_vision_capable_model(self):
        from clippilot.brain.provider import is_vision_capable_model
        self.assertTrue(is_vision_capable_model("google/gemini-2.0-flash-exp:free"))
        self.assertTrue(is_vision_capable_model("qwen/qwen2.5-vl-72b-instruct:free"))
        self.assertFalse(is_vision_capable_model("meta-llama/llama-3.3-70b-instruct:free"))
        self.assertFalse(is_vision_capable_model("nvidia/nemotron-3-ultra-550b-a55b:free"))

    @patch("clippilot.brain.provider.is_vision_capable_model")
    @patch("clippilot.brain.provider.execute_with_retries")
    def test_vision_filtering_skips_text_only(self, mock_execute, mock_is_vision):
        # mock is_vision_capable_model to return False for gpt-4o and True for gemini
        mock_is_vision.side_effect = lambda m: "gemini" in m
        
        from clippilot.brain.provider import OpenAIProvider
        provider = OpenAIProvider(model=["gpt-4o", "google/gemini-flash:free"], api_key="dummy-key")
        
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"passed": true}'
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_execute.return_value = resp
        
        # This should execute on gemini-flash and skip gpt-4o
        result = provider.generate_vision(
            messages=[{"role": "user", "content": "hello"}],
            json_schema={"type": "object"}
        )
        self.assertEqual(provider.model, "google/gemini-flash:free")

    def test_save_failed_response_metadata(self):
        from clippilot.brain.provider import save_failed_response
        with patch("pathlib.Path.write_text") as mock_write:
            save_failed_response(
                stage_name="test_stage",
                raw_response="{\"leaked\": 1}",
                reason="Reasoning leakage",
                provider="OpenAIProvider",
                requested_model="gpt-4o-mini",
                actual_model="gpt-4o-mini-actual",
                correction_attempted=True,
                repair_attempted=True,
                schema_validation_status="not_applicable"
            )
            # Check write_text was called with formatted metadata
            self.assertTrue(mock_write.called)
            written_content = mock_write.call_args[0][0]
            self.assertIn("Stage: test_stage", written_content)
            self.assertIn("Reason: Reasoning leakage", written_content)
            self.assertIn("Provider: OpenAIProvider", written_content)
            self.assertIn("Requested Model: gpt-4o-mini", written_content)
            self.assertIn("Actual Model: gpt-4o-mini-actual", written_content)
            self.assertIn("Correction Attempted: True", written_content)
            self.assertIn("Repair Attempted: True", written_content)
            self.assertIn("Schema Validation Status: not_applicable", written_content)
            self.assertIn("Raw Response:\n{\"leaked\": 1}", written_content)

    def test_circuit_breaker(self):
        from clippilot.brain.provider import mark_model_unavailable, is_model_available, _model_blocklist
        import time
        
        # Test model availability transition
        model_name = "test-rate-limited-model"
        self.assertTrue(is_model_available(model_name))
        
        mark_model_unavailable(model_name)
        self.assertFalse(is_model_available(model_name))
        
        # Clean up blocklist for tests
        if model_name in _model_blocklist:
            del _model_blocklist[model_name]

    def test_schema_validation_in_provider(self):
        from clippilot.brain.provider import validate_schema
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "number": {"type": "integer"},
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["title", "number"]
        }
        
        # Valid data
        valid_data = {"title": "Test Title", "number": 42, "items": ["a", "b"]}
        errors = validate_schema(valid_data, schema)
        self.assertEqual(len(errors), 0)
        
        # Invalid data: missing required key, wrong type
        invalid_data = {"number": "not-an-integer", "items": 123}
        errors = validate_schema(invalid_data, schema)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Missing required key: title" in e for e in errors))
        self.assertTrue(any("number must be an integer" in e for e in errors))
        self.assertTrue(any("items must be a list/array" in e for e in errors))

    def test_lightweight_diagnostics(self):
        from clippilot.brain.provider import pipeline_diagnostics
        
        pipeline_diagnostics["attempts"].clear()
        pipeline_diagnostics["attempts"].append({"model": "test-model", "status": "Success", "latency": 1.23})
        pipeline_diagnostics["json_repairs"] += 1
        
        self.assertEqual(len(pipeline_diagnostics["attempts"]), 1)
        self.assertEqual(pipeline_diagnostics["json_repairs"], 1)


if __name__ == "__main__":
    unittest.main()
