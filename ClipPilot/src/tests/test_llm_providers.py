# -*- coding: utf-8 -*-
import os
import unittest
from unittest.mock import MagicMock, patch

from clippilot.brain.provider import (
    get_provider,
    AnthropicProvider,
    OpenAIProvider,
    OpenRouterProvider,
    tolerant_json_loads,
    get_pricing
)
from clippilot.config import Settings
from clippilot.brain.client import get_client, ProviderVisionClient


class TestLLMProviders(unittest.TestCase):

    def setUp(self):
        self.orig_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_tolerant_json_loads(self):
        # 1. Normal JSON
        data = tolerant_json_loads('{"a": 1, "b": "hello"}')
        self.assertEqual(data["a"], 1)
        self.assertEqual(data["b"], "hello")

        # 2. Markdown fenced JSON
        data = tolerant_json_loads('Some preamble text...\n```json\n{"val": 42}\n```\nSome postamble')
        self.assertEqual(data["val"], 42)

        # 3. Code block without json identifier
        data = tolerant_json_loads('```\n{"val": 99}\n```')
        self.assertEqual(data["val"], 99)

        # 4. Curly braces within other text
        data = tolerant_json_loads('Result is {"score": 9.5} and it is good.')
        self.assertEqual(data["score"], 9.5)

    def test_get_pricing(self):
        in_p, out_p = get_pricing("claude-opus-4-8")
        self.assertEqual(in_p, 15.0)
        self.assertEqual(out_p, 75.0)

        in_p, out_p = get_pricing("gpt-4o")
        self.assertEqual(in_p, 2.50)
        self.assertEqual(out_p, 10.00)

        # Check unknown claude model
        in_p, out_p = get_pricing("claude-custom-model")
        self.assertEqual(in_p, 3.00)
        self.assertEqual(out_p, 15.00)

    @patch("clippilot.config.Settings.load")
    def test_provider_resolution(self, mock_load):
        # Mock settings to have empty model names
        mock_settings = MagicMock()
        mock_settings.llm_provider = "openrouter"
        mock_settings.llm_model = ""
        mock_settings.brain_model = ""
        mock_settings.llm_base_url = ""
        mock_settings.llm_api_key = ""
        mock_load.return_value = mock_settings

        # Test default OpenRouter provider if no environment overrides
        os.environ["OPENROUTER_API_KEY"] = "sk-or-test"
        os.environ["LLM_PROVIDER"] = "openrouter"
        os.environ.pop("LLM_MODEL", None)
        provider = get_provider(mock_settings)
        self.assertIsInstance(provider, OpenRouterProvider)
        self.assertEqual(provider.model, "openrouter/free")

        # Test OpenAI provider resolution
        mock_settings.llm_provider = "openai"
        os.environ["OPENAI_API_KEY"] = "sk-oa-test"
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_MODEL"] = "gpt-4o-mini"
        provider = get_provider(mock_settings)
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.model, "gpt-4o-mini")

        # Test Anthropic provider resolution
        mock_settings.llm_provider = "anthropic"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        os.environ["LLM_PROVIDER"] = "anthropic"
        os.environ["LLM_MODEL"] = "claude-sonnet-4-6"
        provider = get_provider(mock_settings)
        self.assertIsInstance(provider, AnthropicProvider)
        self.assertEqual(provider.model, "claude-sonnet-4-6")

    def test_capability_flags(self):
        os.environ["OPENAI_API_KEY"] = "sk-oa-test"
        provider = OpenAIProvider(model="gpt-4o")
        self.assertTrue(provider.supports_vision())
        self.assertTrue(provider.supports_json())
        self.assertTrue(provider.supports_streaming())
        self.assertTrue(provider.supports_functions())
        self.assertEqual(provider.context_size(), 128000)

    @patch("openai.resources.chat.completions.Completions.create")
    def test_openai_provider_generate_text(self, mock_create):
        # Setup mock response
        mock_choice = MagicMock()
        mock_choice.message.content = "Generated script mock content"
        
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage
        mock_create.return_value = mock_resp

        # Call generate_text
        os.environ["OPENAI_API_KEY"] = "sk-oa-test"
        provider = OpenAIProvider(model="gpt-4o")
        text = provider.generate_text("Test prompt", system_prompt="Test system")

        self.assertEqual(text, "Generated script mock content")
        self.assertEqual(provider.last_usage["input_tokens"], 100)
        self.assertEqual(provider.last_usage["output_tokens"], 50)
        # Cost check: (100 * 2.50 + 50 * 10.00) / 1,000,000 = (250 + 500)/1,000,000 = 0.00075
        self.assertEqual(provider.last_usage["cost"], 0.00075)

    @patch("openai.resources.chat.completions.Completions.create")
    def test_openai_provider_generate_vision(self, mock_create):
        # Setup mock response
        mock_choice = MagicMock()
        mock_choice.message.content = '{"score": 90, "summary": "Passed visual audit."}'
        
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 200
        mock_usage.completion_tokens = 80
        
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage
        mock_create.return_value = mock_resp

        # Call generate_vision
        os.environ["OPENAI_API_KEY"] = "sk-oa-test"
        provider = OpenAIProvider(model="gpt-4o")
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Auditing frame"},
                {"type": "image", "source": {"media_type": "image/png", "data": "b64string"}}
            ]}
        ]
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "summary": {"type": "string"}
            },
            "required": ["score", "summary"]
        }
        res = provider.generate_vision(messages=messages, json_schema=schema, system_prompt="System instructions")

        self.assertEqual(res["score"], 90)
        self.assertEqual(res["summary"], "Passed visual audit.")
        self.assertEqual(provider.last_usage["input_tokens"], 200)
        self.assertEqual(provider.last_usage["output_tokens"], 80)

    def test_provider_vision_client_resolves(self):
        # Setup OpenAI key in settings and environment
        os.environ["OPENAI_API_KEY"] = "sk-oa-test"
        settings = Settings.load()
        settings.llm_provider = "openai"
        settings.llm_api_key = "sk-oa-test"
        settings.vision_model = "gpt-4o"

        # Mock openai package presence
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            client = get_client(settings)
            self.assertIsInstance(client, ProviderVisionClient)
            self.assertEqual(client.model, "gpt-4o")
            self.assertEqual(client.provider_name, "openai")
