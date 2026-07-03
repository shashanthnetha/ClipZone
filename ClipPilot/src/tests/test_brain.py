"""Phase 1 tests for the Claude brain — request building, enrichment via the
mock client, guardrail wiring, and the cost estimator. No `anthropic`, no key."""
from __future__ import annotations

import os
import tempfile
import unittest

from clippilot.brain import build_vision_request, enrich_understanding
from clippilot.brain import client as brain_client
from clippilot.brain import env as brain_env
from clippilot.brain.client import MockVisionClient, estimate_cost_usd
from clippilot.brain.prompt import ENRICHMENT_SCHEMA
from clippilot.understanding import Scene, SourceMeta, Understanding


def _understanding_with_frames(tmp: str, n: int = 3) -> Understanding:
    scenes, paths = [], []
    for i in range(n):
        p = os.path.join(tmp, f"kf{i}.jpg")
        with open(p, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0fakejpeg" + bytes([i]))  # tiny dummy bytes
        scenes.append(Scene(idx=i, start=float(i * 3), end=float(i * 3 + 3), keyframe_path=p,
                            transcript_excerpt=f"line {i}"))
        paths.append(p)
    u = Understanding(
        source=SourceMeta(duration_s=float(n * 3), fps=30.0, resolution="320x240", has_audio=True),
        scenes=scenes, keyframe_paths=paths,
        mood_energy_timeline=[{"t": 0.0, "integrated_lufs": -16.0, "label": ""}],
    )
    return u


class TestBuildRequest(unittest.TestCase):
    def test_request_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            u = _understanding_with_frames(tmp, 3)
            req = build_vision_request(u, u.keyframe_paths, model="claude-opus-4-8")
            self.assertEqual(req["model"], "claude-opus-4-8")
            self.assertEqual(req["output_config"]["format"]["type"], "json_schema")
            self.assertIs(req["output_config"]["format"]["schema"], ENRICHMENT_SCHEMA)
            self.assertIn("video-understanding", req["system"])
            blocks = req["messages"][0]["content"]
            images = [b for b in blocks if b.get("type") == "image"]
            self.assertEqual(len(images), 3)
            self.assertTrue(all(b["source"]["media_type"] == "image/jpeg" for b in images))
            self.assertTrue(all(b["source"]["type"] == "base64" for b in images))

    def test_image_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            u = _understanding_with_frames(tmp, 5)
            req = build_vision_request(u, u.keyframe_paths, max_images=2)
            images = [b for b in req["messages"][0]["content"] if b.get("type") == "image"]
            self.assertEqual(len(images), 2)


class TestEnrichment(unittest.TestCase):
    def test_enrich_fills_semantic_fields_and_likeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            u = _understanding_with_frames(tmp, 3)
            client = MockVisionClient(identifiable_person=True)
            out = enrich_understanding(u, client=client)
            self.assertIn("mock", out.summary)
            self.assertEqual(out.mood_energy_timeline[0]["label"], "energetic")
            self.assertTrue(all(s.visual_desc for s in out.scenes))  # per-scene desc filled
            self.assertEqual(out.on_screen_text[0]["text"], "MOCK CAPTION")
            self.assertTrue(out.highlight_candidates)
            self.assertTrue(out.highlight_candidates[0].reasons)
            # likeness guardrail fired
            self.assertTrue(out.needs_likeness_review())
            self.assertIn("identifiable_person", out.flags["likeness"])
            self.assertEqual(client.calls, [3])

    def test_no_person_skips_likeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            u = _understanding_with_frames(tmp, 2)
            out = enrich_understanding(u, client=MockVisionClient(identifiable_person=False))
            self.assertFalse(out.needs_likeness_review())

    def test_no_client_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            u = _understanding_with_frames(tmp, 2)
            out = enrich_understanding(u, client=None)
            self.assertIn("brain", out.flags)


class TestFrameBudget(unittest.TestCase):
    def test_frame_budget_within_vision_image_cap(self):
        from clippilot import understand
        from clippilot.brain.prompt import MAX_IMAGES
        # never sample more keyframes than the vision request can actually send
        self.assertLessEqual(understand._MAX_VISION_FRAMES, MAX_IMAGES)


class TestClientFactoryAndCost(unittest.TestCase):
    def test_get_client_none_without_key(self):
        orig = brain_env.has_api_key
        brain_client_has = brain_client.env.has_api_key
        try:
            brain_client.env.has_api_key = lambda: False
            self.assertIsNone(brain_client.get_client())
        finally:
            brain_client.env.has_api_key = brain_client_has
            brain_env.has_api_key = orig

    def test_cost_estimate_reasonable(self):
        cost = estimate_cost_usd("claude-opus-4-8", frames=60)
        self.assertGreater(cost, 0.2)
        self.assertLess(cost, 0.5)
        # sonnet is cheaper
        self.assertLess(estimate_cost_usd("claude-sonnet-4-6", frames=60), cost)


class TestProviderAbstraction(unittest.TestCase):
    def test_get_provider_resolves_anthropic(self):
        from clippilot.brain.provider import get_provider, AnthropicProvider
        from clippilot.config import Settings
        
        # Temporarily mock env variables
        orig_provider = os.environ.get("LLM_PROVIDER")
        orig_model = os.environ.get("LLM_MODEL")
        orig_key = os.environ.get("LLM_API_KEY")
        
        os.environ["LLM_PROVIDER"] = "anthropic"
        os.environ["LLM_MODEL"] = "claude-sonnet-4-6"
        os.environ["LLM_API_KEY"] = "sk-test-key"
        
        try:
            provider = get_provider()
            self.assertIsInstance(provider, AnthropicProvider)
            self.assertEqual(provider.model, "claude-sonnet-4-6")
            self.assertEqual(provider.api_key, "sk-test-key")
            
            # Check capability flags
            self.assertTrue(provider.supports_vision())
            self.assertTrue(provider.supports_json())
            self.assertTrue(provider.supports_streaming())
        finally:
            if orig_provider is not None:
                os.environ["LLM_PROVIDER"] = orig_provider
            else:
                os.environ.pop("LLM_PROVIDER", None)
            if orig_model is not None:
                os.environ["LLM_MODEL"] = orig_model
            else:
                os.environ.pop("LLM_MODEL", None)
            if orig_key is not None:
                os.environ["LLM_API_KEY"] = orig_key
            else:
                os.environ.pop("LLM_API_KEY", None)

    def test_provider_missing_key_validation(self):
        from clippilot.brain.provider import AnthropicProvider
        
        orig_key = os.environ.pop("LLM_API_KEY", None)
        orig_ant_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        
        try:
            with self.assertRaises(ValueError):
                # Should raise ValueError because keys are missing from env and parameters
                AnthropicProvider(model="claude-opus-4-8", api_key=None)
        finally:
            if orig_key is not None:
                os.environ["LLM_API_KEY"] = orig_key
            if orig_ant_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = orig_ant_key


if __name__ == "__main__":
    unittest.main(verbosity=2)
