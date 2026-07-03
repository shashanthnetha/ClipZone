# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Voice Provider Abstraction."""
from __future__ import annotations

import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from clippilot.brain.voice_provider import (
    EdgeTTSProvider,
    VoiceAlignment,
    VoiceRequest,
    VoiceResult,
    get_voice_provider,
)


class TestVoiceProvider(unittest.TestCase):
    """Tests for TTS provider resolution, package validations, and mock stream iterations."""

    def test_get_voice_provider_resolution(self) -> None:
        provider = get_voice_provider("edge-tts")
        self.assertIsInstance(provider, EdgeTTSProvider)

        # Unimplemented providers check
        with self.assertRaises(NotImplementedError):
            get_voice_provider("elevenlabs")

        # Unknown provider check
        with self.assertRaises(ValueError):
            get_voice_provider("unknown_provider")

    @patch.dict(sys.modules, {"edge_tts": None})
    def test_validate_missing_package(self) -> None:
        provider = EdgeTTSProvider()
        # When edge_tts is None/missing, validate must raise ImportError
        with self.assertRaises(ImportError):
            provider.validate()

    @patch("clippilot.brain.voice_provider.EdgeTTSProvider.validate")
    @patch("asyncio.run")
    def test_generate_voice_mocked(
        self, mock_asyncio_run: MagicMock, mock_validate: MagicMock
    ) -> None:
        # Mock the edge_tts module dynamically in sys.modules
        mock_communicate_cls = MagicMock()
        mock_edge_tts = MagicMock()
        mock_edge_tts.Communicate = mock_communicate_cls
        
        with patch.dict(sys.modules, {"edge_tts": mock_edge_tts}):
            # Set up mock alignments list return from asyncio.run
            mock_alignments = [
                VoiceAlignment("Hello", 0.0, 0.4),
                VoiceAlignment("world.", 0.4, 0.8),
            ]
            mock_asyncio_run.return_value = (mock_alignments, 0.8)

            provider = EdgeTTSProvider()
            request = VoiceRequest("Hello world.", "en-US-GuyNeural", output_path="out.mp3")

            result = provider.generate_voice(request)

            # Check result timing matches
            self.assertEqual(result.audio_path, "out.mp3")
            self.assertEqual(result.duration_seconds, 0.8)
            self.assertEqual(len(result.alignments), 2)
            self.assertEqual(result.alignments[0].word, "Hello")

            # Verify sentence timing heuristic grouped them
            self.assertEqual(len(result.sentences_timings), 1)
            self.assertEqual(result.sentences_timings[0][0], "Hello world.")
            self.assertEqual(result.sentences_timings[0][1], 0.0)
            self.assertEqual(result.sentences_timings[0][2], 0.8)
