# -*- coding: utf-8 -*-
"""ClipPilot Brain Voice Provider Abstraction.

Defines the common VoiceProvider Protocol, request/result dataclasses,
and implements the EdgeTTSProvider backend with word-level alignment support.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class VoiceAlignment:
    """Represents word-level boundary timing alignment."""
    word: str
    start_seconds: float
    end_seconds: float


@dataclass
class VoiceRequest:
    """Represents a text-to-speech rendering request."""
    text: str
    voice_id: str
    pitch: str = "+0Hz"
    rate: str = "+0%"
    output_path: Optional[str] = None


@dataclass
class VoiceResult:
    """Represents the rendered audio track and timing alignment results."""
    audio_path: str
    duration_seconds: float
    alignments: list[VoiceAlignment] = field(default_factory=list)
    sentences_timings: list[tuple[str, float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class VoiceProvider(Protocol):
    """Protocol defining the standard interface for all TTS provider backends."""

    def generate_voice(self, request: VoiceRequest) -> VoiceResult:
        """Render text to audio and return timing alignments."""
        ...

    def validate(self) -> None:
        """Validate credentials, packages, or API status on startup."""
        ...

    def supports_alignment(self) -> bool:
        """Returns True if the provider supports word-level alignments."""
        ...


class EdgeTTSProvider:
    """TTS provider backend utilizing the Microsoft Edge Translation Service."""

    def __init__(self, voice_default: str = "en-US-GuyNeural") -> None:
        self.voice_default = voice_default

    def validate(self) -> None:
        """Assert edge-tts is importable."""
        try:
            import edge_tts
        except ImportError:
            raise ImportError(
                "edge-tts package is not installed. Please run 'pip install edge-tts'."
            )

    def supports_alignment(self) -> bool:
        return True

    def generate_voice(self, request: VoiceRequest) -> VoiceResult:
        """Synchronously coordinate EdgeTTS generation using asyncio loops."""
        self.validate()
        import edge_tts

        output_path = request.output_path or "temp_voice.mp3"
        voice = request.voice_id or self.voice_default

        async def _generate() -> tuple[list[VoiceAlignment], float]:
            communicate = edge_tts.Communicate(
                request.text,
                voice,
                rate=request.rate,
                pitch=request.pitch,
                boundary="WordBoundary",
            )
            alignments: list[VoiceAlignment] = []
            duration = 0.0

            # Consume the stream exactly once, writing audio data and capturing WordBoundary events
            with open(output_path, "wb") as f:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        # chunk carries: text, offset (100ns units), duration (100ns units)
                        # Convert 100ns units (ticks) to seconds: ticks / 10,000,000
                        offset = chunk["offset"] / 10000000.0
                        word_duration = chunk["duration"] / 10000000.0
                        end = offset + word_duration
                        alignments.append(
                            VoiceAlignment(
                                word=chunk["text"],
                                start_seconds=round(offset, 3),
                                end_seconds=round(end, 3),
                            )
                        )
                        duration = max(duration, end)

            return alignments, round(duration, 3)

        # Run async communciate save loop inside sync boundary
        alignments, duration = asyncio.run(_generate())

        # Group words into sentences timings heuristically
        sentences_timings = []
        if alignments:
            current_sentence = []
            sentence_start = alignments[0].start_seconds
            for word_align in alignments:
                current_sentence.append(word_align.word)
                if word_align.word.endswith((".", "!", "?")):
                    sentence_text = " ".join(current_sentence)
                    sentences_timings.append(
                        (sentence_text, sentence_start, word_align.end_seconds)
                    )
                    current_sentence = []
                    # Next sentence starts at next word or bounds
                    sentence_start = word_align.end_seconds

            # Append remainder
            if current_sentence:
                sentence_text = " ".join(current_sentence)
                sentences_timings.append(
                    (sentence_text, sentence_start, alignments[-1].end_seconds)
                )

        return VoiceResult(
            audio_path=output_path,
            duration_seconds=duration,
            alignments=alignments,
            sentences_timings=sentences_timings,
            metadata={"voice_id": voice, "provider": "edge-tts"},
        )


def get_voice_provider(provider_name: str = "edge-tts") -> VoiceProvider:
    """Factory to retrieve the active, configured TTS VoiceProvider."""
    if provider_name.lower() == "edge-tts":
        return EdgeTTSProvider()

    # Pre-configured error placeholders for other voice providers
    supported = ["elevenlabs", "openai", "cartesia", "kokoro", "chatterbox"]
    if provider_name.lower() in supported:
        raise NotImplementedError(
            f"Voice provider '{provider_name}' is designed but not yet implemented."
        )

    raise ValueError(f"Unknown voice provider '{provider_name}'.")
