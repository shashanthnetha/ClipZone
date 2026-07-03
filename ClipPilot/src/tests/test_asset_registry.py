# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Asset Registry."""
from __future__ import annotations

import unittest

from clippilot.brain.asset_registry import (
    AssetRegistry,
    BackgroundAsset,
    ChartAsset,
    FontAsset,
    TransitionAsset,
    AnimationAsset,
    AudioAsset,
    IconAsset,
    ImageAsset,
)


class TestAssetRegistry(unittest.TestCase):
    """Tests for registering and resolving visual, motion, and audio assets with fallsback defaults."""

    def test_default_assets_loaded(self) -> None:
        registry = AssetRegistry()

        # Check default backgrounds loaded
        bg = registry.resolve("dark_radial", "backgrounds")
        self.assertIsInstance(bg, BackgroundAsset)
        self.assertEqual(bg.bg_type, "dark_radial")

        # Check default charts loaded
        chart = registry.resolve("ScoreDial", "charts")
        self.assertIsInstance(chart, ChartAsset)
        self.assertEqual(chart.chart_type, "ScoreDial")

        # Check default fonts loaded
        font = registry.resolve("PP Neue Montreal", "fonts")
        self.assertIsInstance(font, FontAsset)
        self.assertEqual(font.font_family, "PP Neue Montreal")

    def test_custom_registration(self) -> None:
        registry = AssetRegistry()
        custom_icon = IconAsset("my_custom_star", "icons", "star", "path/to/svg")
        registry.register(custom_icon)

        resolved = registry.resolve("my_custom_star", "icons")
        self.assertEqual(resolved, custom_icon)

    def test_category_validation_mismatch(self) -> None:
        registry = AssetRegistry()
        # "Syne" is registered under "fonts"
        with self.assertRaises(ValueError):
            registry.resolve("Syne", "charts")

    def test_fallback_creation(self) -> None:
        registry = AssetRegistry()
        # Resolve non-existent IDs under categories to test fallback generation
        bg = registry.resolve("non_existent_bg", "backgrounds")
        self.assertIsInstance(bg, BackgroundAsset)
        self.assertEqual(bg.asset_id, "non_existent_bg")

        chart = registry.resolve("non_existent_chart", "charts")
        self.assertIsInstance(chart, ChartAsset)

        anim = registry.resolve("non_existent_anim", "animations")
        self.assertIsInstance(anim, AnimationAsset)

        trans = registry.resolve("non_existent_trans", "transitions")
        self.assertIsInstance(trans, TransitionAsset)

        audio = registry.resolve("non_existent_audio", "audio")
        self.assertIsInstance(audio, AudioAsset)
