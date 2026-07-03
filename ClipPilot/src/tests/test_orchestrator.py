# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Python Orchestrator Phase 1."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clippilot.brain.pipeline_orchestrator import (
    PipelineState,
    PostRecord,
    RunContext,
    Topic,
    VariationRecord,
    choose_topic,
    choose_variation,
    load_state,
    update_ledgers,
)
from clippilot.brain.parsers import parse_markdown_table, update_topic_status


class TestOrchestrator(unittest.TestCase):
    """Tests for the markdown parsers and state/solver loop functions."""

    def test_parse_markdown_table(self) -> None:
        table_content = (
            "| Header A | Header B |\n"
            "|---|---|\n"
            "| Row 1 A | Row 1 B |\n"
            "| Row 2 A | Row 2 B |\n"
        )
        parsed = parse_markdown_table(table_content)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["header a"], "Row 1 A")
        self.assertEqual(parsed[1]["header b"], "Row 2 B")

    def test_update_topic_status(self) -> None:
        content = (
            "| # | Status | Title |\n"
            "|---|---|---|\n"
            "| 001 | unused | Topic Title 1 |\n"
            "| 002 | unused | Topic Title 2 |\n"
        )
        updated = update_topic_status(content, "001", "USED 2026-07-03")
        self.assertIn("| 001 | USED 2026-07-03 | Topic Title 1 |", updated)
        self.assertIn("| 002 | unused | Topic Title 2 |", updated)

    def test_choose_topic(self) -> None:
        # Case A: Has unused topic
        state = PipelineState(
            topics=[
                Topic("001", "USED 2026-06-29", "Topic A", "niche", "angle", "guardrail"),
                Topic("002", "unused", "Topic B", "niche", "angle", "guardrail"),
            ]
        )
        picked = choose_topic(state)
        self.assertIsNotNone(picked)
        self.assertEqual(picked.num, "002")

        # Case B: No unused topics
        state_empty = PipelineState(
            topics=[
                Topic("001", "USED 2026-06-29", "Topic A", "niche", "angle", "guardrail"),
            ]
        )
        picked_none = choose_topic(state_empty)
        self.assertIsNone(picked_none)

    def test_choose_variation_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            v_dir = Path(tmp_dir)
            # Create catalogs
            (v_dir / "title_shapes.md").write_text("| id | Type | Shape |\n|---|---|---|\n| T01 | type | S |\n| T02 | type | S |\n| T03 | type | S |\n| T04 | type | S |\n| T05 | type | S |\n| T06 | type | S |\n| T07 | type | S |\n")
            (v_dir / "formats.md").write_text("| id | Format | Structure |\n|---|---|---|\n| F1 | explainer | S |\n| F2 | explainer | S |\n| F3 | explainer | S |\n| F4 | explainer | S |\n| F5 | explainer | S |\n| F6 | explainer | S |\n")
            (v_dir / "visual_skins.md").write_text("| id | Skin | Background |\n|---|---|---|\n| S1 | skin | B |\n| S2 | skin | B |\n| S3 | skin | B |\n| S4 | skin | B |\n| S5 | skin | B |\n")
            (v_dir / "voices.md").write_text("| id | Voice | Character |\n|---|---|---|\n| V1 | voice | C |\n| V2 | voice | C |\n| V3 | voice | C |\n| V4 | voice | C |\n| V5 | voice | C |\n")

            state = PipelineState(
                variation_history=[
                    # Log 6 past records of T01, F1, S1, V1
                    VariationRecord("2026-06-29", "daily001", "T01", "F1", "S1", "V1", "L3", "0%", "nicheA", "hookA"),
                    VariationRecord("2026-06-29", "daily002", "T01", "F1", "S1", "V1", "L3", "0%", "nicheA", "hookA"),
                    VariationRecord("2026-06-29", "daily003", "T01", "F1", "S1", "V1", "L3", "0%", "nicheA", "hookA"),
                    VariationRecord("2026-06-30", "daily004", "T01", "F1", "S1", "V1", "L3", "0%", "nicheA", "hookA"),
                    VariationRecord("2026-07-01", "daily005", "T01", "F1", "S1", "V1", "L3", "0%", "nicheA", "hookA"),
                    VariationRecord("2026-07-01", "daily006", "T01", "F1", "S1", "V1", "L3", "0%", "nicheA", "hookA"),
                ]
            )

            # Diversity + Rotation check: solver must NOT pick T01, F1, S1, V1
            combo = choose_variation(state, v_dir, "nicheB", seed=42)
            self.assertNotEqual(combo.title, "T01")
            self.assertNotEqual(combo.fmt, "F1")
            self.assertNotEqual(combo.skin, "S1")
            self.assertNotEqual(combo.voice, "V1")
            self.assertEqual(combo.cluster, "nicheB")

    def test_choose_variation_impossible_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            v_dir = Path(tmp_dir)
            # Create catalogs with only ONE option
            (v_dir / "title_shapes.md").write_text("| id |\n|---|\n| T01 |\n")
            (v_dir / "formats.md").write_text("| id |\n|---|\n| F1 |\n")
            (v_dir / "visual_skins.md").write_text("| id |\n|---|\n| S1 |\n")
            (v_dir / "voices.md").write_text("| id |\n|---|\n| V1 |\n")

            state = PipelineState(
                variation_history=[
                    VariationRecord("2026-07-01", "daily006", "T01", "F1", "S1", "V1", "L3", "0%", "nicheA", "hookA"),
                ]
            )

            # Should raise ValueError because the single T01/F1/S1/V1 is locked by rotation rule
            with self.assertRaises(ValueError):
                choose_variation(state, v_dir, "nicheB")

    def test_update_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_dir = Path(tmp_dir)
            topics_content = (
                "| # | Status | Title | Niche / CPM | Angle (the true mechanism) | Brand-safety guardrail |\n"
                "|---|---|---|---|---|---|\n"
                "| 006 | unused | Title 6 | credit | angle 6 | guardrail 6 |\n"
            )
            posts_content = (
                "| Date | Slug | Title | Topic # | Scheduled (local) | Final mp4 | GATE |\n"
                "|---|---|---|---|---|---|---|\n"
            )
            variation_content = (
                "| Date | Slug | title | fmt | skin | voice | len | pace | cluster | hook |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n"
            )

            (workspace_dir / "daily_topics.md").write_text(topics_content, encoding="utf-8")
            (workspace_dir / "daily_posts_ledger.md").write_text(posts_content, encoding="utf-8")
            (workspace_dir / "variation_ledger.md").write_text(variation_content, encoding="utf-8")

            context = RunContext(
                workspace_dir=workspace_dir,
                variation_dir=workspace_dir,
                date_str="2026-07-03",
                slug="daily008_testing",
            )
            state = PipelineState()
            selected_topic = Topic("006", "unused", "Title 6", "credit", "angle 6", "guardrail 6")
            selected_variation = VariationRecord(
                date="", slug="", title="T02", fmt="F2", skin="S2", voice="V2", len="L2", pace="0%", cluster="credit", hook="question"
            )

            update_ledgers(context, state, selected_topic, selected_variation)

            # Check status update
            topics_after = (workspace_dir / "daily_topics.md").read_text(encoding="utf-8")
            self.assertIn("USED 2026-07-03", topics_after)

            # Check posts append
            posts_after = (workspace_dir / "daily_posts_ledger.md").read_text(encoding="utf-8")
            self.assertIn("daily008_testing", posts_after)
            self.assertIn("Title 6", posts_after)

            # Check variation append
            variation_after = (workspace_dir / "variation_ledger.md").read_text(encoding="utf-8")
            self.assertIn("T02", variation_after)
            self.assertIn("F2", variation_after)
            self.assertIn("S2", variation_after)
