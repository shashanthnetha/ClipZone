# -*- coding: utf-8 -*-
"""Unit tests for the ClipPilot Execution Engine."""
from __future__ import annotations

import unittest

from clippilot.brain.execution_engine import (
    ExecutionContext,
    ExecutionEngine,
    ExecutionResult,
    ExecutionStage,
    ExecutionStatus,
)


class TestExecutionEngine(unittest.TestCase):
    """Tests for sequential pipeline run orchestration, retry rules, and cancellations."""

    def test_pipeline_sequential_success(self) -> None:
        engine = ExecutionEngine()
        context = ExecutionContext("run_1")

        order = []
        # Register simple success runners
        engine.register_runner(ExecutionStage.VOICE, lambda ctx: order.append("VOICE"), retries=3)
        engine.register_runner(ExecutionStage.RENDER, lambda ctx: order.append("RENDER"), retries=2)
        engine.register_runner(ExecutionStage.QA, lambda ctx: order.append("QA"), retries=1)
        engine.register_runner(ExecutionStage.PUBLISH, lambda ctx: order.append("PUBLISH"), retries=3)

        results = engine.execute(context)

        # Check order is VOICE -> RENDER -> QA -> PUBLISH
        self.assertEqual(order, ["VOICE", "RENDER", "QA", "PUBLISH"])

        # All stages should be COMPLETED
        for res in results:
            self.assertEqual(res.status, ExecutionStatus.COMPLETED)
            self.assertEqual(res.attempts, 1)

    def test_pipeline_failure_halts_propagation(self) -> None:
        engine = ExecutionEngine()
        context = ExecutionContext("run_2")

        order = []
        # VOICE succeeds, RENDER fails persistently
        engine.register_runner(ExecutionStage.VOICE, lambda ctx: order.append("VOICE"), retries=1)

        def render_fail(ctx: ExecutionContext) -> None:
            order.append("RENDER")
            raise ValueError("GPU out of memory")

        engine.register_runner(ExecutionStage.RENDER, render_fail, retries=2)
        engine.register_runner(ExecutionStage.QA, lambda ctx: order.append("QA"), retries=1)

        results = engine.execute(context)

        # Verify that QA was NEVER executed (halted after RENDER failed)
        self.assertEqual(order, ["VOICE", "RENDER", "RENDER"])  # 2 attempts at RENDER
        self.assertEqual(len(results), 2)  # Results only for VOICE and RENDER

        self.assertEqual(results[0].status, ExecutionStatus.COMPLETED)
        self.assertEqual(results[1].status, ExecutionStatus.FAILED)
        self.assertEqual(results[1].attempts, 2)
        self.assertEqual(results[1].error, "GPU out of memory")

    def test_pipeline_retry_recovery_success(self) -> None:
        engine = ExecutionEngine()
        context = ExecutionContext("run_3")

        calls = 0

        def render_flaky(ctx: ExecutionContext) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("Transient drive locking")
            # Succeeds on attempt 2

        engine.register_runner(ExecutionStage.VOICE, lambda ctx: None, retries=1)
        engine.register_runner(ExecutionStage.RENDER, render_flaky, retries=3)
        engine.register_runner(ExecutionStage.QA, lambda ctx: None, retries=1)

        results = engine.execute(context)

        self.assertEqual(results[1].status, ExecutionStatus.COMPLETED)
        self.assertEqual(results[1].attempts, 2)  # Succeeded on 2nd attempt

    def test_pipeline_cancellation(self) -> None:
        engine = ExecutionEngine()
        context = ExecutionContext("run_4")

        # Flaky runner that sets cancellation token
        def voice_runner(ctx: ExecutionContext) -> None:
            ctx.cancelled = True  # Cancel pipeline after voice

        engine.register_runner(ExecutionStage.VOICE, voice_runner, retries=1)
        engine.register_runner(ExecutionStage.RENDER, lambda ctx: None, retries=1)

        results = engine.execute(context)

        # VOICE should succeed
        self.assertEqual(results[0].status, ExecutionStatus.COMPLETED)
        # RENDER, QA, PUBLISH should be cancelled/skipped
        self.assertEqual(results[1].status, ExecutionStatus.CANCELLED)
        self.assertEqual(results[1].attempts, 0)
