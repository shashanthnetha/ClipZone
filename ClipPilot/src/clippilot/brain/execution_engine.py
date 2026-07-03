# -*- coding: utf-8 -*-
"""ClipPilot Brain Execution Engine.

coordinates sequence of execution, timing tracking, progress reporting,
retries, and cancellation loops for pipeline stages.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ExecutionStage(Enum):
    """Available stages in the orchestrator pipeline execution flow."""
    VOICE = "VOICE"
    RENDER = "RENDER"
    QA = "QA"
    PUBLISH = "PUBLISH"


class ExecutionStatus(Enum):
    """State status of a running execution pipeline."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class ExecutionContext:
    """Carries runtime execution context and cancel tokens."""
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False


@dataclass
class ExecutionResult:
    """Report summarizing the outcome of a stage execution run."""
    stage: ExecutionStage
    status: ExecutionStatus
    duration_seconds: float
    attempts: int
    error: Optional[str] = None
    logs: list[str] = field(default_factory=list)


class ExecutionEngine:
    """Coordinates sequential stage executions, retries, and cancellation gates."""

    def __init__(self) -> None:
        self._runners: dict[ExecutionStage, Callable[[ExecutionContext], Any]] = {}
        self._retry_limits: dict[ExecutionStage, int] = {
            ExecutionStage.VOICE: 3,
            ExecutionStage.RENDER: 2,
            ExecutionStage.QA: 1,
            ExecutionStage.PUBLISH: 3,
        }

    def register_runner(
        self, stage: ExecutionStage, runner: Callable[[ExecutionContext], Any], retries: int = 1
    ) -> None:
        """Register a executable handler for a specific stage."""
        self._runners[stage] = runner
        self._retry_limits[stage] = retries

    def execute(self, context: ExecutionContext) -> list[ExecutionResult]:
        """Runs registered stages sequentially. Halts on unrecoverable failures or cancellations."""
        results: list[ExecutionResult] = []
        pipeline_stages = [
            ExecutionStage.VOICE,
            ExecutionStage.RENDER,
            ExecutionStage.QA,
            ExecutionStage.PUBLISH,
        ]

        for stage in pipeline_stages:
            # 1. Cancellation check
            if context.cancelled:
                results.append(
                    ExecutionResult(
                        stage=stage,
                        status=ExecutionStatus.CANCELLED,
                        duration_seconds=0.0,
                        attempts=0,
                        error="Pipeline execution cancelled by user.",
                        logs=["Stage skipped due to cancellation token."],
                    )
                )
                continue

            runner = self._runners.get(stage)
            if not runner:
                # Stage is not registered; treat as skipped/completed
                results.append(
                    ExecutionResult(
                        stage=stage,
                        status=ExecutionStatus.COMPLETED,
                        duration_seconds=0.0,
                        attempts=0,
                        logs=["Stage completed (no runner registered)."],
                    )
                )
                continue

            attempts = 0
            limit = self._retry_limits.get(stage, 1)
            start_time = time.time()
            stage_status = ExecutionStatus.RUNNING
            last_error: Optional[str] = None
            stage_logs = []

            while attempts < limit:
                # Timing cancellation within retry loop
                if context.cancelled:
                    stage_status = ExecutionStatus.CANCELLED
                    stage_logs.append("Attempt cancelled mid-run.")
                    break

                attempts += 1
                stage_logs.append(f"Starting stage attempt {attempts}/{limit}...")
                try:
                    runner(context)
                    stage_status = ExecutionStatus.COMPLETED
                    stage_logs.append("Stage execution succeeded.")
                    last_error = None
                    break
                except Exception as e:
                    last_error = str(e)
                    stage_logs.append(f"Attempt {attempts} failed: {last_error}")
                    stage_status = ExecutionStatus.FAILED

            duration = time.time() - start_time

            results.append(
                ExecutionResult(
                    stage=stage,
                    status=stage_status,
                    duration_seconds=round(duration, 3),
                    attempts=attempts,
                    error=last_error,
                    logs=stage_logs,
                )
            )

            # If a stage fails, halt the pipeline execution (do not propagate to next stages)
            if stage_status == ExecutionStatus.FAILED:
                break

        return results
