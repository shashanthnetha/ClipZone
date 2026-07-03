"""ClipPilot — native Windows app with Claude (via MCP) as the orchestrating brain
for an authorized/owned-source clip & short-form video pipeline.

Phase 0 (this package, stdlib-only): config, SQLite job store, the job state
machine, guardrails, a stub stage-runner registry, and a CLI to exercise the
whole orchestration loop end-to-end before any heavy media/ML/GUI dependency
exists. See ../docs/ for the plan; 06-decisions-and-product.md for the why.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
