"""Configuration: paths, feature flags, and guardrail toggles.

Guardrail defaults encode the legal posture chosen in docs/06: the owner opted
for "third-party freely" sourcing, so the hard ingest allow-list is OFF, but the
harm-reduction guardrails default ON (and are user-toggleable) — except the
real-person face-swap block, which is a hard line.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_NAME = "ClipPilot"

# ── Paths ────────────────────────────────────────────────────────────────────
PKG_DIR = Path(__file__).resolve().parent           # .../src/clippilot
SRC_DIR = PKG_DIR.parent                             # .../src
PROJECT_ROOT = SRC_DIR.parent                        # .../ClipPilot


def _default_data_dir() -> Path:
    env = os.environ.get("CLIPPILOT_DATA")
    if env:
        return Path(env)
    return PROJECT_ROOT / "data"


DATA_DIR = _default_data_dir()
DB_PATH = DATA_DIR / "clippilot.db"
MEDIA_DIR = DATA_DIR / "media"
LOG_DIR = DATA_DIR / "logs"
SETTINGS_PATH = DATA_DIR / "settings.json"


def ensure_dirs() -> None:
    for d in (DATA_DIR, MEDIA_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Env-aware path helpers — recompute from CLIPPILOT_DATA at call time, so a
# process (MCP server, service) that has CLIPPILOT_DATA set at runtime (incl.
# tests) resolves the right store without re-importing this module.
def data_dir() -> Path:
    return _default_data_dir()


def db_path() -> Path:
    return data_dir() / "clippilot.db"


def settings_path() -> Path:
    return data_dir() / "settings.json"


def media_dir() -> Path:
    return data_dir() / "media"


def merge_settings(current: dict, updates: dict) -> dict:
    """Merge a partial settings update into a settings dict; the nested
    `guardrails` object merges field-by-field (untouched toggles are preserved).
    Shared by the MCP `set_settings` tool and the GUI Settings tab."""
    d = dict(current)
    for k, v in updates.items():
        if k == "guardrails" and isinstance(v, dict):
            g = dict(d.get("guardrails") or {})
            g.update(v)
            d["guardrails"] = g
        else:
            d[k] = v
    return d


# ── Guardrails (legal/harm-reduction; see docs/04 + docs/06) ─────────────────
@dataclass
class Guardrails:
    approval_gate: bool = True          # require human approval before publish
    ai_disclosure: bool = True          # auto-apply "altered/synthetic" label on publish
    strike_tracking: bool = True        # track copyright strikes per channel
    strike_pause_threshold: int = 2     # auto-pause a channel at N strikes (termination = 3 / 90 days)
    transformative_nudge: bool = True   # instruct the brain to add commentary/value, not raw re-upload
    block_realperson_faceswap: bool = True  # HARD LINE — no non-consensual real-person face/voice swap


# ── Settings ─────────────────────────────────────────────────────────────────
@dataclass
class Settings:
    auto_approve: bool = False          # skip the human gate (NOT recommended; for trusted templates only)
    max_attempts: int = 3               # per-stage retry budget before NEEDS_ATTENTION
    default_section: str = "A"          # A=paid clipping/DFY, B=faceless funnel, C=ad-share
    # The Claude model the brain uses for the vision pass. Default is the most
    # capable (claude-opus-4-8, $5/$25 per 1M tok). For cheaper bulk vision set
    # "claude-sonnet-4-6" ($3/$15). See docs/07 cost model.
    brain_model: str = "claude-opus-4-8"
    brain_frame_budget_per_min: int = 6  # docs/07 default
    compose_concat: bool = False         # stitch a job's clips into one compilation (default: separate shorts)
    bgm_path: str = ""                   # user-supplied CLEARED/royalty-free music bed for Section B (never auto-fetched — guardrail)
    bgm_volume: float = 0.12             # BGM mixed low under the narration
    caption_skin: str = "karaoke_yellow"  # default caption look (edit.CAPTION_SKINS; docs/10)
    guardrails: Guardrails = field(default_factory=Guardrails)

    # ── LLM Provider Configuration ──
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-4-8"
    llm_base_url: str = ""
    llm_api_key: str = ""

    # ── persistence ──
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        # Tolerate unknown/typo'd or non-dict guardrail keys instead of crashing
        # (reachable via the MCP set_settings tool, which accepts a free-form object).
        raw_g = d.get("guardrails")
        if not isinstance(raw_g, dict):
            raw_g = {}
        allowed = set(Guardrails.__dataclass_fields__)
        g = Guardrails(**{k: v for k, v in raw_g.items() if k in allowed})
        return cls(
            auto_approve=bool(d.get("auto_approve", False)),
            max_attempts=int(d.get("max_attempts", 3)),
            default_section=str(d.get("default_section", "A")),
            brain_model=str(d.get("brain_model", "claude-opus-4-8")),
            brain_frame_budget_per_min=int(d.get("brain_frame_budget_per_min", 6)),
            compose_concat=bool(d.get("compose_concat", False)),
            bgm_path=str(d.get("bgm_path", "")),
            bgm_volume=float(d.get("bgm_volume", 0.12)),
            caption_skin=str(d.get("caption_skin", "karaoke_yellow")),
            guardrails=g,
            llm_provider=str(d.get("llm_provider", "anthropic")),
            llm_model=str(d.get("llm_model", d.get("brain_model", "claude-opus-4-8"))),
            llm_base_url=str(d.get("llm_base_url", "")),
            llm_api_key=str(d.get("llm_api_key", "")),
        )

    def save(self, path: Path | None = None) -> None:
        path = path or SETTINGS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or SETTINGS_PATH
        if path.exists():
            try:
                return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return cls()
