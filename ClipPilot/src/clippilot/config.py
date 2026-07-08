"""Configuration: paths, feature flags, and guardrail toggles.

Guardrail defaults encode the legal posture chosen in docs/06: the owner opted
for "third-party freely" sourcing, so the hard ingest allow-list is OFF, but the
harm-reduction guardrails default ON (and are user-toggleable) — except the
real-person face-swap block, which is a hard line.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Automatically detect test environment and force mock mode
if "unittest" in sys.modules or "pytest" in sys.modules or os.environ.get("CLIPPILOT_TESTING") == "true":
    os.environ["CLIPPILOT_TESTING"] = "true"
    for key in [
        "LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "PEXELS_API_KEY", "PIXABAY_API_KEY", "UNSPLASH_API_KEY"
    ]:
        if key in os.environ:
            del os.environ[key]

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
    enable_reasoning_correction: bool = True
    script_model: str = ""
    critic_model: str = ""
    vision_model: str = ""
    asset_model: str = ""
    script_models: list[str] = field(default_factory=lambda: [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "qwen/qwen-2.5-72b-instruct:free"
    ])
    critic_models: list[str] = field(default_factory=lambda: [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "qwen/qwen-2.5-72b-instruct:free"
    ])
    vision_models: list[str] = field(default_factory=lambda: [
        "google/gemini-2.0-flash-exp:free",
        "google/gemini-flash-1.5-8b:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "qwen/qwen2.5-vl-72b-instruct:free"
    ])
    asset_models: list[str] = field(default_factory=lambda: [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free"
    ])

    # ── Asset Provider Configuration ──
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    unsplash_api_key: str = ""
    asset_provider_priority: list[str] = field(default_factory=lambda: ["pexels", "pixabay", "unsplash"])

    # ── persistence ──
    # ── validation and overrides ──
    def validate(self) -> None:
        """Validate settings fields and raise ValueError with helpful messages if invalid."""
        if not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError(f"max_attempts must be an integer >= 1, got {self.max_attempts}")
        
        if self.default_section not in ("A", "B", "C"):
            raise ValueError(f"default_section must be 'A', 'B', or 'C', got '{self.default_section}'")
            
        if not isinstance(self.brain_frame_budget_per_min, int) or self.brain_frame_budget_per_min < 1:
            raise ValueError(f"brain_frame_budget_per_min must be an integer >= 1, got {self.brain_frame_budget_per_min}")
            
        if not (0.0 <= self.bgm_volume <= 1.0):
            raise ValueError(f"bgm_volume must be between 0.0 and 1.0 inclusive, got {self.bgm_volume}")
            
        valid_providers = ("anthropic", "openai", "openrouter")
        if self.llm_provider not in valid_providers:
            raise ValueError(f"llm_provider must be one of {valid_providers}, got '{self.llm_provider}'")

    # ── persistence ──
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        raw_g = d.get("guardrails")
        if not isinstance(raw_g, dict):
            raw_g = {}
        allowed = set(Guardrails.__dataclass_fields__)
        g = Guardrails(**{k: v for k, v in raw_g.items() if k in allowed})
        
        priority = d.get("asset_provider_priority")
        if isinstance(priority, str):
            priority = [x.strip() for x in priority.split(",") if x.strip()]
        elif not isinstance(priority, list):
            priority = ["pexels", "pixabay", "unsplash"]

        def parse_model_list(val: Any, default_list: list[str]) -> list[str]:
            if val is None:
                return default_list
            if isinstance(val, str):
                return [x.strip() for x in val.split(",") if x.strip()]
            elif isinstance(val, list):
                return [str(x) for x in val]
            return default_list

        script_models = parse_model_list(d.get("script_models"), [
            "openai/gpt-oss-120b:free",
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "meta-llama/llama-3.3-70b-instruct:free"
        ])
        critic_models = parse_model_list(d.get("critic_models"), [
            "openai/gpt-oss-120b:free",
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free"
        ])
        vision_models = parse_model_list(d.get("vision_models"), [
            "openrouter/free",
            "google/gemini-flash-1.5-8b:free",
            "qwen/qwen2.5-vl-72b-instruct:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free"
        ])
        asset_models = parse_model_list(d.get("asset_models"), [
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "openai/gpt-oss-120b:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free"
        ])

        return cls(
            auto_approve=bool(d.get("auto_approve", False)),
            max_attempts=int(d.get("max_attempts", d.get("max_attempts", 3))),
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
            enable_reasoning_correction=bool(d.get("enable_reasoning_correction", False)),
            script_model=str(d.get("script_model", "")),
            critic_model=str(d.get("critic_model", "")),
            vision_model=str(d.get("vision_model", "")),
            asset_model=str(d.get("asset_model", "")),
            script_models=script_models,
            critic_models=critic_models,
            vision_models=vision_models,
            asset_models=asset_models,
            pexels_api_key=str(d.get("pexels_api_key", "")),
            pixabay_api_key=str(d.get("pixabay_api_key", "")),
            unsplash_api_key=str(d.get("unsplash_api_key", "")),
            asset_provider_priority=priority,
        )

    def save(self, path: Path | None = None) -> None:
        path = path or SETTINGS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or SETTINGS_PATH
        d = {}
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        
        # Instantiate
        settings = cls.from_dict(d)
        
        # Apply environment variable overrides
        from .brain import env as benv
        benv.load_dotenv()
        
        # Standard field overrides
        for field_name in cls.__dataclass_fields__:
            if field_name == "guardrails":
                continue
            
            # Check env keys: e.g. CLIPPILOT_LLM_PROVIDER, LLM_PROVIDER
            env_keys = [f"CLIPPILOT_{field_name.upper()}", field_name.upper()]
            for env_key in env_keys:
                env_val = os.environ.get(env_key)
                if env_val is not None:
                    # Cast value to correct type based on default/type
                    field_type = cls.__dataclass_fields__[field_name].type
                    if field_type is bool or field_type == "bool":
                        setattr(settings, field_name, env_val.lower() in ("true", "yes", "1"))
                    elif field_type is int or field_type == "int":
                        setattr(settings, field_name, int(env_val))
                    elif field_type is float or field_type == "float":
                        setattr(settings, field_name, float(env_val))
                    elif field_name in ("asset_provider_priority", "script_models", "critic_models", "vision_models", "asset_models"):
                        setattr(settings, field_name, [x.strip() for x in env_val.split(",") if x.strip()])
                    else:
                        setattr(settings, field_name, env_val)
                    break
        
        # Map secrets from provider-specific variables if not already set
        if not settings.llm_api_key:
            if settings.llm_provider == "anthropic":
                settings.llm_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            elif settings.llm_provider == "openai":
                settings.llm_api_key = os.environ.get("OPENAI_API_KEY", "")
            elif settings.llm_provider == "openrouter":
                settings.llm_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not settings.pexels_api_key:
            settings.pexels_api_key = os.environ.get("PEXELS_API_KEY", "")
        if not settings.pixabay_api_key:
            settings.pixabay_api_key = os.environ.get("PIXABAY_API_KEY", "")
        if not settings.unsplash_api_key:
            settings.unsplash_api_key = os.environ.get("UNSPLASH_API_KEY", "") or os.environ.get("UNSPLASH_ACCESS_KEY", "")
        
        # Override llm_api_key specifically if LLM_API_KEY is present
        api_key_override = os.environ.get("CLIPPILOT_LLM_API_KEY") or os.environ.get("LLM_API_KEY")
        if api_key_override:
            settings.llm_api_key = api_key_override
            
        # Validate the final merged settings
        settings.validate()
        
        return settings
