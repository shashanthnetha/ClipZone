# -*- coding: utf-8 -*-
"""ClipPilot Readiness Doctor.

Diagnostic checks verifying system tool dependencies, node ecosystem,
TTS, API keys, assets, configuration, and directory status.
"""
from __future__ import annotations

import sys
import subprocess
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def _check(name: str, ok: bool, detail: str, unlocks: str, required: bool = False) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail, "unlocks": unlocks, "required": required}


def _run_cmd(cmd: List[str]) -> Tuple[bool, str]:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=5)
        return True, res.stdout.decode("utf-8").strip()
    except Exception as e:
        return False, str(e)


def check_readiness(workspace_dir: Optional[Path] = None) -> dict[str, Any]:
    """Inspect the environment and return {ready, checks, next_steps}."""
    if workspace_dir is None:
        workspace_dir = Path(__file__).resolve().parent.parent.parent.parent

    # 1. Python version check
    py_ok = sys.version_info >= (3, 8)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # 2. System command dependencies
    node_ok, node_ver = _run_cmd(["node", "--version"])
    npm_ok, npm_ver = _run_cmd(["npm", "--version"])
    
    # 3. Remotion check (via npx remotion --version or package.json scan)
    remotion_ok, remotion_ver = _run_cmd(["npx", "remotion", "--version"])
    explainer_dir = workspace_dir / "ClipPilot" / "remotion_explainer"
    if not remotion_ok:
        # Fallback check package.json
        pkg_json_path = explainer_dir / "package.json"
        if pkg_json_path.exists():
            try:
                pkg_data = json.loads(pkg_json_path.read_text(encoding="utf-8"))
                deps = pkg_data.get("dependencies", {})
                rem_ver = deps.get("remotion")
                if rem_ver:
                    remotion_ok = True
                    remotion_ver = f"configured: {rem_ver}"
            except Exception:
                pass

    # 4. ffmpeg check
    from .media.ffmpeg import ffmpeg_available
    ffmpeg_ok = ffmpeg_available()

    # 5. faster-whisper check
    from .media.transcribe import whisper_available
    whisper_ok = whisper_available()

    # 6. edge-tts / TTS check
    from .media.tts import tts_available
    tts_ok = tts_available()

    # 7. yt-dlp check
    from .media.download import ytdlp_available
    ytdlp_ok = ytdlp_available()

    # 8. API Keys status
    from .brain import env as benv
    benv.load_dotenv()
    has_key = benv.has_api_key()

    from .config import Settings
    settings = Settings.load()
    pexels_key_set = bool(settings.pexels_api_key)
    pixabay_key_set = bool(settings.pixabay_api_key)
    unsplash_key_set = bool(settings.unsplash_api_key)

    # 9. Assets and folders check
    public_assets_ok = (explainer_dir / "public").exists()
    output_folder_ok = (explainer_dir / "out").exists()
    
    # 10. Settings configuration check
    from .config import SETTINGS_PATH
    settings_ok = False
    settings_detail = "settings.json missing"
    if SETTINGS_PATH.exists():
        try:
            json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            settings_ok = True
            settings_detail = f"valid JSON at {SETTINGS_PATH}"
        except Exception as e:
            settings_detail = f"invalid JSON at {SETTINGS_PATH}: {e}"

    # YouTube config checks
    yt = all(os.environ.get(k) for k in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"))
    yt_partial = bool(os.environ.get("YOUTUBE_CLIENT_ID") and os.environ.get("YOUTUBE_CLIENT_SECRET"))
    up = bool(os.environ.get("UPLOAD_POST_API_KEY") and os.environ.get("UPLOAD_POST_USERNAME"))
    
    # PySide6 GUI check
    try:
        import PySide6  # noqa: F401
        pyside_ok = True
    except ImportError:
        pyside_ok = False

    checks = [
        _check("Python version", py_ok, py_ver, "run engine code", required=True),
        _check("ffmpeg", ffmpeg_ok, "bundled via imageio-ffmpeg" if ffmpeg_ok else "missing", "signals, clipping, captions, compose", required=True),
        _check("Node.js", node_ok, node_ver if node_ok else "missing", "npx package executor", required=True),
        _check("npm", npm_ok, npm_ver if npm_ok else "missing", "Remotion package installer", required=True),
        _check("Remotion", remotion_ok, remotion_ver if remotion_ok else "missing", "React video rendering compiler", required=True),
        _check("faster-whisper", whisper_ok, "installed" if whisper_ok else "pip install faster-whisper", "transcription + word-timed karaoke captions"),
        _check("TTS (Chatterbox / edge-tts)", tts_ok, "ready" if tts_ok else "no engine (Chatterbox venv or edge-tts)", "Section B/C narration"),
        _check("yt-dlp", ytdlp_ok, "installed" if ytdlp_ok else "pip install yt-dlp", "download a source video from a URL for Section-A clipping"),
        _check("Anthropic API key", has_key, "set" if has_key else "add ANTHROPIC_API_KEY to .env", "Claude vision understanding, smart highlight picks, script/metadata (deterministic fallback otherwise)"),
        _check("Pexels API key", pexels_key_set, "set" if pexels_key_set else "not configured", "Pexels stock video downloads"),
        _check("Pixabay API key", pixabay_key_set, "set" if pixabay_key_set else "not configured", "Pixabay stock video downloads"),
        _check("Unsplash API key", unsplash_key_set, "set" if unsplash_key_set else "not configured", "Unsplash stock image downloads"),
        _check("YouTube publishing (free)", yt, "configured" if yt else ("client set — run youtube_auth for a refresh token" if yt_partial else "not configured"), "FREE first-party publish to YouTube"),
        _check("Upload-Post (paid)", up, "configured" if up else "not configured", "paid cross-post: YouTube + IG/FB + TikTok"),
        _check("Hosted image-gen (paid, optional)", bool(os.environ.get("GEN_IMAGE_API_KEY")), "configured" if os.environ.get("GEN_IMAGE_API_KEY") else "not configured (uses stock b-roll)", "AI-generated cinematic Section-B stills (GEN_IMAGE_API_KEY)"),
        _check("PySide6 GUI", pyside_ok, "installed" if pyside_ok else "pip install PySide6", "the native Windows app (run.bat)"),
        _check("Public assets", public_assets_ok, "found" if public_assets_ok else f"missing at {explainer_dir}/public", "Remotion graphic resource loading"),
        _check("Output folders", output_folder_ok, "found" if output_folder_ok else f"missing at {explainer_dir}/out (run pipeline to create)", "video compilation outputs"),
        _check("Configuration file", settings_ok, settings_detail, "engine runtime custom overrides"),
    ]

    ready = {
        "clip_sectionA": ffmpeg_ok,
        "captions": ffmpeg_ok and whisper_ok,
        "generate_sectionB": ffmpeg_ok and tts_ok,
        "brain": has_key,
        "publish_free": ffmpeg_ok and yt,
        "publish_paid": ffmpeg_ok and up,
        "can_publish": ffmpeg_ok and (yt or up),
    }

    steps = []
    if ready["clip_sectionA"] and ready["can_publish"]:
        steps.append("READY to clip AND publish. Add a Section-A job (owned/authorized source), run the engine, and enable auto-approve on a trusted lane for unattended runs.")
    elif ready["clip_sectionA"]:
        steps.append("READY to clip (output saved locally). Configure a publisher to post automatically.")
    if not ffmpeg_ok:
        steps.append("Install deps so ffmpeg is available: pip install -r src/requirements.txt.")
    if not (yt or up):
        steps.append("Set up FREE publishing: create a Google OAuth *Desktop* client (enable YouTube Data API v3), put YOUTUBE_CLIENT_ID/SECRET in .env, then run `python -m clippilot.publish.youtube_auth --write-env` (docs/09 step 7).")
    elif yt_partial and not yt:
        steps.append("Finish YouTube auth: run `python -m clippilot.publish.youtube_auth --write-env` to obtain YOUTUBE_REFRESH_TOKEN.")
    if not has_key:
        steps.append("Optional: add ANTHROPIC_API_KEY to .env for human-like understanding + smart clip picks (works without it via deterministic fallback).")
    if not whisper_ok:
        steps.append("Install captions support: pip install faster-whisper.")

    return {"ready": ready, "checks": checks, "next_steps": steps}


def format_report(report: dict[str, Any]) -> str:
    """Human-readable rendering for the CLI."""
    lines = ["ClipPilot readiness", "=" * 40]
    for c in report["checks"]:
        mark = "[OK]" if c["ok"] else ("[!!]" if c["required"] else "[--]")
        req = " (required)" if c["required"] and not c["ok"] else ""
        lines.append(f"  {mark} {c['name']}: {c['detail']}{req}")
        lines.append(f"        unlocks: {c['unlocks']}")
    
    r = report["ready"]
    lines += ["", "Capabilities:",
              f"  clip Section A : {'yes' if r['clip_sectionA'] else 'no'}",
              f"  captions       : {'yes' if r['captions'] else 'no'}",
              f"  generate Sec B : {'yes' if r['generate_sectionB'] else 'no'}",
              f"  Claude brain   : {'yes' if r['brain'] else 'no (deterministic fallback)'}",
              f"  publish (free) : {'yes' if r['publish_free'] else 'no'}",
              f"  publish (paid) : {'yes' if r['publish_paid'] else 'no'}"]
              
    if report["next_steps"]:
        lines += ["", "Next steps:"]
        lines += [f"  {i}. {s}" for i, s in enumerate(report["next_steps"], 1)]
    return "\n".join(lines)
