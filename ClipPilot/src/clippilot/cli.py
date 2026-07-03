"""Command-line interface for the Phase 0 engine.

    python -m clippilot init
    python -m clippilot enqueue --source video.mp4 --section A --rights owned --channel my_yt
    python -m clippilot run
    python -m clippilot list [--status pending]
    python -m clippilot show 1
    python -m clippilot approve 1
    python -m clippilot reject 1 --reason "off-brand"
    python -m clippilot strike my_yt
    python -m clippilot status
    python -m clippilot demo        # self-contained smoke test of the whole loop
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from . import config as cfg
from . import db as dbmod
from .engine import Engine
from .models import JobStatus, RightsTag, Section
from .queue import JobQueue


def _queue() -> JobQueue:
    cfg.ensure_dirs()
    conn = dbmod.connect(cfg.DB_PATH)
    return JobQueue(conn, cfg.Settings.load())


def _skin_choices() -> list[str]:
    """Caption-skin keys for argparse choices (lazy import — edit pulls in ffmpeg)."""
    from .media.edit import CAPTION_SKINS
    return list(CAPTION_SKINS)


def _print_job(q: JobQueue, job) -> None:
    print(f"#{job.id}  [{job.section.value}] {job.status.value:<16} stage={job.stage.value:<15}"
          f" rights={job.rights_tag.value:<11} attempts={job.attempts}/{job.max_attempts}"
          f" channel={job.channel or '-'}")
    if job.error:
        print(f"      error: {job.error}")


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_init(_: argparse.Namespace) -> int:
    q = _queue()
    print(f"Initialized ClipPilot data at: {cfg.DATA_DIR}")
    print(f"  db:       {cfg.DB_PATH}")
    print(f"  settings: {cfg.SETTINGS_PATH}")
    cfg.Settings.load().save()
    return 0


def cmd_enqueue(a: argparse.Namespace) -> int:
    q = _queue()
    job = q.enqueue(
        source_ref=a.source,
        section=Section(a.section),
        rights_tag=RightsTag(a.rights),
        channel=a.channel,
        idempotency_key=a.key,
    )
    print("Enqueued:")
    _print_job(q, job)
    if job.rights_tag == RightsTag.THIRD_PARTY:
        print("  ⚠ third-party source: elevated copyright/strike risk (guardrails apply).")
    return 0


def cmd_enqueue_folder(a: argparse.Namespace) -> int:
    from .batch import enqueue_folder
    q = _queue()
    res = enqueue_folder(q, a.folder, section=a.section, rights=a.rights,
                         channel=a.channel, recursive=not a.no_recursive)
    print(f"Found {res['found']} video(s) in {a.folder}; enqueued {res['count']}.")
    for jid in res["enqueued_ids"]:
        print(f"  job #{jid}")
    return 0


def cmd_profile_list(_: argparse.Namespace) -> int:
    from .profiles import load_profiles
    profs = load_profiles()
    if not profs:
        print("(no profiles)")
        return 0
    for p in profs.values():
        print(f"  {p.name}: section={p.section} rights={p.rights} channel={p.channel or '-'} "
              f"voice={p.voice or '-'} bgm={p.bgm_path or '-'} secs={p.target_seconds}")
    return 0


def cmd_profile_save(a: argparse.Namespace) -> int:
    from .profiles import Profile, save_profile
    p = save_profile(Profile(name=a.name, section=a.section, rights=a.rights, channel=a.channel,
                             voice=a.voice, bgm_path=a.bgm, target_seconds=a.seconds,
                             mode=a.mode, facts_count=a.facts_count, caption_skin=a.skin))
    print(f"Saved DFY template {p.name!r}.")
    return 0


def cmd_profile_delete(a: argparse.Namespace) -> int:
    from .profiles import delete_profile
    print("Deleted." if delete_profile(a.name) else f"No profile {a.name!r}.")
    return 0


def cmd_enqueue_profile(a: argparse.Namespace) -> int:
    from .profiles import enqueue_with_profile, get_profile
    p = get_profile(a.profile)
    if not p:
        print(f"No profile {a.profile!r}", file=sys.stderr)
        return 1
    q = _queue()
    job = enqueue_with_profile(q, a.source, p)
    print(f"Enqueued with template {p.name!r}:")
    _print_job(q, job)
    return 0


def cmd_resync_srt(a: argparse.Namespace) -> int:
    from .media.subsync import sync_subtitles
    res = sync_subtitles(a.srt, a.audio, out_path=a.out)
    if not res:
        print("Could not resync (need ffmpeg + an existing SRT + media).", file=sys.stderr)
        return 1
    print(f"Resynced by {res['offset_s']:+.3f}s → {res['path']} ({res['cues']} cues)")
    return 0


def cmd_download(a: argparse.Namespace) -> int:
    from .media.download import download_source
    out = a.out or str(cfg.media_dir() / "downloads")
    res = download_source(a.url, out, max_height=a.max_height)
    if not res.get("available"):
        print(f"Download failed: {res.get('reason')}", file=sys.stderr)
        return 1
    print(f"Downloaded → {res['path']}  ({res.get('title') or ''})")
    return 0


def cmd_enqueue_url(a: argparse.Namespace) -> int:
    """Download a source video from a URL and enqueue it as a (Section A) clipping
    job. Only for content you own or are authorized to repurpose (rights-gated)."""
    from .media.download import download_source
    res = download_source(a.url, str(cfg.media_dir() / "downloads"), max_height=a.max_height)
    if not res.get("available"):
        print(f"Download failed: {res.get('reason')}", file=sys.stderr)
        return 1
    q = _queue()
    job = q.enqueue(res["path"], section=Section(a.section), rights_tag=RightsTag(a.rights),
                    channel=a.channel)
    print(f"Downloaded → {res['path']}; enqueued job #{job.id} [Section {a.section}, "
          f"rights={a.rights}]")
    return 0


def cmd_editor_render(a: argparse.Namespace) -> int:
    from .editor.project import load_project
    from .editor.render import render_timeline
    res = render_timeline(load_project(a.project), a.out)
    if not res:
        print("Editor render failed (check the project JSON + media paths).", file=sys.stderr)
        return 1
    print(f"Rendered timeline → {res}")
    return 0


def cmd_explainer(a: argparse.Namespace) -> int:
    from .generate.explainer import explainer_clip
    bullets = [b for b in (a.bullets or "").split(";") if b.strip()]
    res = explainer_clip(a.title, bullets, a.out, duration=a.seconds)
    if not res:
        print("Explainer render failed (need ffmpeg + a system font).", file=sys.stderr)
        return 1
    print(f"Explainer clip → {res}")
    return 0


def cmd_cutout(a: argparse.Namespace) -> int:
    from .media.cutout import cutout_available, cutout_broll
    if not cutout_available():
        print("Subject cutout needs rembg (pip install rembg).", file=sys.stderr)
        return 1
    res = cutout_broll(a.image, a.bg, a.out)
    if not res:
        print("Cutout failed (model/compose error).", file=sys.stderr)
        return 1
    print(f"Cutout composited → {res}")
    return 0


def cmd_publish_reel(a: argparse.Namespace) -> int:
    from .publish.instagram import publisher_from_env
    pub = publisher_from_env()
    if not pub:
        print("Instagram not configured (set INSTAGRAM_USER_ID + INSTAGRAM_ACCESS_TOKEN).",
              file=sys.stderr)
        return 1
    res = pub.upload_reel(a.url, caption=a.caption or "")
    if res.get("success"):
        print(f"Published Reel: {res.get('media_id')}")
        return 0
    print(f"Failed: {res.get('error') or res}", file=sys.stderr)
    return 1


def cmd_run(a: argparse.Namespace) -> int:
    if getattr(a, "legacy", False):
        q = _queue()
        eng = Engine(q)
        steps = eng.drain(max_steps=a.max)
        print(f"Engine ran {steps} step(s). Status counts: {q.counts()}")
        blocked = q.list(JobStatus.BLOCKED_APPROVAL)
        if blocked:
            print(f"  {len(blocked)} job(s) awaiting approval: {[j.id for j in blocked]}")
        attn = q.list(JobStatus.NEEDS_ATTENTION)
        if attn:
            print(f"  {len(attn)} job(s) need attention: {[j.id for j in attn]}")
        return 0

    from clippilot.brain.pipeline_demo import execute_production_pipeline
    from pathlib import Path
    print("Running end-to-end production pipeline...")
    report = execute_production_pipeline(
        workspace_dir=Path(a.workspace),
        variation_dir=Path(a.variation),
        date_str=a.date,
        slug=a.slug,
    )
    if report.get("success"):
        print("✔️ Pipeline completed successfully.")
        return 0
    else:
        print(f"❌ Pipeline failed: {report.get('error')}")
        return 1


def cmd_render(a: argparse.Namespace) -> int:
    from clippilot.brain.pipeline_demo import execute_production_pipeline
    from pathlib import Path
    print("Compiling and rendering Remotion video...")
    report = execute_production_pipeline(
        workspace_dir=Path(a.workspace),
        variation_dir=Path(a.variation),
        date_str=a.date,
        slug=a.slug,
        skip_qa_publish=True,
    )
    if report.get("success"):
        print("✔️ Video rendering completed successfully.")
        return 0
    else:
        print(f"❌ Video rendering failed: {report.get('error')}")
        return 1


def cmd_learn(a: argparse.Namespace) -> int:
    from clippilot.brain.performance_store import PerformanceStore
    from clippilot.brain.learning_engine import LearningEngine
    from pathlib import Path
    store_path = Path(a.workspace) / "ClipPilot" / "performance_history.jsonl"
    store = PerformanceStore(store_path)
    le = LearningEngine(store)
    summary_path = le.save_summary()
    print(f"✔️ Learning Engine run complete. Summary saved to {summary_path}")
    return 0


def cmd_analytics(a: argparse.Namespace) -> int:
    from clippilot.brain.performance_store import PerformanceStore
    from clippilot.brain.youtube_analytics import YouTubeAnalyticsProvider
    from clippilot.brain.analytics_models import AnalyticsMetrics
    from pathlib import Path
    store_path = Path(a.workspace) / "ClipPilot" / "performance_history.jsonl"
    store = PerformanceStore(store_path)
    provider = YouTubeAnalyticsProvider()
    records = store.load_records()
    synced = 0
    for r in records:
        yt_id = r.upload_metrics.video_id_on_platform
        if yt_id and yt_id != "dry_run_yt_123":
            try:
                stats = provider.get_video_statistics(yt_id)
                metrics = AnalyticsMetrics(
                    views=stats.get("views", 0),
                    likes=stats.get("likes", 0),
                    comments=stats.get("comments", 0),
                    impressions=stats.get("impressions", 0),
                    ctr=stats.get("ctr", 0.0),
                    avg_view_duration_seconds=stats.get("average_view_duration_seconds", 0.0),
                    avg_percentage_viewed=stats.get("average_percentage_viewed", 0.0),
                    watch_time_hours=stats.get("watch_time_hours", 0.0),
                    subscribers_gained=stats.get("subscribers_gained", 0),
                    revenue_usd=stats.get("estimated_revenue_usd", 0.0),
                )
                store.update_analytics(r.video_id, metrics)
                print(f"Synced video {r.video_id} (YouTube ID: {yt_id})")
                synced += 1
            except Exception as e:
                print(f"⚠️ Failed to sync {r.video_id}: {e}")
    print(f"✔️ Analytics sync complete. Total synced: {synced}")
    return 0


def cmd_strategy(a: argparse.Namespace) -> int:
    from clippilot.brain.performance_store import PerformanceStore
    from clippilot.brain.learning_engine import LearningEngine
    from clippilot.brain.strategy_engine import StrategyEngine
    from clippilot.brain.pipeline_orchestrator import RunContext, load_state
    from pathlib import Path
    store_path = Path(a.workspace) / "ClipPilot" / "performance_history.jsonl"
    store = PerformanceStore(store_path)
    le = LearningEngine(store)
    se = StrategyEngine(store, le)
    run_ctx = RunContext(workspace_dir=Path(a.workspace), variation_dir=Path(a.variation), date_str=a.date, slug=a.slug)
    state = load_state(run_ctx)
    decision = se.make_decision(state, Path(a.variation))
    print("\n🎯 Next Strategy Decision:")
    print(f"Topic:      {decision.topic.num} - {decision.topic.title}")
    print(f"Hook:       {decision.hook}")
    print(f"Voice:      {decision.voice}")
    print(f"Skin:       {decision.skin}")
    print(f"Format:     {decision.format}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reasoning:  {decision.reasoning}")
    return 0


def cmd_history(a: argparse.Namespace) -> int:
    from clippilot.brain.performance_store import PerformanceStore
    from pathlib import Path
    store_path = Path(a.workspace) / "ClipPilot" / "performance_history.jsonl"
    store = PerformanceStore(store_path)
    records = store.load_records()
    if not records:
        print("No performance records found.")
        return 0
    print(f"\nRecorded History ({len(records)} videos):")
    for r in records:
        print(f"- [{r.timestamp}] {r.video_id}: Topic {r.topic.get('num', '???')} - '{r.topic.get('title', '???')}'")
        print(f"  Variation: hook={r.variation.get('hook')} voice={r.variation.get('voice')} skin={r.variation.get('skin')} format={r.variation.get('fmt')}")
        if r.strategy_metadata:
            print(f"  Strategy: final_ucb={r.strategy_metadata.get('final_ucb_score')}")
    return 0


def cmd_config(a: argparse.Namespace) -> int:
    from clippilot.config import Settings
    settings = Settings.load()
    if a.action == "list":
        print("\nConfiguration Settings:")
        for k, v in settings.to_dict().items():
            print(f"  {k}: {v}")
    elif a.action == "get":
        if not a.key:
            print("Error: --key is required for get action", file=sys.stderr)
            return 1
        val = getattr(settings, a.key, None)
        if val is None:
            print(f"Error: key '{a.key}' not found", file=sys.stderr)
            return 1
        print(f"{a.key}: {val}")
    elif a.action == "set":
        if not a.key or a.value is None:
            print("Error: --key and --value are required for set action", file=sys.stderr)
            return 1
        orig = getattr(settings, a.key, None)
        val = a.value
        if orig is not None:
            if isinstance(orig, bool):
                val = val.lower() in ("true", "yes", "1")
            elif isinstance(orig, int):
                val = int(val)
            elif isinstance(orig, float):
                val = float(val)
        setattr(settings, a.key, val)
        settings.save()
        print(f"✔️ Set '{a.key}' to '{val}'. Saved config.")
    return 0


def cmd_list(a: argparse.Namespace) -> int:
    q = _queue()
    jobs = q.list(JobStatus(a.status) if a.status else None, limit=a.limit)
    if not jobs:
        print("(no jobs)")
        return 0
    for j in jobs:
        _print_job(q, j)
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    q = _queue()
    job = q.get(a.id)
    if not job:
        print(f"no job {a.id}", file=sys.stderr)
        return 1
    _print_job(q, job)
    print(f"  source: {job.source_ref}")
    print(f"  payload keys: {list(job.payload.keys())}")
    print("  events:")
    for e in q.events(a.id):
        print(f"    {e['ts']}  {e['kind']:<18} {e['detail'] or ''}")
    return 0


def cmd_approve(a: argparse.Namespace) -> int:
    q = _queue()
    job = q.approve(a.id)
    print(f"Approved job #{job.id} → now PENDING at stage {job.stage.value}")
    return 0


def cmd_reject(a: argparse.Namespace) -> int:
    q = _queue()
    job = q.reject(a.id, a.reason or "")
    print(f"Rejected job #{job.id}")
    return 0


def cmd_strike(a: argparse.Namespace) -> int:
    q = _queue()
    ch = q.record_strike(a.channel, a.platform)
    print(f"Channel '{ch.channel}': strikes={ch.strikes} paused={ch.paused}")
    if ch.paused:
        print("  ⛔ channel auto-paused by strike guardrail (termination = 3 strikes / 90 days).")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    q = _queue()
    print(f"Status counts: {q.counts()}")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    from .doctor import check_readiness, format_report
    print(format_report(check_readiness()))
    return 0


def cmd_demo_short(a: argparse.Namespace) -> int:
    """Generate a full Section-B short from a topic (b-roll + narration + karaoke
    captions) and print the finished file so you can watch it."""
    import os

    from .engine import Engine
    from .ui.controller import AppController

    q = _queue()
    prof: dict = {}
    if getattr(a, "facts", 0):
        prof.update(mode="facts", facts_count=int(a.facts))
    if getattr(a, "skin", None):
        prof["caption_skin"] = a.skin
    payload = {"profile": prof} if prof else None
    job = q.enqueue(a.topic, section=Section.FACELESS_FUNNEL, rights_tag=RightsTag.OWNED,
                    payload=payload)
    kind = f"{a.facts}-facts " if getattr(a, "facts", 0) else ""
    print(f"Generating a {kind}short for {a.topic!r} — TTS + b-roll + whisper + karaoke captions "
          f"(~30-60s, first run downloads a small whisper model)…")
    Engine(q).drain(max_steps=50)
    job = q.get(job.id)

    ctrl = AppController(q)
    clips = ctrl.clips_for(job)
    ing = job.payload.get("ingest") or {}
    print(f"\n  status: {job.status.value}")
    print(f"  visual: {ing.get('visual', '-')}")
    print(f"  script: {(ing.get('script') or '')[:140]}")
    if clips:
        path = os.path.abspath(clips[0])
        print(f"\n[OK] Your short is ready — open it:\n  {path}")
        if getattr(a, "open", False):
            try:
                os.startfile(path)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
    else:
        reason = job.error or ing.get("reason") or "see status above"
        print(f"\n[!] No clip was produced (reason: {reason}). "
              f"Run `python -m clippilot doctor` to check ffmpeg/whisper/TTS.")
    return 0


def cmd_demo(_: argparse.Namespace) -> int:
    """Self-contained smoke test of the full loop using an in-memory DB."""
    conn = dbmod.connect(":memory:")
    q = JobQueue(conn, cfg.Settings())  # default settings: approval gate ON
    eng = Engine(q)

    print("1) enqueue a Section-A job from an owned source")
    job = q.enqueue("demo_podcast.mp4", section=Section.PAID_CLIPPING,
                    rights_tag=RightsTag.OWNED, channel="demo_yt")
    print("2) drain — should run ingest→...→compose, then park at the approval gate")
    eng.drain()
    job = q.get(job.id)
    _print_job(q, job)
    assert job.status == JobStatus.BLOCKED_APPROVAL, job.status

    print("3) approve, then drain — should publish and finish")
    q.approve(job.id)
    eng.drain()
    job = q.get(job.id)
    _print_job(q, job)
    assert job.status == JobStatus.DONE, job.status

    print("4) strike the channel twice — guardrail should auto-pause it")
    q.record_strike("demo_yt"); ch = q.record_strike("demo_yt")
    print(f"   channel paused={ch.paused}")
    assert ch.paused

    print("\n✅ demo passed: state machine + approval gate + strike guardrail all work.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clippilot", description="ClipPilot Phase 0 engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create data dir + db").set_defaults(func=cmd_init)

    e = sub.add_parser("enqueue", help="add a job")
    e.add_argument("--source", required=True)
    e.add_argument("--section", default="A", choices=[s.value for s in Section])
    e.add_argument("--rights", default="owned", choices=[r.value for r in RightsTag])
    e.add_argument("--channel", default=None)
    e.add_argument("--key", default=None, help="idempotency key")
    e.set_defaults(func=cmd_enqueue)

    ef = sub.add_parser("enqueue-folder", help="enqueue every video in a folder (DFY batch)")
    ef.add_argument("--folder", required=True)
    ef.add_argument("--section", default="A", choices=[s.value for s in Section])
    ef.add_argument("--rights", default="owned", choices=[r.value for r in RightsTag])
    ef.add_argument("--channel", default=None)
    ef.add_argument("--no-recursive", action="store_true", help="don't descend into subfolders")
    ef.set_defaults(func=cmd_enqueue_folder)

    sub.add_parser("profile-list", help="list DFY templates").set_defaults(func=cmd_profile_list)

    ps = sub.add_parser("profile-save", help="create/update a DFY template (per-client preset)")
    ps.add_argument("--name", required=True)
    ps.add_argument("--section", default="A", choices=[s.value for s in Section])
    ps.add_argument("--rights", default="owned", choices=[r.value for r in RightsTag])
    ps.add_argument("--channel", default=None)
    ps.add_argument("--voice", default=None, help="edge-tts narration voice for Section B/C (Chatterbox uses its own/synthetic voice)")
    ps.add_argument("--bgm", default=None, help="cleared music file for generated Section B/C")
    ps.add_argument("--seconds", type=int, default=35, help="target length for generated shorts")
    ps.add_argument("--mode", default="standard",
                    choices=["standard", "facts", "narrative", "motion", "montage", "auto"],
                    help="script engine: facts=N-facts; narrative/motion/montage=genre router; auto=classify")
    ps.add_argument("--facts-count", dest="facts_count", type=int, default=5,
                    help="number of facts when --mode facts")
    ps.add_argument("--skin", default=None, choices=_skin_choices(),
                    help="caption skin (default: settings/karaoke_yellow)")
    ps.set_defaults(func=cmd_profile_save)

    pd = sub.add_parser("profile-delete", help="delete a DFY template")
    pd.add_argument("--name", required=True)
    pd.set_defaults(func=cmd_profile_delete)

    ep = sub.add_parser("enqueue-profile", help="enqueue a source/topic using a DFY template")
    ep.add_argument("--source", required=True)
    ep.add_argument("--profile", required=True)
    ep.set_defaults(func=cmd_enqueue_profile)

    pr = sub.add_parser("publish-reel", help="publish a Reel to Instagram from a public video URL")
    pr.add_argument("--url", required=True)
    pr.add_argument("--caption", default="")
    pr.set_defaults(func=cmd_publish_reel)

    rs = sub.add_parser("resync-srt", help="auto-align an SRT to a media file's audio (ffsubsync)")
    rs.add_argument("--srt", required=True)
    rs.add_argument("--audio", required=True)
    rs.add_argument("--out", default=None)
    rs.set_defaults(func=cmd_resync_srt)

    dl = sub.add_parser("download", help="download a source video from a URL (yt-dlp; rights-gated)")
    dl.add_argument("--url", required=True)
    dl.add_argument("--out", default=None)
    dl.add_argument("--max-height", dest="max_height", type=int, default=1080)
    dl.set_defaults(func=cmd_download)

    eu = sub.add_parser("enqueue-url", help="download a URL + enqueue it as a clipping job")
    eu.add_argument("--url", required=True)
    eu.add_argument("--section", default="A", choices=[s.value for s in Section])
    eu.add_argument("--rights", default="owned", choices=[r.value for r in RightsTag])
    eu.add_argument("--channel", default=None)
    eu.add_argument("--max-height", dest="max_height", type=int, default=1080)
    eu.set_defaults(func=cmd_enqueue_url)

    co = sub.add_parser("cutout", help="cut a subject out of an image + composite over a background (rembg)")
    co.add_argument("--image", required=True)
    co.add_argument("--bg", required=True)
    co.add_argument("--out", required=True)
    co.set_defaults(func=cmd_cutout)

    er = sub.add_parser("editor-render", help="composite a timeline-editor project (JSON) to mp4")
    er.add_argument("--project", required=True)
    er.add_argument("--out", required=True)
    er.set_defaults(func=cmd_editor_render)

    ex = sub.add_parser("explainer", help="render an animated explainer/diagram card clip")
    ex.add_argument("--title", required=True)
    ex.add_argument("--bullets", default="", help="semicolon-separated bullet points")
    ex.add_argument("--out", required=True)
    ex.add_argument("--seconds", type=float, default=5.0)
    ex.set_defaults(func=cmd_explainer)

    r = sub.add_parser("run", help="run the production pipeline or legacy queue engine")
    r.add_argument("--legacy", action="store_true", help="run legacy queue drain mode")
    r.add_argument("--max", type=int, default=1000, help="max steps to drain in legacy mode")
    r.add_argument("--workspace", default=".", help="workspace directory path")
    r.add_argument("--variation", default="variation", help="variation directory path")
    r.add_argument("--date", default="2026-07-03", help="date string for the run")
    r.add_argument("--slug", default="daily_006", help="default slug prefix")
    r.set_defaults(func=cmd_run)

    rnd = sub.add_parser("render", help="compile and render the Remotion video composition only")
    rnd.add_argument("--workspace", default=".", help="workspace directory path")
    rnd.add_argument("--variation", default="variation", help="variation directory path")
    rnd.add_argument("--date", default="2026-07-03", help="date string for the run")
    rnd.add_argument("--slug", default="daily_006", help="default slug prefix")
    rnd.set_defaults(func=cmd_render)

    lrn = sub.add_parser("learn", help="run the Learning Engine to generate performance analysis summary")
    lrn.add_argument("--workspace", default=".", help="workspace directory path")
    lrn.set_defaults(func=cmd_learn)

    anl = sub.add_parser("analytics", help="import YouTube performance analytics and update records")
    anl.add_argument("--workspace", default=".", help="workspace directory path")
    anl.set_defaults(func=cmd_analytics)

    strg = sub.add_parser("strategy", help="output the next Strategy Engine decision configuration")
    strg.add_argument("--workspace", default=".", help="workspace directory path")
    strg.add_argument("--variation", default="variation", help="variation directory path")
    strg.add_argument("--date", default="2026-07-03", help="date string for the run")
    strg.add_argument("--slug", default="daily_006", help="default slug prefix")
    strg.set_defaults(func=cmd_strategy)

    hist = sub.add_parser("history", help="display recorded video performance history logs")
    hist.add_argument("--workspace", default=".", help="workspace directory path")
    hist.set_defaults(func=cmd_history)

    cfg_cmd = sub.add_parser("config", help="view and update engine settings configuration")
    cfg_cmd.add_argument("action", choices=["list", "get", "set"], help="config action to perform")
    cfg_cmd.add_argument("--key", default=None, help="settings key name")
    cfg_cmd.add_argument("--value", default=None, help="value to set the key to")
    cfg_cmd.set_defaults(func=cmd_config)

    ls = sub.add_parser("list", help="list jobs")
    ls.add_argument("--status", default=None, choices=[s.value for s in JobStatus])
    ls.add_argument("--limit", type=int, default=100)
    ls.set_defaults(func=cmd_list)

    sh = sub.add_parser("show", help="show a job + its events")
    sh.add_argument("id", type=int)
    sh.set_defaults(func=cmd_show)

    ap = sub.add_parser("approve", help="approve a job at the gate")
    ap.add_argument("id", type=int)
    ap.set_defaults(func=cmd_approve)

    rj = sub.add_parser("reject", help="reject a job at the gate")
    rj.add_argument("id", type=int)
    rj.add_argument("--reason", default="")
    rj.set_defaults(func=cmd_reject)

    st = sub.add_parser("strike", help="record a copyright strike on a channel")
    st.add_argument("channel")
    st.add_argument("--platform", default="youtube")
    st.set_defaults(func=cmd_strike)

    sub.add_parser("status", help="status counts").set_defaults(func=cmd_status)
    sub.add_parser("doctor", help="readiness check + next steps to first $").set_defaults(func=cmd_doctor)

    dsh = sub.add_parser("demo-short", help="generate a full Section-B short from a topic and show the file")
    dsh.add_argument("topic")
    dsh.add_argument("--open", action="store_true", help="open the finished video when done")
    dsh.add_argument("--facts", type=int, default=0, metavar="N",
                     help="make an 'N facts about <topic>' short (ShortGPT facts engine)")
    dsh.add_argument("--skin", default=None, choices=_skin_choices(),
                     help="caption skin (opaque_box, kinetic_pop, neon_pop, …)")
    dsh.set_defaults(func=cmd_demo_short)

    sub.add_parser("demo", help="run a self-contained smoke test").set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    # Windows consoles default to cp1252; force UTF-8 so arrows/emoji in output
    # don't raise UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError) as exc:  # ordinary user errors → clean message, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
