"""Scheduler: run the full pipeline on an interval, and react to new tracked URLs.

Uses APScheduler's BlockingScheduler at ``run_interval_hours`` (default 6) from
tracked_urls.yaml. A watchdog observer on tracked_urls.yaml runs non-interactive
recon for any newly added URLs between scheduled ticks. All scheduled work is
non-interactive (recon NEEDS_ATTENTION URLs are skipped + logged, never blocking).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler

from agent.recon_agent import config_path
from pipeline.runner import Runner, load_tracked_urls

logger = logging.getLogger(__name__)


class _TrackedUrlsHandler:
    """watchdog handler: on tracked_urls.yaml change, recon any new URLs."""

    def __init__(self, scheduler: "PipelineScheduler"):
        self.scheduler = scheduler

    # watchdog calls dispatch(event); we only care about our file being modified
    def dispatch(self, event):
        try:
            src = getattr(event, "src_path", "") or ""
            if src.replace("\\", "/").endswith("config/tracked_urls.yaml"):
                self.scheduler.on_tracked_urls_changed()
        except Exception as exc:
            logger.warning("watchdog dispatch error: %s", exc)


class PipelineScheduler:
    def __init__(self, runner: Optional[Runner] = None):
        self.runner = runner or Runner()
        self._known_urls: set[str] = set()
        self._observer = None

    # ------------------------------------------------------------------- tick
    def _tick(self) -> None:
        cfg = load_tracked_urls()
        entries = cfg.get("companies", []) or []
        self._known_urls = {e.get("url") for e in entries if e.get("url")}
        logger.info("Scheduled run starting for %d companies", len(entries))
        self.runner.run(entries)

    def on_tracked_urls_changed(self) -> None:
        cfg = load_tracked_urls()
        entries = cfg.get("companies", []) or []
        new_entries = [e for e in entries if e.get("url") and e["url"] not in self._known_urls]
        if not new_entries:
            return
        logger.info("Detected %d new tracked URL(s); running recon+scrape", len(new_entries))
        for e in new_entries:
            self._known_urls.add(e["url"])
        self.runner.run(new_entries, interactive=False)

    # ------------------------------------------------------------------ start
    def start(self, run_immediately: bool = True) -> None:
        cfg = load_tracked_urls()
        hours = float(cfg.get("run_interval_hours", 6) or 6)
        self._known_urls = {e.get("url") for e in cfg.get("companies", []) or [] if e.get("url")}

        self._start_watchdog()

        scheduler = BlockingScheduler()
        scheduler.add_job(
            self._tick,
            "interval",
            hours=hours,
            next_run_time=datetime.now() if run_immediately else None,
            id="pipeline",
            max_instances=1,
            coalesce=True,
        )
        logger.info("Scheduler started: every %s hour(s)", hours)
        self.runner.console.print(f"[bold]Scheduler running[/] — every {hours} hour(s). Ctrl+C to stop.")
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")
        finally:
            self._stop_watchdog()

    def _start_watchdog(self) -> None:
        try:
            from watchdog.observers import Observer

            observer = Observer()
            observer.schedule(_TrackedUrlsHandler(self), config_path(""), recursive=False)
            observer.start()
            self._observer = observer
            logger.info("watchdog watching tracked_urls.yaml")
        except Exception as exc:
            logger.warning("Could not start watchdog (continuing without it): %s", exc)

    def _stop_watchdog(self) -> None:
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception:
                pass
